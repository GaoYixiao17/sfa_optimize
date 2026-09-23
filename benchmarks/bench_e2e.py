#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bench_e2e.py — DSA 端到端链路基准: LightningIndexer → 索引转换 → SparseFlashAttention

模拟真实 serving 的一步 prefill (短序列):
    LI v1 topk 选 token -> 转成 block 级稀疏索引 -> SFA (MLA-absorb) 注意力

转换逻辑 (与 vLLM-Ascend DSA 实现一致):
    token_idx // sparse_block_size -> 每行去重 -> 有效在前, -1 填充尾部
    (注意 causal: 每行需包含当前 token 所在块)

用法:
    python bench_e2e.py
"""

import math
import os
import sys

import torch
import torch_npu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common import BenchConfig, ResultCollector, make_li_inputs, make_sfa_inputs, time_callable  # noqa: E402

MAX_INT64 = (1 << 63) - 1


def token_indices_to_blocks(token_idx: torch.Tensor, sbs: int, kv_len: int):
    """
    token_idx: [rows, topk] int32, 无效为 -1
    返回: [rows, K] int32 block 索引, 每行有效在前, 含当前行所在块, 尾部 -1
    """
    rows = token_idx.shape[0]
    valid = token_idx >= 0
    blocks = torch.where(valid, token_idx // sbs, torch.tensor(-1, dtype=token_idx.dtype, device=token_idx.device))
    # 每行唯一化(保序) + 排序, 用 sort 然后 mask 掉重复
    blocks = torch.where(valid, blocks, torch.iinfo(torch.int32).max)  # 无效排到最后
    blocks, _ = torch.sort(blocks, dim=-1)
    # 去重: 保留每个连续段第一个
    dup = torch.zeros_like(blocks, dtype=torch.bool)
    dup[..., 1:] = blocks[..., 1:] == blocks[..., :-1]
    # 无效值(==INT32_MAX)也去掉
    invalid = blocks == torch.iinfo(torch.int32).max
    keep = (~dup) & (~invalid)
    # 压缩: 用 cumsum 得到每行保留个数, 然后scatter到前部
    max_blocks = blocks.shape[-1]
    pos = torch.cumsum(keep.to(torch.int32), dim=-1) - 1
    out = torch.full_like(blocks, -1)
    out.scatter_(-1, torch.where(keep, pos, torch.zeros_like(pos)), torch.where(keep, blocks, torch.zeros_like(blocks)))
    return out


def run_e2e(cfg: BenchConfig, collector: ResultCollector, sbs=64):
    """LI topk=64 -> block(sbs=64) -> SFA"""
    for batch in cfg.batch_sizes:
        topk = min(int(cfg.li_topk[0]), cfg.seq_len)
        # --- LI 输入 (indexer 用 128 维头) ---
        li = make_li_inputs(cfg, batch, topk, layout="BSND", layout_key="BSND",
                            dtype=cfg.dtype("li"))
        # --- SFA 输入 (MLA 576) ---
        sfa = make_sfa_inputs(cfg, batch, sbs, topk, layout_q="BSND", layout_kv="BSND",
                              dtype=cfg.dtype("sfa"))

        def li_call():
            return torch_npu.npu_lightning_indexer(
                li["query"], li["key"], li["weights"],
                actual_seq_lengths_query=li["actual_seq_lengths_query"],
                actual_seq_lengths_key=li["actual_seq_lengths_key"],
                block_table=li["block_table"],
                layout_query="BSND", layout_key="BSND",
                sparse_count=topk, sparse_mode=cfg.li_sparse_mode,
                pre_tokens=MAX_INT64, next_tokens=MAX_INT64,
            )

        def sfa_call(indices):
            return torch_npu.npu_sparse_flash_attention(
                query=sfa["query"], key=sfa["key"], value=sfa["value"],
                sparse_indices=indices,
                scale_value=1.0 / math.sqrt(cfg.sfa_head_dim + cfg.sfa_rope_dim),
                block_table=sfa["block_table"],
                actual_seq_lengths_query=sfa["actual_seq_lengths_query"],
                actual_seq_lengths_kv=sfa["actual_seq_lengths_kv"],
                query_rope=sfa["query_rope"], key_rope=sfa["key_rope"],
                sparse_block_size=sbs, layout_query="BSND", layout_kv="BSND",
                sparse_mode=cfg.sfa_sparse_mode,
                pre_tokens=MAX_INT64, next_tokens=MAX_INT64,
                attention_mode=2, return_softmax_lse=False,
            )

        def full_pipeline():
            sparse_idx, _ = li_call()  # [B, S1, N1, topk] token 索引
            # N2=1: 取第一个 KV head 的行; 转 block 索引
            idx = sparse_idx[:, :, 0, :]  # [B, S1, topk]
            blk = token_indices_to_blocks(idx.reshape(-1, topk), sbs, cfg.seq_len)
            K = blk.shape[-1]
            return sfa_call(blk.view(batch, cfg.seq_len, 1, K))

        out = full_pipeline()
        torch.npu.synchronize()

        case = (
            f"B={batch} S={cfg.seq_len} LI.topk={topk} sbs={sbs} "
            f"{cfg.li_dtype}/{cfg.sfa_dtype} mode={cfg.li_sparse_mode}"
        )
        # 分段计时
        stats = time_callable(li_call, warmup=cfg.warmup, iters=cfg.iters)
        collector.add("e2e.LI", case, stats)
        stats = time_callable(full_pipeline, warmup=cfg.warmup, iters=cfg.iters)
        collector.add("e2e.full", case, stats, extra={"out_shape": list(out[0].shape) if isinstance(out, (tuple, list)) else list(out.shape)})


def main():
    cfg = BenchConfig()
    collector = ResultCollector(cfg.label, cfg.out_json or f"result_e2e_{cfg.label}.json")
    print(f"=== DSA e2e benchmark (LI -> blocks -> SFA), label={cfg.label} ===")
    run_e2e(cfg, collector)
    collector.save()


if __name__ == "__main__":
    main()
