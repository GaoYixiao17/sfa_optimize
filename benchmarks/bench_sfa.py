#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bench_sfa.py — SparseFlashAttention (MLA-absorb) 性能基准

场景: A5 (Ascend 950), B=32/64, S1=S2=128, N1=64, N2=1, D=512, Dr=64

用法:
    python bench_sfa.py
    BENCH_SFA_CASES=64:1,128:1,2:64 python bench_sfa.py   # K:sbs 显式配对
"""

import math
import os
import sys

import torch
import torch_npu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common import BenchConfig, ResultCollector, env_list, make_sfa_inputs, time_callable  # noqa: E402

MAX_INT64 = (1 << 63) - 1


def run_sfa(cfg: BenchConfig, collector: ResultCollector):
    # 用例为 "K:sbs" 显式配对 (K=sparse_size, sbs=sparse_block_size):
    # A5 上 sbs=1 为 token 粒度 (K=64/128), sbs=64 为 block 粒度 (K=2)
    cases = env_list("BENCH_SFA_CASES", ["64:1", "128:1", "2:64"])
    for batch in cfg.batch_sizes:
        for case in cases:
            k_str, bs_str = case.split(":")
            bs = int(bs_str)
            k = int(k_str)
            inputs = make_sfa_inputs(
                cfg, batch, bs, k,
                layout_q=cfg.sfa_layout_q, layout_kv=cfg.sfa_layout_kv,
                dtype=cfg.dtype("sfa"),
            )

                def call():
                    return torch_npu.npu_sparse_flash_attention(
                        query=inputs["query"],
                        key=inputs["key"],
                        value=inputs["value"],
                        sparse_indices=inputs["sparse_indices"],
                        scale_value=1.0 / math.sqrt(cfg.sfa_head_dim + cfg.sfa_rope_dim),
                        block_table=inputs["block_table"],
                        actual_seq_lengths_query=inputs["actual_seq_lengths_query"],
                        actual_seq_lengths_kv=inputs["actual_seq_lengths_kv"],
                        query_rope=inputs["query_rope"],
                        key_rope=inputs["key_rope"],
                        sparse_block_size=bs,
                        layout_query=cfg.sfa_layout_q,
                        layout_kv=cfg.sfa_layout_kv,
                        sparse_mode=cfg.sfa_sparse_mode,
                        pre_tokens=MAX_INT64,
                        next_tokens=MAX_INT64,
                        attention_mode=2,  # MLA-absorb
                        return_softmax_lse=cfg.sfa_return_lse,
                    )

                out = call()
                torch.npu.synchronize()
                case = (
                    f"B={batch} S={cfg.seq_len} N={cfg.sfa_heads} D={cfg.sfa_head_dim}+{cfg.sfa_rope_dim} "
                    f"{cfg.sfa_dtype} sbs={bs} K={k} mode={cfg.sfa_sparse_mode} "
                    f"Q:{cfg.sfa_layout_q} KV:{cfg.sfa_layout_kv}"
                )
                stats = time_callable(call, warmup=cfg.warmup, iters=cfg.iters)
                out_shape = list(out[0].shape) if isinstance(out, (tuple, list)) else list(out.shape)
                collector.add("SparseFlashAttention", case, stats, extra={"out_shape": out_shape})


def main():
    cfg = BenchConfig()
    collector = ResultCollector(cfg.label, cfg.out_json or f"result_sfa_{cfg.label}.json")
    print(f"=== SparseFlashAttention benchmark, label={cfg.label} ===")
    print(f"    B={cfg.batch_sizes} S={cfg.seq_len} K:sbs={env_list('BENCH_SFA_CASES', ['64:1', '128:1', '2:64'])} "
          f"dtype={cfg.sfa_dtype}")
    run_sfa(cfg, collector)
    collector.save()


if __name__ == "__main__":
    main()
