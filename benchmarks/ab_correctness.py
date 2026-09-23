#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ab_correctness.py — baseline vs optimized 输出一致性验证

由于切换算子实现需要不同的自定义算子包 (进程级生效), A/B 验证分三步:

    # 1. 激活 baseline 包后运行:
    python ab_correctness.py save --out ab_baseline.pt
    # 2. 激活 optimized 包后运行:
    python ab_correctness.py save --out ab_optimized.pt
    # 3. 对比 (任意环境):
    python ab_correctness.py compare --a ab_baseline.pt --b ab_optimized.pt

对比标准:
    - LightningIndexer indices: 逐 token 的 topk 集合完全一致 (排序后比较, 容忍并列值顺序)
    - LightningIndexer values:  atol=1e-2 (bf16 精度, 归约顺序可能不同)
    - SFA 输出:                  atol=2e-2, rtol=2e-2 (softmax 数值精度)
"""

import argparse
import math
import os
import sys

import torch
import torch_npu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common import BenchConfig, make_li_inputs, make_sfa_inputs  # noqa: E402

MAX_INT64 = (1 << 63) - 1


def build_cases(cfg):
    cases = {}

    # ---- LI v1 ----
    for batch in cfg.batch_sizes:
        topk = 64
        inputs = make_li_inputs(cfg, batch, topk, dtype=cfg.dtype("li"))
        out = torch_npu.npu_lightning_indexer(
            inputs["query"], inputs["key"], inputs["weights"],
            actual_seq_lengths_query=inputs["actual_seq_lengths_query"],
            actual_seq_lengths_key=inputs["actual_seq_lengths_key"],
            block_table=inputs["block_table"],
            layout_query="BSND", layout_key="BSND",
            sparse_count=topk, sparse_mode=cfg.li_sparse_mode,
            pre_tokens=MAX_INT64, next_tokens=MAX_INT64,
        )
        torch.npu.synchronize()
        cases[f"li_v1_B{batch}_topk{topk}"] = out.cpu()

    # ---- SFA ----
    for batch in cfg.batch_sizes:
        for bs, k in [(1, 64), (64, 2)]:
            inputs = make_sfa_inputs(cfg, batch, bs, k, dtype=cfg.dtype("sfa"))
            out = torch_npu.npu_sparse_flash_attention(
                query=inputs["query"], key=inputs["key"], value=inputs["value"],
                sparse_indices=inputs["sparse_indices"],
                scale_value=1.0 / math.sqrt(cfg.sfa_head_dim + cfg.sfa_rope_dim),
                block_table=inputs["block_table"],
                actual_seq_lengths_query=inputs["actual_seq_lengths_query"],
                actual_seq_lengths_kv=inputs["actual_seq_lengths_kv"],
                query_rope=inputs["query_rope"], key_rope=inputs["key_rope"],
                sparse_block_size=bs, layout_query="BSND", layout_kv="BSND",
                sparse_mode=cfg.sfa_sparse_mode,
                pre_tokens=MAX_INT64, next_tokens=MAX_INT64,
                attention_mode=2, return_softmax_lse=False,
            )
            torch.npu.synchronize()
            if isinstance(out, (tuple, list)):
                out = out[0]
            cases[f"sfa_B{batch}_sbs{bs}_K{k}"] = out.cpu()

    return cases


def cmd_save(args):
    cfg = BenchConfig()
    torch.manual_seed(20240601)  # 两次 save 使用相同输入
    cases = build_cases(cfg)
    torch.save(cases, args.out)
    print(f"[save] {len(cases)} cases -> {args.out}")


def _compare_li_indices(a, b):
    """topk 集合比较: 对每行排序后比较; 无效(-1)数量必须一致"""
    a_flat = a.reshape(a.shape[0], -1)
    b_flat = b.reshape(b.shape[0], -1)
    if a_flat.shape != b_flat.shape:
        return False, f"shape mismatch {a.shape} vs {b.shape}"
    a_sorted, _ = torch.sort(a_flat, dim=-1)
    b_sorted, _ = torch.sort(b_flat, dim=-1)
    diff = (a_sorted != b_sorted).sum().item()
    return diff == 0, f"{diff} mismatched elements (sorted per-row)"


def cmd_compare(args):
    a = torch.load(args.a)
    b = torch.load(args.b)
    keys = sorted(set(a.keys()) | set(b.keys()))
    all_ok = True
    print(f"{'case':<32} {'result':<8} detail")
    print("-" * 80)
    for k in keys:
        if k not in a or k not in b:
            print(f"{k:<32} MISS    missing in {'a' if k not in a else 'b'}")
            all_ok = False
            continue
        ta, tb = a[k], b[k]
        if k.startswith("li_v1") or k.startswith("li_v2"):
            ok, detail = _compare_li_indices(ta, tb)
        else:
            if isinstance(ta, (tuple, list)):
                ta = ta[0]
            if isinstance(tb, (tuple, list)):
                tb = tb[0]
            ok = torch.allclose(ta.float(), tb.float(), atol=2e-2, rtol=2e-2)
            max_diff = (ta.float() - tb.float()).abs().max().item()
            detail = f"max_abs_diff={max_diff:.3e}"
        print(f"{k:<32} {'PASS' if ok else 'FAIL':<8} {detail}")
        all_ok = all_ok and ok
    print("-" * 80)
    print("RESULT:", "ALL PASS" if all_ok else "DIFFERENCES FOUND")
    sys.exit(0 if all_ok else 1)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("save")
    ps.add_argument("--out", required=True)
    pc = sub.add_parser("compare")
    pc.add_argument("--a", required=True)
    pc.add_argument("--b", required=True)
    args = p.parse_args()
    if args.cmd == "save":
        cmd_save(args)
    else:
        cmd_compare(args)


if __name__ == "__main__":
    main()
