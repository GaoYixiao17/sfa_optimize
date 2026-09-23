#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bench_li.py — LightningIndexer (v1) + LightningIndexerV2 性能基准

场景: A5 (Ascend 950), B=32/64, S1=S2=128, N1=64, D=128

用法:
    python bench_li.py                 # v1 + v2 全矩阵
    BENCH_OPS=v1 python bench_li.py    # 仅 v1
    BENCH_OPS=v2 python bench_li.py    # 仅 v2

计时口径:
    v1: 单次 npu_lightning_indexer 调用 (含host侧tiling+launch)
    v2: (a) metadata+算子 两次调用的真实时延  (b) 仅算子(metadata预生成)
"""

import os
import sys

import torch
import torch_npu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common import (  # noqa: E402
    BenchConfig,
    ResultCollector,
    make_li_inputs,
    time_callable,
    env_list,
    env_bool,
)

MAX_INT64 = (1 << 63) - 1


def run_li_v1(cfg: BenchConfig, collector: ResultCollector):
    for batch in cfg.batch_sizes:
        for topk in cfg.li_topk:
            for layout, layout_key in [(cfg.li_layout, cfg.li_layout_key)]:
                inputs = make_li_inputs(
                    cfg, batch, topk,
                    layout=layout, layout_key=layout_key,
                    dtype=cfg.dtype("li"),
                    weight_fp32=cfg.li_weight_fp32,
                )

                def call():
                    return torch_npu.npu_lightning_indexer(
                        inputs["query"],
                        inputs["key"],
                        inputs["weights"],
                        actual_seq_lengths_query=inputs["actual_seq_lengths_query"],
                        actual_seq_lengths_key=inputs["actual_seq_lengths_key"],
                        block_table=inputs["block_table"],
                        layout_query=layout,
                        layout_key=layout_key,
                        sparse_count=topk,
                        sparse_mode=cfg.li_sparse_mode,
                        pre_tokens=MAX_INT64,
                        next_tokens=MAX_INT64,
                    )

                # 正确性冒烟: 跑一次看输出shape
                out = call()
                torch.npu.synchronize()
                case = (
                    f"B={batch} S={cfg.seq_len} N={cfg.li_heads} D={cfg.li_head_dim} "
                    f"{cfg.li_dtype} topk={topk} mode={cfg.li_sparse_mode} "
                    f"Q:{layout} K:{layout_key}"
                )
                stats = time_callable(call, warmup=cfg.warmup, iters=cfg.iters)
                collector.add("LightningIndexer", case, stats,
                              extra={"out_shape": list(out.shape) if out is not None else None})


def run_li_v2(cfg: BenchConfig, collector: ResultCollector):
    try:
        from cann_ops_transformer.ops import lightning_indexer, lightning_indexer_metadata
    except Exception as e:  # noqa: BLE001
        print(f"[skip] LightningIndexerV2 需要 cann_ops_transformer 包: {e}")
        return

    for batch in cfg.batch_sizes:
        for topk in cfg.li_topk:
            inputs = make_li_inputs(
                cfg, batch, topk,
                layout=cfg.li_layout, layout_key=cfg.li_layout_key,
                dtype=cfg.dtype("li"),
                weight_fp32=cfg.li_weight_fp32,
            )
            q, k, w = inputs["query"], inputs["key"], inputs["weights"]

            # 950 上 v2 必须先调用 metadata 算子
            metadata = lightning_indexer_metadata(
                cfg.li_heads, 1, cfg.li_head_dim, topk,
                batch_size=batch,
                max_seqlen_q=cfg.seq_len,
                max_seqlen_k=cfg.seq_len,
                layout_q=cfg.li_layout,
                layout_k=cfg.li_layout_key,
                mask_mode=cfg.li_sparse_mode,
            )
            torch.npu.synchronize()

            kwargs = dict(
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                seqused_q=inputs["actual_seq_lengths_query"],
                seqused_k=inputs["actual_seq_lengths_key"],
                cmp_residual_k=None,
                block_table=inputs["block_table"],
                output_idx_offset=None,
                metadata=metadata,
                max_seqlen_q=cfg.seq_len,
                layout_q=cfg.li_layout,
                layout_k=cfg.li_layout_key,
                mask_mode=cfg.li_sparse_mode,
                cmp_ratio=1,
                return_value=1,
            )

            def call_op_only():
                return lightning_indexer(q, k, w, topk, **kwargs)

            def call_with_metadata():
                md = lightning_indexer_metadata(
                    cfg.li_heads, 1, cfg.li_head_dim, topk,
                    batch_size=batch,
                    max_seqlen_q=cfg.seq_len,
                    max_seqlen_k=cfg.seq_len,
                    layout_q=cfg.li_layout,
                    layout_k=cfg.li_layout_key,
                    mask_mode=cfg.li_sparse_mode,
                )
                return lightning_indexer(q, k, w, topk, **{**kwargs, "metadata": md})

            out = call_op_only()
            torch.npu.synchronize()
            base_case = (
                f"B={batch} S={cfg.seq_len} N={cfg.li_heads} {cfg.li_dtype} topk={topk} "
                f"mask={cfg.li_sparse_mode} Q:{cfg.li_layout} K:{cfg.li_layout_key}"
            )
            stats = time_callable(call_op_only, warmup=cfg.warmup, iters=cfg.iters)
            collector.add("LightningIndexerV2", base_case + " [op only]", stats,
                          extra={"out_shape": list(out[0].shape) if isinstance(out, tuple) else list(out.shape)})
            stats = time_callable(call_with_metadata, warmup=cfg.warmup, iters=cfg.iters)
            collector.add("LightningIndexerV2", base_case + " [op+metadata]", stats)


def main():
    cfg = BenchConfig()
    ops = env_list("BENCH_OPS", ["v1", "v2"])
    collector = ResultCollector(cfg.label, cfg.out_json or f"result_li_{cfg.label}.json")
    print(f"=== LightningIndexer benchmark, label={cfg.label}, ops={ops} ===")
    print(f"    B={cfg.batch_sizes} S={cfg.seq_len} topk={cfg.li_topk} "
          f"dtype={cfg.li_dtype} layout Q:{cfg.li_layout} K:{cfg.li_layout_key}")
    if "v1" in ops:
        run_li_v1(cfg, collector)
    if "v2" in ops:
        run_li_v2(cfg, collector)
    collector.save()


if __name__ == "__main__":
    main()
