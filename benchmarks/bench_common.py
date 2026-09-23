#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bench_common.py — A5 (Ascend 950) 算子对比测试公共工具

目标场景: bs=32/64, seq_len=128 的 DSA (DeepSeek Sparse Attention) 调用栈
  - LightningIndexer / LightningIndexerV2 : 稀疏索引算子 (top-k 选token)
  - SparseFlashAttention                  : MLA-absorb 稀疏注意力

计时方式:
  1. torch.npu.Event 计时 (默认)          —— 端到端 launch+kernel 时间
  2. 可选 torch_npu.profiler              —— kernel 级耗时分解 (--profile)

用法: 见 run_all.sh / README.md
"""

import json
import os
import statistics
import time
from dataclasses import dataclass, field, asdict

import torch
import torch_npu  # noqa: F401

torch.manual_seed(1234)

# ---------------------------------------------------------------------------
# 环境变量解析
# ---------------------------------------------------------------------------


def env_list(name, default):
    v = os.environ.get(name)
    if v is None:
        return default
    return [x.strip() for x in v.split(",") if x.strip() != ""]


def env_int(name, default):
    return int(os.environ.get(name, default))


def env_bool(name, default=False):
    v = os.environ.get(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "on")


@dataclass
class BenchConfig:
    """一次基准测试的可调参数(均可用环境变量覆盖)"""

    # 形状
    batch_sizes: list = field(default_factory=lambda: env_list("BENCH_BS", ["32", "64"]))
    seq_len: int = env_int("BENCH_SEQ", 128)
    # LightningIndexer 参数
    li_heads: int = env_int("BENCH_LI_HEADS", 64)     # N1 (g), K_N 恒为 1
    li_head_dim: int = env_int("BENCH_LI_HEAD_DIM", 128)
    li_topk: list = field(default_factory=lambda: env_list("BENCH_LI_TOPK", ["64", "2048"]))
    li_sparse_mode: int = env_int("BENCH_LI_SPARSE_MODE", 3)
    li_layout: str = os.environ.get("BENCH_LI_LAYOUT", "BSND")       # BSND / TND
    li_layout_key: str = os.environ.get("BENCH_LI_LAYOUT_KEY", "BSND")  # BSND / TND / PA_BSND
    li_dtype: str = os.environ.get("BENCH_LI_DTYPE", "bf16")          # bf16 / fp16
    li_weight_fp32: bool = env_bool("BENCH_LI_WEIGHT_FP32", False)
    # SFA 参数 (MLA-absorb)
    sfa_heads: int = env_int("BENCH_SFA_HEADS", 64)    # N1
    sfa_head_dim: int = env_int("BENCH_SFA_HEAD_DIM", 512)
    sfa_rope_dim: int = env_int("BENCH_SFA_ROPE_DIM", 64)
    sfa_sparse_block_size: list = field(
        default_factory=lambda: env_list("BENCH_SFA_BLOCK_SIZE", ["1", "64"])
    )  # sparse_block_size 属性 (token粒度=1 / block粒度=64)
    sfa_sparse_size: list = field(default_factory=lambda: env_list("BENCH_SFA_K", ["64", "128"]))
    sfa_sparse_mode: int = env_int("BENCH_SFA_SPARSE_MODE", 3)
    sfa_layout_q: str = os.environ.get("BENCH_SFA_LAYOUT_Q", "BSND")
    sfa_layout_kv: str = os.environ.get("BENCH_SFA_LAYOUT_KV", "BSND")  # BSND / TND / PA_BSND
    sfa_dtype: str = os.environ.get("BENCH_SFA_DTYPE", "bf16")
    sfa_return_lse: bool = env_bool("BENCH_SFA_RETURN_LSE", False)
    # 计时参数
    warmup: int = env_int("BENCH_WARMUP", 20)
    iters: int = env_int("BENCH_ITERS", 100)
    # 标签: baseline / optimized / builtin 等, 用于区分两次运行
    label: str = os.environ.get("BENCH_LABEL", "default")
    out_json: str = os.environ.get("BENCH_OUT_JSON", "")
    profile: bool = env_bool("BENCH_PROFILE", False)

    def dtype(self, name):
        return {"bf16": torch.bfloat16, "fp16": torch.float16}[getattr(self, f"{name}_dtype")]


# ---------------------------------------------------------------------------
# 计时
# ---------------------------------------------------------------------------


def time_callable(fn, warmup=20, iters=100, sync_each=False):
    """用 NPU event 计时, 返回统计(us).  fn 必须是异步提交到当前 stream 的调用."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    if sync_each:
        # 每次迭代单独同步: 测得的是 launch+kernel 的完整时间, 数值更稳定,
        # 但包含同步开销; 适合小 kernel 的相对对比 (两次运行用同一模式)
        times = []
        for i in range(iters):
            starts[i].record()
            fn()
            ends[i].record()
            torch.npu.synchronize()
            times.append(starts[i].elapsed_time(ends[i]) * 1000.0)  # ms -> us
    else:
        for i in range(iters):
            starts[i].record()
            fn()
            ends[i].record()
        torch.npu.synchronize()
        times = [starts[i].elapsed_time(ends[i]) * 1000.0 for i in range(iters)]
    times.sort()
    return {
        "mean_us": round(statistics.mean(times), 2),
        "median_us": round(statistics.median(times), 2),
        "min_us": round(times[0], 2),
        "p90_us": round(times[int(len(times) * 0.9) - 1 if iters > 10 else -1], 2),
        "max_us": round(times[-1], 2),
    }


def profile_callable(fn, warmup=5, iters=10, out_prefix="prof"):
    """torch_npu profiler 抓 kernel 级耗时, 输出 trace 与摘要(返回kernel统计)."""
    from torch_npu.profiler import profile, ProfilerActivity

    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU]) as prof:
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
    path = f"{out_prefix}_trace.json"
    prof.export_chrome_trace(path)
    print(f"[profile] chrome trace -> {path}")

    try:  # 摘要 kernel 耗时
        events = prof.key_averages()
        rows = []
        for ev in events:
            if getattr(ev, "device_type", None) is not None and "NPU" in str(ev.device_type):
                rows.append((ev.key, ev.count, ev.self_device_time_total))
        rows.sort(key=lambda r: -r[2])
        print(f"[profile] top NPU kernels (name, count, total_us):")
        for r in rows[:20]:
            print("   ", r)
        return rows
    except Exception as e:  # noqa: BLE001
        print(f"[profile] summary parse failed: {e}")
        return None


# ---------------------------------------------------------------------------
# 结果收集
# ---------------------------------------------------------------------------


class ResultCollector:
    def __init__(self, label, out_json=""):
        self.label = label
        self.out_json = out_json
        self.records = []
        self.meta = {
            "label": label,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": {},
        }
        try:
            prop = torch.npu.get_device_properties(0)
            self.meta["device"] = {
                "name": str(getattr(prop, "name", "unknown")),
                "aicore_count": getattr(prop, "aicore_count", None),
            }
        except Exception:  # noqa: BLE001
            pass

    def add(self, op, case, stats, extra=None):
        rec = {
            "op": op,
            "case": case,
            "label": self.label,
            **stats,
        }
        if extra:
            rec.update(extra)
        self.records.append(rec)
        print(
            f"[{self.label}] {op:<12} {case:<44} "
            f"mean={stats['mean_us']:>10.2f}us  min={stats['min_us']:>10.2f}us  "
            f"median={stats.get('median_us', 0):>10.2f}us"
        )

    def save(self):
        if not self.out_json:
            return
        data = {"meta": self.meta, "records": self.records}
        os.makedirs(os.path.dirname(os.path.abspath(self.out_json)), exist_ok=True)
        with open(self.out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n[result] saved -> {self.out_json}")


# ---------------------------------------------------------------------------
# 输入生成 (LI 公共)
# ---------------------------------------------------------------------------


def make_li_inputs(cfg: BenchConfig, batch, topk, layout="BSND", layout_key="BSND",
                   dtype=torch.bfloat16, weight_fp32=False, with_actual=False):
    """
    生成 LightningIndexer 输入.
    返回 dict(query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key,
              block_table, shapes...)
    seq_len = cfg.seq_len (q 与 k 相同), heads = cfg.li_heads, head_dim=128
    """
    S = cfg.seq_len
    N = cfg.li_heads
    D = cfg.li_head_dim
    dev = "npu"

    q = torch.randn(batch, S, N, D, dtype=torch.float32).uniform_(-1, 1).to(dtype).to(dev)
    weights = torch.randn(batch, S, N, dtype=torch.float32).uniform_(-130, 130)
    weights = (weights.to(torch.float32 if weight_fp32 else dtype)).to(dev)

    block_table = None
    key = None
    if layout_key == "PA_BSND":
        block_size = 128
        block_num = batch * ((S + block_size - 1) // block_size)
        key = torch.randn(block_num, block_size, 1, D, dtype=torch.float32).uniform_(-1, 1).to(dtype).to(dev)
        # 每个 batch 恰好 S/block_size 个物理块, 乱序映射
        perm = torch.randperm(block_num, dtype=torch.int32)
        bt = perm.view(batch, -1).to(dev)
        block_table = bt
    else:
        key = torch.randn(batch, S, 1, D, dtype=torch.float32).uniform_(-1, 1).to(dtype).to(dev)

    if layout == "TND":
        q = q.view(batch * S, N, D)
        weights = weights.view(batch * S, N)

    actual_q = None
    actual_k = None
    if with_actual:
        # TND 布局要求前缀和; BSND 为每 batch 长度
        if layout == "TND":
            actual_q = torch.arange(1, batch + 1, dtype=torch.int32, device=dev) * S
        else:
            actual_q = torch.full((batch,), S, dtype=torch.int32, device=dev)
        if layout_key == "TND":
            actual_k = torch.arange(1, batch + 1, dtype=torch.int32, device=dev) * S
        else:  # BSND / PA_BSND: 每 batch 实际长度
            actual_k = torch.full((batch,), S, dtype=torch.int32, device=dev)

    return {
        "query": q,
        "key": key,
        "weights": weights,
        "actual_seq_lengths_query": actual_q,
        "actual_seq_lengths_key": actual_k,
        "block_table": block_table,
        "batch": batch,
        "seq": S,
        "heads": N,
        "head_dim": D,
        "topk": topk,
    }


def make_sfa_inputs(cfg: BenchConfig, batch, sparse_block_size, sparse_size,
                    layout_q="BSND", layout_kv="BSND", dtype=torch.bfloat16,
                    with_actual=False):
    """
    生成 SparseFlashAttention (MLA-absorb) 输入.
    S1=S2=cfg.seq_len, N1=cfg.sfa_heads, N2=1, D=512, rope=64
    sparse_indices 满足约束: 每行有效值在前半部分, 且含因果边界块(最后一个有效值
    必须是当前token所在的block, 参考官方golden生成方式).
    """
    import math

    S = cfg.seq_len
    N1 = cfg.sfa_heads
    N2 = 1
    D = cfg.sfa_head_dim
    DR = cfg.sfa_rope_dim
    dev = "npu"
    K = sparse_size
    bs = sparse_block_size

    query = torch.randn(batch, S, N1, D, dtype=torch.float32).uniform_(-1, 1).to(dtype).to(dev)
    query_rope = torch.randn(batch, S, N1, DR, dtype=torch.float32).uniform_(-1, 1).to(dtype).to(dev)

    # sparse_indices: [B, S1, N2, K], 前半有效, 尾部-1, 且当前块必须包含
    # sparse_mode=3: token s 的有效 KV 长度 = S - S + s + 1 = s+1
    si = torch.full((batch, S, N2, K), -1, dtype=torch.int32)
    for b in range(batch):
        for s in range(S):
            threshold = s + 1 if cfg.sfa_sparse_mode == 3 else S
            if threshold <= 0:
                continue
            valid_blocks_max = math.ceil(max(0, threshold) / bs)
            valid_topk = min(valid_blocks_max, K)
            if valid_topk <= 0:
                continue
            # 随机选前 valid_topk-1 个块 + 当前块(最后一个有效必须是当前块)
            pool = torch.randperm(max(valid_blocks_max - 1, 0), dtype=torch.int32)
            si[b, s, 0, : valid_topk - 1] = pool[: valid_topk - 1]
            si[b, s, 0, valid_topk - 1] = valid_blocks_max - 1
    si = si.to(dev)

    block_table = None
    if layout_kv == "PA_BSND":
        block_size = 128
        block_num = batch * ((S + block_size - 1) // block_size)
        key = torch.randn(block_num, block_size, N2, D, dtype=torch.float32).uniform_(-1, 1).to(dtype).to(dev)
        value = torch.randn(block_num, block_size, N2, D, dtype=torch.float32).uniform_(-1, 1).to(dtype).to(dev)
        key_rope = torch.randn(block_num, block_size, N2, DR, dtype=torch.float32).uniform_(-1, 1).to(dtype).to(dev)
        perm = torch.randperm(block_num, dtype=torch.int32)
        block_table = perm.view(batch, -1).to(dev)
    else:
        key = torch.randn(batch, S, N2, D, dtype=torch.float32).uniform_(-1, 1).to(dtype).to(dev)
        value = torch.randn(batch, S, N2, D, dtype=torch.float32).uniform_(-1, 1).to(dtype).to(dev)
        key_rope = torch.randn(batch, S, N2, DR, dtype=torch.float32).uniform_(-1, 1).to(dtype).to(dev)

    actual_q = None
    actual_kv = None
    if with_actual or layout_q == "TND" or layout_kv in ("TND", "PA_BSND"):
        # TND 的 actual_seq 必须是前缀和; BSND/PA 为每 batch 实际长度
        if layout_q == "TND":
            actual_q = torch.arange(1, batch + 1, dtype=torch.int32, device=dev) * S
        else:
            actual_q = torch.full((batch,), S, dtype=torch.int32, device=dev)
        if layout_kv == "TND":
            actual_kv = torch.arange(1, batch + 1, dtype=torch.int32, device=dev) * S
        else:
            actual_kv = torch.full((batch,), S, dtype=torch.int32, device=dev)

    # TND 布局转换
    if layout_q == "TND":
        query = query.view(batch * S, N1, D)
        query_rope = query_rope.view(batch * S, N1, DR)
        si = si.view(batch * S, N2, K)
    if layout_kv == "TND" and layout_kv != "PA_BSND":
        key = key.view(batch * S, N2, D)
        value = value.view(batch * S, N2, D)
        key_rope = key_rope.view(batch * S, N2, DR)

    return {
        "query": query,
        "key": key,
        "value": value,
        "query_rope": query_rope,
        "key_rope": key_rope,
        "sparse_indices": si,
        "block_table": block_table,
        "actual_seq_lengths_query": actual_q,
        "actual_seq_lengths_kv": actual_kv,
        "batch": batch,
        "seq": S,
        "heads": N1,
        "kv_heads": N2,
        "head_dim": D,
        "rope_dim": DR,
        "sparse_block_size": bs,
        "sparse_size": K,
    }
