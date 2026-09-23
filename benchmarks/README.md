# 对比测试使用说明 (benchmarks/)

> 场景: A5 (Ascend 950), **B=32/64, S1=S2=128**, LI: N1=64/D=128/bf16/topk∈{64,2048};
> SFA: D=512+Dr=64, K∈{64,128}(sbs=1) 与 K=2(sbs=64)。
> 环境依赖: `torch`, `torch_npu`, (v2 需要) `cann_ops_transformer`。

## 文件一览

| 文件 | 用途 |
|---|---|
| `bench_common.py` | 公共配置 (环境变量驱动)、输入构造、NPU Event 计时 |
| `bench_li.py` | LightningIndexer v1 + v2 单算子基准 |
| `bench_sfa.py` | SparseFlashAttention 单算子基准 |
| `bench_e2e.py` | LI→SFA 组合链路基准 |
| `ab_correctness.py` | baseline vs optimized 输出一致性验证 |
| `run_all.sh` | 一键跑全部基准 (按 LABEL 保存结果) |
| `compare_report.py` | 汇总两轮 JSON → Markdown 对比报告 |

## 环境变量 (bench_common.py)

| 变量 | 默认 | 说明 |
|---|---|---|
| `BENCH_LABEL` | `default` | 结果标签 (baseline / optimized), 写入结果 JSON 与文件名 |
| `BENCH_BS` | `32,64` | batch size 列表 |
| `BENCH_SEQ` | `128` | 序列长度 (S1=S2) |
| `BENCH_ITERS` | `100` | 计时迭代次数 |
| `BENCH_WARMUP` | `20` | 预热次数 |
| `BENCH_OUT_JSON` | `result_<op>_<label>.json` | 结果输出路径 |
| `BENCH_OPS` | `v1,v2` | bench_li.py 用: 只跑 v1/v2 |
| `BENCH_SFA_CASES` | `64:1,128:1,2:64` | SFA 用例 `K:sbs` 列表 (K=sparse_size, sbs=sparse_block_size) |
| `BENCH_LI_TOPK` | `64,2048` | LI topk 列表 |
| `BENCH_LI_LAYOUT` / `BENCH_LI_LAYOUT_KEY` | `BSND` | LI 布局 (BSND/TND/PA_BSND) |
| `BENCH_PROFILE` | `0` | 输出 torch_npu profiler kernel 分解 |

计时方式: NPU Event (host 侧 tiling/launch 开销包含在内), 取均值/中位/P99。

## 完整对比流程

```bash
cd op_optimize/benchmarks

# ========= 第一轮: baseline =========
# (新终端) 激活 baseline 实现
source ../scripts/install_pkg.sh builtin      # 或 baseline (重编译原版)
LABEL=baseline ./run_all.sh

# ========= 第二轮: optimized =========
# (新终端) 激活优化实现
source ../scripts/install_pkg.sh optimized
LABEL=optimized ./run_all.sh

# ========= 对比报告 =========
python3 compare_report.py -o report.md
cat report.md
```

`run_all.sh` 会依次运行 li / sfa / e2e 三个基准, 生成
`result_{li,sfa,e2e}_{label}.json`。 单独运行某个基准:

```bash
BENCH_OPS=v2 BENCH_LI_TOPK=2048 python3 bench_li.py
BENCH_BS=64 python3 bench_sfa.py
```
## 正确性 A/B 验证

切换实现是进程级的 (软链 + LD_LIBRARY_PATH), 所以 A/B 分两轮跑:

```bash
# 终端 1 (baseline 激活)
python3 ab_correctness.py save --out ab_baseline.pt

# 终端 2 (optimized 激活)
python3 ab_correctness.py save --out ab_optimized.pt

# 任意终端
python3 ab_correctness.py compare --a ab_baseline.pt --b ab_optimized.pt
```

对比标准:
- LI indices: 逐 token 的 topk **集合**一致 (排序后比较, 容忍并列分数的次序差);
- LI values: `allclose(atol=1e-2)` (bf16);
- SFA 输出: `allclose(atol=2e-2, rtol=2e-2)` (softmax 数值精度)。

覆盖用例: LI v1 (topk=64), LI v2 (op-only 与 op+metadata 两种计时口径在
bench_li.py 中分别给出), SFA (sbs=1/K=64 与 sbs=64/K=2)。

## 计时口径说明 (LI v2)

`bench_li.py` 对 v2 输出两个数字:
- **[op only]**: metadata 张量预生成后仅计时 `lightning_indexer` 调用 —
  这是算子本身的优化对比口径;
- **[op+metadata]**: 每次迭代重新调用 `lightning_indexer_metadata` + 算子 —
  这是真实 serving 时延口径。 **建议在调用方缓存 metadata**
  (BSND 无变长时它是 (shape, seqlens) 的纯函数), 缓存后两者相等。

## 结果 JSON 格式

```json
{
  "meta": {"label": "optimized", "device": {...}, "timestamp": "..."},
  "records": [
    {"op": "LightningIndexer", "case": "B=32 ... topk=2048 ...",
     "mean_us": 123.4, "median_us": 120.0, "min_us": 118.0,
     "p90_us": 140.0, "max_us": 150.0}
  ]
}
```

`compare_report.py` 按 (op, case) 对齐两轮结果, 输出每个用例的
baseline/optimized 均值与加速比表。

## 性能分析 (建议)

优化是否命中瓶颈假设, 建议在 A5 上先用 msProf 采集 base 版 profile
(算子仓 `tests/pytest/` 下自带 `collect_perf_data.py`):
`msprof --application="python3 bench_li.py" --output=./prof`，
对照 `../analysis/*_bottleneck.md` 的瓶颈清单校准权重后再决定进一步优化方向。
