# sfa_optimize — A5 (Ascend 950) 算子优化工作仓

LightningIndexer / LightningIndexerV2 / SparseFlashAttention 三个 CANN 算子在
**A5 服务器 (Ascend 950PR, DAV_3510, CANN 9.2.0-beta.2)** 上 `bs=32/64, seq_len=128`
场景（DSA / DeepSeek Sparse Attention 调用栈）的：

- 瓶颈分析（`analysis/`）
- 优化实现（三个算子源码树，相对根目录原始 zip 的差异即全部改动，共 8 项）
- 对比测试（`benchmarks/`：性能基准 + 正确性 A/B + 报告生成）
- 编译脚本（`scripts/`：baseline / optimized 双实现编译安装切换）

## 目录结构

```
ops-transformer-9.2.0-attention-lightning_indexer/       # LI v1 源码 (含优化)
ops-transformer-9.2.0-attention-lightning_indexer_v2/    # LI v2 源码 (含优化)
ops-transformer-9.2.0-attention-sparse_flash_attention/  # SFA 源码 (含优化)
ops-transformer-9.2.0-attention-*.zip                    # 原始基线源码 (未修改, baseline 用)
ops-transformer/                                        # 官方构建框架树 (gitcode 9.2.0 镜像, 编译用)
analysis/            # 三算子瓶颈分析 + SUMMARY
benchmarks/          # bench_li / bench_sfa / bench_e2e / ab_correctness / run_all.sh / compare_report
scripts/             # prepare_sources.sh / build_ops.sh / install_pkg.sh
OPTIMIZATION_NOTES.md  # 全部优化的实现说明、未实施项理由、验证清单
BUILD.md               # A5 上编译/安装/切换 baseline vs optimized 完整流程
```

## 快速开始（A5 服务器）

```bash
# 1. 编译两套算子包 (构建框架树已内置仓内 ops-transformer/, 见 BUILD.md)
bash scripts/prepare_sources.sh
bash scripts/build_ops.sh              # 框架默认取仓内 ops-transformer/
bash scripts/install_pkg.sh            # 安装

# 2. 对比测试
cd benchmarks
source ../scripts/install_pkg.sh builtin       # 或 baseline
LABEL=baseline ./run_all.sh
source ../scripts/install_pkg.sh optimized     # 新终端
LABEL=optimized ./run_all.sh
python3 compare_report.py -o report.md

# 3. 正确性 A/B (实现等价性验证)
python3 ab_correctness.py save --out ab_baseline.pt
python3 ab_correctness.py save --out ab_optimized.pt
python3 ab_correctness.py compare --a ab_baseline.pt --b ab_optimized.pt
```

## 优化概要（详见 OPTIMIZATION_NOTES.md）

| 算子 | 已实施 |
|---|---|
| LI v1 | 退化路径模板化（topk=2048 全 token 受益）、K 跨块 L1/L0 缓存（GM 流量 64×→2×）、seq_len 标量读缓存 |
| LI v2 | 模板化（移植）、K 缓存（移植）、ProcessDecode SyncAll 跳过（S2≤128）、tiling workspace 精确公式（24.6MB→0.2MB） |
| SFA | 稀疏索引批量预取 + blockTable 标量缓存（串行 GM 读 ~53万 → ~3.3万 次批拷贝，预估 1.5–3×） |

> 注意：LI v1 的 kernel 通过相对路径 `../../../../lightning_indexer_v2/...` 引用 v2 源码，
> 两个算子目录必须保持兄弟目录关系（`scripts/build_ops.sh` 已保证）。
> 建议先在真机用 msProf 采集 builtin profile 校准瓶颈（见 `analysis/SUMMARY.md` 第 4 节）。
