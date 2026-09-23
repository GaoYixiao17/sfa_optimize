# 优化工作总结 (SUMMARY)

> 场景: **A5 (Ascend 950PR, DAV_3510), bs=32/64, S1=S2=128**, DSA (DeepSeek Sparse
> Attention) 调用栈三算子。 详细瓶颈分析见同目录三份报告; 优化实现细节见
> 根目录 `OPTIMIZATION_NOTES.md`; 编译/测试见 `BUILD.md` 与 `benchmarks/README.md`。

## 1. 瓶颈分析结论 (一句话版)

| 算子 | 主要瓶颈 (按预期收益排序) | 依据 |
|---|---|---|
| LI v1 | ① topk=2048 时**全部 token** 走退化路径, 每 token 全宽 -1/-inf 填充 + 16 轮值转换; ② g=64 时同一 K 被 64 个 gS1 块重复搬运 (GM→L1→L0 各 64 次, GM 流量 64× 张量本身); ③ seq_len GM 标量读 | `li_v1_bottleneck.md` |
| LI v2 | ① 同 v1 退化路径; ② 同 v1 K 重复搬运 (块数还是 v1 的 2 倍); ③ S2≤128 时仍执行全量 ProcessDecode SyncAll (2048/4096 块); ④ tiling workspace 按最坏情况分配 (topk=64 时 24.6MB, 且 score 区公式与 kernel 寻址不符) | `li_v2_bottleneck.md` |
| SFA | ① gather 循环**串行 GM 标量读** (~53 万次 @B=32/K=128, 估计 0.55–1.1ms, 最大单项嫌疑); ② KV 每 token 经 GM workspace 中转 4 次搬运; ③ 任务粒度 1 token 无批量化 | `sfa_bottleneck.md` |

共性: 三份报告的 roofline 估算都显示实际耗时远超带宽/算力下限
(SFA 估计 0.6–1.2ms vs 96µs roofline), 开销大头在**控制流/重复搬运/标量串行**,
而非算力 — 这决定了优化选型: 消除冗余, 而非逼近峰值。

## 2. 已实施的优化 (8 项)

| # | 算子 | 优化 | 文件 | 预期效果 |
|---|---|---|---|---|
| 1 | LI v1 | 退化路径模板化 + 转换宽度缩窄 (P0) | `lightning_indexer_service_vector_arch35.h` | topk=2048 每 token 从 2×全宽填充+16 轮转换 → 2 次短向量操作+1 轮 |
| 2 | LI v1 | seq_len GM 读 per-launch 缓存 (P2) | `lightning_indexer_kernel_arch35.h` | 传 actual_seq_lens 时消除重复标量读 |
| 3 | LI v1 | K 跨块缓存 L1+L0 (P3) | `lightning_indexer_service_cube_arch35.h` | K 的 GM 流量 64×→2× (~67MB→2MB @B=32); L1→L0 搬运同比例 |
| 4 | LI v2 | tiling workspace 精确公式 (正确性+内存) | `lightning_indexer_v2_tiling.cpp` | topk=64: 24.6MB→0.2MB; 修正 score 区欠分配 |
| 5 | LI v2 | ProcessDecode SyncAll 跳过 (S2≤128) | `lightning_indexer_v2_kernel_arch35.h` | 消除 2048/4096 块的全量同步屏障 |
| 6 | LI v2 | 退化路径模板化 (P0 移植, 含 offset/LD 处理) | `lightning_indexer_v2_service_vector_arch35.h` | 同 #1 |
| 7 | LI v2 | K 跨块缓存 L1+L0 (P3 移植) | `lightning_indexer_v2_service_cube_arch35.h` | 同 #3 |
| 8 | SFA | 稀疏索引批量预取 + blockTable 标量缓存 (O1) | `sparse_flash_attention_service_vector_mla_arch35.h` | 串行 GM 读 ~53 万 → ~3.3 万次批拷贝 (+缓存命中); 若 B2 坐实预计 1.5–3× |

所有改动遵循统一约束: **只做语义保持的低风险变换** — 事件协议的槽级 Set/Wait 全部
保留, 仅跳过搬运本体; 模板/缓存失效点显式管理; 越界与对齐按最坏情况防护
(SFA 预取窗口用行尾钳制 + 标量回退)。

## 3. 明确不做的项 (风险/收益权衡)

见 `OPTIMIZATION_NOTES.md` 第 4 节。 最重要的三个:
- **SFA-O2 整批 KV 常驻 L1 + vselr 列掩码** (估计 3–6×): 依赖不在本仓库的 VF 掩码
  协议 (`vf_mul_sel_softmaxflashv2_cast_nz_sfa.h`), 必须先真机差分实验;
- **V2-⑤ s1Base=4+splitM**: v2 cube 无 splitM 路径, 属结构性移植;
- **V2-① metadata 缓存**: 属 serving 层 (benchmark 已在计时区外预生成, 并输出
  op-only / op+metadata 两种口径供参考)。

## 4. 建议的验证顺序 (A5 真机)

1. `msProf` 采集 builtin 的 profile → 校准瓶颈权重 (尤其确认 SFA B2 标量读链
   与 LI 退化路径占比);
2. `BUILD.md` 流程编译安装 baseline + optimized;
3. `ab_correctness.py` 两轮保存 + compare (覆盖 topk 64/2048、B 32/64、sbs=1/64);
4. `run_all.sh` 两轮 + `compare_report.py` 出加速比表;
5. 把实测结果回填本目录, 按真实瓶颈决定是否投入第 3 节的高风险项。

## 5. 交付物清单

```
analysis/li_v1_bottleneck.md      LI v1 瓶颈分析 (数据流/成本模型/优化分级)
analysis/li_v2_bottleneck.md      LI v2 瓶颈分析 (11 项优化排序)
analysis/sfa_bottleneck.md        SFA 瓶颈分析 (9 项优化排序)
OPTIMIZATION_NOTES.md             全部优化实现说明 + 未实施项理由 + 验证清单
BUILD.md                          A5 上编译/安装/切换 baseline vs optimized
benchmarks/README.md              对比测试使用说明
benchmarks/*.py, run_all.sh       基准/A-B 正确性/报告生成
scripts/{prepare_sources,build_ops,install_pkg}.sh   编译三件套
ops-transformer-9.2.0-attention-{lightning_indexer,lightning_indexer_v2,sparse_flash_attention}/
                                  优化后的算子源码 (相对原始 zip 的差异即全部改动)
```
