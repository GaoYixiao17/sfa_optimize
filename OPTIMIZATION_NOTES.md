# 算子优化说明 (OPTIMIZATION_NOTES)

> 目标场景: **A5 服务器 (Ascend 950PR, DAV_3510, CANN 9.2.0-beta.2)**,
> `bs ∈ {32, 64}`, `S1 = S2 = 128`, LI: `N1=g=64, D=128, bf16, topk ∈ {64, 2048}, sparse_mode=3`;
> SFA: `D=512+Dr=64, N2=1, K ∈ {64,128}(sbs=1) / {2}(sbs=64)`, BSND。
>
> 三份瓶颈分析见 `analysis/{li_v1,li_v2,sfa}_bottleneck.md`。
> 所有改动均为**低风险、语义保持**型优化; 每项都给出验证方式 (见文末)。

## 目录

- [1. LightningIndexer v1](#1-lightningindexer-v1)
- [2. LightningIndexer v2](#2-lightningindexer-v2)
- [3. SparseFlashAttention](#3-sparseflashattention)
- [4. 已评估但未实施的优化](#4-已评估但未实施的优化)
- [5. 验证方案](#5-验证方案)

---

## 1. LightningIndexer v1

源码: `ops-transformer-9.2.0-attention-lightning_indexer/.../attention/lightning_indexer/`

### V1-P0 退化路径模板化 【P0, 预期收益最大】

**文件**: `op_kernel/arch35/lightning_indexer_service_vector_arch35.h`

**背景**: causal mask 下每个 token 的有效长度 `v = token_idx + 1`。 `v < topkCount`
时走退化路径: indices = `[outputIdxOffset .. +v) + (-1 填充)`, values = 前缀转换 +
`-inf 填充`。 本场景中 **topk=2048 时全部 128 个 token**、topk=64 时前 ~63 个 token
都走该路径。 原实现对每个 token 都做:
1. 全宽 `Duplicate(-1)` / masked Duplicate 尾部填充 (topk=2048 时每次 8KB×2);
2. 全宽 `UIntToFloatReturnValue` 转换 (内部按 128 元素/轮循环, topk=2048 时 16 轮)。

**优化**: 在 `indicesOutLocal_` / `valueOutLocal_` 中常驻一份"当前 v"的模板
(前缀 = 正确值, 后缀 = -1 / -inf)。 相邻 token 的 v 单调 +1 (mask3 语义), 每个 token
只需 patch 边界增量:
- `v > state`: `CreateVecIndex(state, v)` 补 `[state, v)` 一段 (values 同理由转换后缀补 -inf);
- `v < state`: `Duplicate(-1)` 回退 `[v, state)` 一段;
- 新 batch / offset 变化 / 真实 topk 路径复用 buffer 后: 整体重建一次 (state 置 -1)。

同时删除了原实现中永远不生效的死代码填充块, 并把退化路径的 rv 转换宽度从
`topkCountAlign256_` 缩窄为 `Align(v, 128)` (转换循环轮数 topk=2048 时 16 轮 → 1 轮),
转换产生的 `[v, Align(v,128))` 垃圾区由模板 patch 修正。

**事件协议**: v≤0 分支用单一 `SetFlag/WaitFlag(V_MTE3)` 对覆盖两份拷出;
`CleanInvalidOutput` 把 valueOutLocal_ 刷成全 -inf 后同步 `degenValState_ = 0`
(该操作与 v=0 模板等价, 避免无谓重建)。

**正确性论证**: 模板后缀维护区间 `[v, max(AlignUp(v,128), oldState))` 同时覆盖
转换垃圾区与回退残留; 模板失效点 (真实 topk / buffer 复用) 全部显式置 state=-1。

### V1-P2 seq_len GM 标量读缓存 【P2】

**文件**: `op_kernel/arch35/lightning_indexer_kernel_arch35.h`

原 `GetS1S2ActualSeqLen` 每个块每 token 都从 GM 标量读 actual_seq_lens。
新增 128 项 per-launch 缓存 (`SEQ_LEN_CACHE_MAX_B=128`, 哨兵初始化), 同一 bIdx
只读一次 GM, 后续命中直接读寄存器数组。 kernel 对象每次 launch 都是新建的,
缓存生命周期 = 单次 launch, 无跨调用一致性风险。 bIdx ≥ 128 时回退直读。

### V1-P3 K 数据跨块缓存 (L1 + L0) 【P3】

**文件**: `op_kernel/arch35/lightning_indexer_service_cube_arch35.h`

**背景**: g=64 时每个 (b,n2) 有 64 个 gS1 块, 每块都完整搬运同一份 K:
`KeyNd2Nz` GM→L1 (32KB) + `LoadData` L1→L0B (32KB)。 B=32 时 K 的 GM 流量
≈ 64×张量本身 (67MB vs 1.05MB), L1/L0 搬运量同倍数。

**优化**: 以 `(runInfo.tensorKeyOffset, s2GmBaseOffset + s2GmOffset)` 为 key
记录每个槽的内容:
- L1 K 槽 (3 轮换, chunk 粒度): 命中 → 跳过 KeyNd2Nz/KeyNd2NzForPA;
- L0 K 槽 (kl0BufIdx_ 2 轮换, 块粒度): 命中 → 跳过 LoadKeyToL0b。

**事件协议 (关键约束)**: 槽级事件 `KEY_MTE1_MTE2` / `M_MTE1` 的 Set/Wait 全部保留
(它们保护的是槽复用次序, 与是否真的搬运无关); 只跳过搬运本体和块内自包含的
`MTE2_MTE1` / `MTE1_M` 配对屏障 (无搬运则无危害)。 跳过数据搬运只会**移除**危害,
不会引入危害。

**tag 粒度**: L1 tag 含 `s2GmOffset` (chunk 粒度); L0 tag 也含 `s2GmOffset`
(一个块内多 chunk 时各 chunk K 不同, 避免假命中)。 本场景 S2=128=s2BaseSize 单
chunk, 每个 (b,n2) 组: K 的 GM→L1 从 64 次降为 2 次 (两个 L0 槽各预热一次),
L1→L0 同样 64→2。

**PA 说明**: cache key 用 `runInfo.tensorKeyOffset` 而非 bIdx, PA blockTable 场景
同样精确。

---

## 2. LightningIndexer v2

源码: `ops-transformer-9.2.0-attention-lightning_indexer_v2/.../attention/lightning_indexer_v2/`

### V2-W tiling workspace 尺寸修正 【host, 正确性+内存】

**文件**: `op_host/lightning_indexer_v2_tiling.cpp` (DAV_3510 分支)

原 tiling 的 workspace 公式与 kernel 实际寻址不匹配: score 区按 `aicNum × 4` 计,
但 kernel 用 `GetBlockNum() × s1BaseSize` 寻址 (原实现靠 LD 区富余空间"吸收"了
越界, 属隐性 bug); 且 s1BaseSize 未按 kernel 规则 (`gSize>32 || topk>2048 → 2, 否则 4`)
计算, topk=64 时按最大值分配 (24.6MB)。

修正为与 kernel `InitTilingData` 完全一致的布局:
```
score 区   : blockDim × s1BaseSize × Align(S2,128) × 4B
ldScore 区 : 2 × blockDim × s1BaseSize × Align(topk,16) × 4B   (head/tail 两槽)
ldIndex 区 : 同 ldScore
```
topk=64/B=32 时 24.6MB → ~0.2MB (host 分配与 memset 开销同步消失);
s1Base=2 场景不再越界依赖富余空间。

### V2-S ProcessDecode SyncAll 跳过 【S2≤128 必然生效】

**文件**: `op_kernel/arch35/lightning_indexer_v2_kernel_arch35.h`

`kSeqSize ≤ s2BaseSize` (本场景 S2=128) 时结构上不可能产生 LD 任务
(split-head 需要超过一个 s2 base 块), ProcessDecode 提前返回, 跳过
`InitLDBuffers + ICachePreLoad + SyncAll`。 该条件由 tiling 常量决定,
同 launch 内所有 AIV 判定一致, 不破坏同步语义。 v2 在 g=64 时是
2048/4096 个 block (v1 的 2 倍), 全体 SyncAll 的等待气泡被消除。

### V2-P0 退化路径模板化 【P0, 移植自 v1】

**文件**: `op_kernel/arch35/lightning_indexer_v2_service_vector_arch35.h`

与 V1-P0 同思路, v2 特有差异:
- indices 前缀带 `outputIdxOffset` (per-row GM 读出): 模板记录当前 offset,
  offset 变化时整体重建 (BSND 无 varlen 时通常恒为 0, 不触发重建);
- `valueOutLocal_` 为 float (`NEG_INF_FLOAT` 位型), 模板值区用
  `ReinterpretCast<uint32_t>` 维护;
- rv 转换调用 `liV2Vector1::UIntToFloatReturnValue(out, in, topK, negInf)`
  (内部 `ceil(topK/128)` 轮), 退化路径宽度 `Align(v,128)` + 模板 patch,
  topk=2048 时 16 轮 → 1 轮;
- **LD 路径失效**: ProcessTopK 入口对 `info.isNeedLD` 置 state=-1
  (LD 分支的 `Adds`/尾部填充会改写 indicesOutLocal_, 模板与内容脱钩);
  ProcessDecode 在所有 ProcessTopK 之后执行, 不构成反向风险;
- LD 分支 (isNeedLD) 的退化 token 保持原 `CreateVecIndex` 路径
  (其后续 `Adds(curS2StartIdx)` 会改写 buffer, 不能用模板);
- `CleanInvalidOutput` 全 -inf 刷写后同步 `degenValState_ = 0`。

### V2-P3 K 数据跨块缓存 (L1 + L0) 【P3, 移植自 v1】

**文件**: `op_kernel/arch35/lightning_indexer_v2_service_cube_arch35.h`

同 V1-P3。 v2 差异: 无独立 kl0BufIdx_, K 的 L0 槽 = `l0BufIdx_ % 2`
(每个 L0 迭代轮换), 因此 L0 命中判断放在 s1gL1Offset 循环内部按当前槽判定;
L1 判断独立于 L0 (L1 命中时旧数据仍可被 L0 miss 的 LoadKeyToL0b 安全读取,
槽级 `KEY_MTE1_MTE2` 事件保证旧写入已完成)。 tag 均含 `s2GmOffset`。
v2 每个 (b,n2) 64 个 gS1 块 → K 搬运 64→2 次。

---

## 3. SparseFlashAttention

源码: `ops-transformer-9.2.0-attention-sparse_flash_attention/.../attention/sparse_flash_attention/`

### SFA-O1 稀疏索引批量预取 + blockTable 缓存 【首要嫌疑项, 低风险】

**文件**: `op_kernel/arch35/sparse_flash_attention_service_vector_mla_arch35.h`

**背景** (分析报告 B2, 估计 0.55–1.1ms @ B=32/K=128, 为该算子最大单项嫌疑):
`ProcessSparseKv` 的 gather 循环对每对 token 串行执行 4 次 GM 标量读
(`sparseIndices.GetValue ×2` + PA 时 `blockTable.GetValue ×2`), 全网约 53 万次
串行 load, 每次百余 ns 级延迟且完全不可重叠。

**优化 1 — sparse_indices 批量预取**: 外层每 16 行批次开始时, 一次 `DataCopy`
把本批最多 16 个索引预取到 UB (新增 96B `sparseIdxStageBuf`), 内层循环改从 UB
标量读。 三个关键实现约束:
- **32B 对齐**: DataCopy 要求 src 32B 对齐、长度 32B 的倍数 → 拷贝起点向下对齐
  到 8 元素, 长度向上对齐; 窗口用 **当前 (b,s1) 行尾** 钳制 (绝不越出本行,
  从根本上排除张量越界读);
- **行尾回退**: 行尾余量不足 32B 窗口时 (仅每行最后一批可能发生), 回退为逐元素
  `GetValue` 填 UB — 与原实现同代价, 语义不变;
- **预取窗口大小论证**: 内层循环每对 token 要么双 -1 break, 要么贡献 2 行
  (`s2 += 2`), 8 对 × 2 = 16 个位置, 16 项窗口必然覆盖。

**优化 2 — blockTable 1 项标量缓存**: PA 场景 `GetkeyOffset` 里相邻选中 token
大多落在同一 KV cache block (blockSize 通常 128), 以 `(boIdx<<32)|blkTableIdx`
为 key 缓存上次查询值, 命中免 GM 读。 未命中路径与原实现完全一致。

**收益估算**: B=32/K=128 时 sparseIndices 串行读 53 万 → ~3.3 万次批拷贝;
blockTable 读按 70–90% 命中率估算再省 ~30 万次。 若 B2 瓶颈坐实 (msProf 验证),
预计整体 1.5–3×。

---

## 4. 已评估但未实施的优化

按 (收益 × 风险) 排序说明取舍 — 全部无硬件验证环境下, 只实施低风险改动:

| 编号 | 内容 | 不做的原因 |
|---|---|---|
| SFA-O2 | 整批 KV 常驻 L1 + vselr 列掩码 (估计 3–6×) | 依赖 `vf_mul_sel_softmaxflashv2_cast_nz_sfa.h` 的 VF 掩码协议逆向 (源码不在本仓库), 必须先在硬件上做 -1 padding 差分实验验证; 风险高 |
| SFA-O3 | block 级 sparseBlockSize (sbs>1 场景) | A5 上 tiling 强制 sbs=1 (tiling.cpp:793), 本场景无收益 |
| SFA-O4 | 去掉 KV 的 GM workspace 中转 | 涉及 cube/vector 跨核 Buffer 协议重构 |
| SFA-O5 | Q 批量化 | 任务粒度已是 1 token (qSNumInOneBlock=1), 无批可合 |
| SFA-O7 | needInit 全量 memset 精度 | 变长 batch 才触发; 本场景等长不触发 |
| V2-⑤ | s1Base=4 + splitM (g=64, S2≤128) | v2 cube 无 splitM 路径, 移植属结构性改动 (mBase 128→256、dualDst fixpipe 配对、vector 行映射); 收益 (块数减半) 大但无硬件验证风险不可控 |
| V2-① | metadata torch 层缓存 | 属调用方 (serving 层) 优化而非算子优化; benchmark 已在计时区外预生成。 建议: BSND 无 varlen 时 metadata 是 (shape, seqlens) 的纯函数, 可按 key 缓存复用 |
| V1/V2-P4(⑨) | score UB 直读 (vector 不经 GM 读 score) | 需确认 AIV 对 AIC fixpipe dualDst 写入区的可见性协议; 与 MIX_AIC_1_2 UB 模型交互复杂 |
| V1-P1 | ProcessVec1 权重/规约微优化 | 收益小 (每 2 token 一次 MulWeightAndReduceSum), 已被 P0/P3 覆盖主要矛盾 |

> 三份报告共同建议: 先用 msProf (tests/pytest 下有 `collect_perf_data.py`) 在
> 真机采集 base 版 profile, 校准瓶颈权重后再决定是否投入高风险项 (尤其 SFA-O2)。

---

## 5. 验证方案

1. **编译验证**: A5 服务器上 `scripts/build_ops.sh --impl optimized --framework <ops-transformer 树>`
   (详见 `BUILD.md`)。 注意 LI v1 的 kernel 通过相对路径
   `../../../../lightning_indexer_v2/op_kernel/arch35/...` 引用 v2 的源码,
   两个算子目录必须同时存在于构建树 (build_ops.sh 已保证)。
2. **正确性 A/B**: `benchmarks/ab_correctness.py` — baseline 与 optimized 各跑一轮
   保存输出, indices 排序后集合比较 (容忍 topk 同分次序差), values allclose(2e-2)。
   建议覆盖: topk ∈ {64, 2048} × B ∈ {32, 64} × PA/BSND × 有无 actual_seq_lens。
3. **性能对比**: `benchmarks/run_all.sh` + `compare_report.py` (详见
   `benchmarks/README.md`)。
4. **激活机制核验**: 优化算子是否真的被加载 — `source scripts/install_pkg.sh status`
   + 两轮计时应有可测差异; 若无差异说明回退到了内置算子。
5. **点验清单** (重点回归面):
   - LI v1/v2 topk=2048 (全退化 token, 模板压力路径) 与 topk=64 (混合路径, 模板
     重建边界 v=63→64);
   - LI v2 `return_value=1` (values 输出转换宽度变化);
   - LI PA block_table 非连续页表 (K 缓存 key 正确性);
   - SFA sparse_indices 行尾非 8 对齐 (K=100 之类的奇数 topk, 走标量回退路径);
   - SFA TND 布局 + split-G (sfaSparseCalSize < 16 的批次)。
