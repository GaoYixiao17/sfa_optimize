# LightningIndexer (LI v1) 瓶颈分析 — A5 (Ascend 950, arch35) bs=32/64, S1=S2=128

> 分析对象: `ops-transformer-9.2.0-attention-lightning_indexer/attention/lightning_indexer`
> 目标场景: B=32/64, S1=S2=128, N1(g)=64, N2=1, D=128, bf16, sparse_mode=3, layout BSND/BSND（兼顾 TND/PA）
> topk 两种代表取值: 64（真实 topk 路径）与 2048（S2<topk 退化路径）

## 1. 数据流与流水线结构

kernel 为 AIC+AIV 混合（`KERNEL_TYPE_MIX_AIC_1_2`，1 cube block + 2 vector block 共享同一物理核 UB）。

```
每个 base block = (batch b, gS1 块, s2 块)，mBase=256(=s1Base 4 × g 64)，s2Base=128:
  [AIC]  QueryNd2Nz  GM→L1 64KB（仅首个 s2 块）
         KeyNd2Nz    GM→L1 32KB（PA 布局时逐 128-token 块查 blockTable）
         LoadData    L1→L0A/L0B
         Mmad        256×128×128 bf16（splitMFlag=g==64&&topk<=2048 时拆 2×128）
         Fixpipe     L0C→UB 128KB fp32，dualDst 按 M 拆两半给 2 个 AIV
         CrossCoreSetFlag(CV) ×2
  [AIV×2] 读共享 UB 的 mm1 结果（各 128 行×128 列）
         DataCopyPad weights GM→UB
         MulWeightAndReduceSum: 对 g=64 行做 ReLU·W 加权规约 → 128 个 uint32 score（VF 指令,
           每行 2×64 lane, 64 次 MulAddDst）
         DataCopyPad score → scoreGm workspace
         若本 gS1 块为最后一个 s2 块: ProcessTopK 对每个 token:
           - v = validS2Len = (mask3) actS2SizeOrig-actS1Size+s1Idx+1
           - v < topkCount: 退化路径 CreateVecIndex(0..v-1) + Duplicate(-1) 填充 → 写 GM topkCount×4B
           - v >= topkCount: 单轮/多轮直方图 radix-select topk（4 个 8bit pass, 每元素对齐 256 lane）
         returnValue=1: score→bf16 转换 + -inf 填充 → 写 GM topkCount×2B
         CrossCoreSetFlag(VC) ×2 → AIC 才能复用该 ping-pong 槽
```

分核：`usedCoreNum = CalcTschBlockDim(aivNum, aicNum, aivNum)`（整芯片），`SplitCore` 每核本地执行，
把 totalBlockNum 均分（前面核多 1 块）。B=32: totalBlock = 32×⌈128/4⌉×1 = **1024 块**；B=64: **2048 块**。

## 2. 量化成本模型（B=32, topk=2048 / topk=64）

| 阶段 | 每块工作量 | 全核总量（假设 ~20-40 AIC） | 说明 |
|---|---|---|---|
| cube mm1 | 256×128×128×2=8.4 MFLOP；GM→L1 96KB（Q 64KB+K 32KB）；L1→L0 96KB；L0C→UB 128KB | 8.6 GFLOP；GM 读 98MB（K 重复读 32MB）；UB 写 128MB（核内） | K 在同 batch 的 32 个 gS1 块间重复搬运 |
| vec1 规约 | 每 AIV 2 token：128 列 × 64 g × fp32 MulAdd ≈ 16K MAC | 与 cube 流水重叠 | VF `MulAddDst`，无 bank 冲突问题 |
| topk (topk=2048) | **全部 token 走退化路径**：CreateVecIndex(v≤128) + masked Duplicate + bulk Duplicate(~2048) + DataCopyPad 8KB + (rv: 16 轮转换+2 Duplicate+4KB copy) | 4096 token × ~30 VF 指令 + 32MB(indices)+16MB(values) GM 写 | **输出写带宽 + 每 token 固定向量开销是瓶颈** |
| topk (topk=64) | token 0..62 退化；63..127 走单轮直方图 topk（~60-80 VF 指令/token + 512B score 读 + 256B 写） | 2048 token × ~0.5µs/AIV + 2048 token 退化路径 | 每 token 一次 radix-select 是向量侧主要成本 |
| SplitCore 扫描 | 每核: GetTotalBaseBlockNum + SplitCore 两遍 O(B×⌈S1/4⌉) 标量循环；actual_seq 非空时每 batch 4 次 GM 标量读(GetValue) | B=64: ~4096 ALU 迭代 + 256 次 GM 标量读/核 | actual_seq=None 时无 GM 读（defaultSeqLen） |
| workspace | scoreGm = aicNum × s1Base4 × Align(S2,128) × 4B | ~24×4×128×4 ≈ 50KB | 小，无压力 |

计时口径估算（量级）：topk=2048 全退化 → 向量侧每 token 固定开销 + 48MB GM 写 ≈ **60-120µs**；
topk=64 → cube 流水 ~40µs 与向量 topk ~25-50µs 取 max ≈ **50-70µs**。实际以 A5 实测为准。

## 3. 瓶颈结论（按影响排序）

1. **[topk=2048] 退化路径每 token 固定开销**：`ProcessTopK` L509-520/L522-534 对每 token 执行
   CreateVecIndex + 2 次 Duplicate（其中 bulk Duplicate 填 ~topkCount 个 -1）+ 4 次 Set/WaitFlag；
   returnValue 时 `UIntToFloatReturnValue` 固定跑 topkCountAlign256/128=16 轮转换（L546-552）。
   输出本身只需 [0..v) + -1 填充，与 token 无关的部分被重复计算 4096 次。
   → 模板化：一次构建 [0..K)+(-1) 模板，逐 token 只 patch 边界（因果 mask 下相邻 token 边界 +1）。
2. **[topk=2048] 输出写带宽**：B=32 输出 32MB indices（+16MB values）绝大部分是 -1 填充，
   这是算子契约的硬开销；优化空间只有"别让向量开销叠在写带宽之上"。
3. **[topk=64] 每 token radix-select**：直方图 4 pass + 边界扫描 + GT/EQ 选择 + gather 重排 ≈ 60-80 VF 指令；
   与 cube 流水部分重叠后仍是向量侧主项。相邻 token 的 score 行连续，可合并搬运，但 select 本身难省。
4. **cube 侧 K 重复搬运**：同 batch 内 32 个 gS1 块重复 KeyNd2Nz(GM→L1 32KB) + LoadKeyToL0b(L1→L0)。
   B=32 全核 K 重复读 GM 32MB（L2 可缓存但 GM→L1/L0 指令与带宽仍在）。
   → 记录 lastBIdx/lastS2Idx，相同则跳过（S2=128 时 31/32 的块可跳过）。
5. **SplitCore 的 GM 标量读**（actual_seq 非空时）：每 batch 4 次 `GetValue`（Q/K × 2 遍扫描），
   B=64 每核 256 次串行 GM 标量读（~几百 ns 延迟 each）≈ 几十 µs 级风险。
   → 一次性 DataCopy 到 UB 后从 UB 读。
6. **ping-pong 深度=2**：cube 最多领先向量 1 个块。块均 ~1-2µs 时流水可填满；块数少的场景（B=32 若核多）
   会出现流水填充损耗，属结构性限制，本次不动。

## 4. 优化机会清单

| # | 优化点 | 位置 | 思路 | 预期收益 | 风险 |
|---|---|---|---|---|---|
| P0 | 退化路径模板化+边界 patch | service_vector_arch35.h ProcessTopK else 分支 (L509-534) | indicesOutLocal_ 常驻模板 [0..K)+-1，逐 token 只写边界变化元素；valueOutLocal_ 的 -inf 后缀同理，转换只做 [0..v) | topk=2048 消除每 token ~30 VF 指令（kernel 估 1.3-2x）；topk=64 退化 token 同享 | 低：语义逐 token 等价；需处理 batch 边界 v 回退时的后缀恢复 |
| P1 | AIV 内两 token 的 score 读合并 | ProcessTopK 单轮分支 L404-422 | 同 AIV 相邻 token 的 scoreGm 行连续，blockCount=2 一次 DataCopyPad | 小（每 token 省 1 次 DMA issue） | 低 |
| P2 | actual_seq 批量预取 UB | kernel_base_arch35.h GetActualSeqLen + InitActualSeqLen | Init 时 DataCopy B 个 int32 到 UB（AIC/AIV 各自），GetActualSeqLen 改读 UB | actual_seq 传入时消除 256 次 GM 标量读 | 低；注意 TND 前缀和模式 |
| P3 | cube 跳过重复 K 搬运 | service_cube_arch35.h ComputeMm1/KeyNd2Nz | 增 lastKeyBatchIdx_/lastKeyS2Idx_，同 (b, s2Block) 时跳过 KeyNd2Nz 与 LoadKeyToL0b | S2=128 时 cube 侧 -10~20% | 中：必须确保 L1/L0B 生命周期（L1 是 TPipe 管理、不会被冲掉——KEY_BUF_NUM=3 环形缓冲会在 3 块后复用，需把"跳过"条件限制在环形缓冲未越界时） |
| P4 | 单轮 topk 的 256 全量 Duplicate 精简 | ProcessTopK L410-411 | v 对齐 256 时只补 [v, 256) 尾部零 | 极小 | 低 |

不做的（记录理由）：
- 重写 256 元素专用 topk（sort/双调）：直方图 radix-select 已接近该规模最优，收益不确定且正确性风险大。
- 加深 ping-pong（>2）：涉及 UB 容量（resMm1 已 128KB×2），不可行。
- mBase 加大（s1Base 4→8）：mm1Res UB 需 256KB 超限，不可行。

## 5. 正确性约束

- sparse_mode=3 下每 token 的 v=validS2Len 不同；v<topkCount 时输出 [0..v) 的顺序索引 + 其余 -1；
  语义是"全部有效 token 均选中"（topk≥v 时无选择发生），输出与 score 值无关（indices 恒为 arange）。
- v==0 时全 -1；return_value=1 时 values 的无效位置为 -inf（bf16 0xFF80 / fp16 0xFC00）。
- v>=topkCount 时走真实 topk：top-k 按 score 降序（直方图对 float 转 sortable uint32 key 后 radix 选择），
  并列值时索引顺序由算法决定 —— baseline 与优化实现必须逐 bit 一致（ab_correctness 用排序后集合比较）。
- TND 布局 actual_seq 为前缀和；PA 布局 key 0 轴非连续通过 blockTable 间接寻址。
- 输出 layout 与 query 相同；无效 batch（actSeqLen=0）需清 -1（CleanInvalidOutput/ProcessInvalid）。

## 6. 待实测确认的假设

- A5 (950PR) 的 aivNum/aicNum（影响每核块数与带宽分摊）；UB 实际 256KB。
- 退化路径是否真是热点（topk=2048 场景）→ 用 msProf op_summary + kernel trace 验证。
- scoreGm 写回（vec1）与 topk 读 scoreGm 之间的 L2 命中率。
