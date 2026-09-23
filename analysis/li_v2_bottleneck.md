# LightningIndexerV2 (LI v2) Ascend 950 (A5 / DAV_3510 / arch35) 小形状性能瓶颈分析

分析对象：`D:\code\op_optimize\ops-transformer-9.2.0-attention-lightning_indexer_v2\ops-transformer-9.2.0-attention-lightning_indexer_v2\attention\lightning_indexer_v2\`（下文以 `<ROOT>` 指代该目录）
对比对象（v1）：`D:\code\op_optimize\ops-transformer-9.2.0-attention-lightning_indexer\ops-transformer-9.2.0-attention-lightning_indexer\attention\lightning_indexer\`（下文以 `<V1>` 指代）

目标场景：
- 芯片 Ascend 950PR（A5，`__CCE_AICORE__==310` → `<ROOT>\op_kernel\lightning_indexer_v2.cpp:19-20` 走 arch35）
- B=32/64，S1=128，S2=128（短序列 prefill），N1=64（g=64，N2=1），D=128，bf16
- topk=64 与 2048；mask_mode=3 与 0；cmp_ratio=1（附带 cmp_ratio>1 行为）
- layout_q/layout_k = BSND/BSND 为主，兼顾 TND/TND
- 只读代码分析，未修改任何源码。

**核心结论（TL;DR）**
1. g=64>32 触发 v2 的 `s1BaseSize=2`（`<ROOT>\op_kernel\arch35\lightning_indexer_v2_kernel_arch35.h:206-212`），基本块 m=128，B=32 时全图 2048 个基本块、每 AIC ~85 块，**每块 4 次跨核同步 + 1 次 Fixpipe + 1 次 KeyNd2Nz**。相比 v1（g=64/topk≤2048 用 s1BaseSize=4 + splitM，1024 块），v2 的块数、跨核同步次数、K 的 GM→L1 搬运次数全部翻倍 —— 这是小形状下 v2 相对 v1 的主要退化。
2. topk=2048 且 S2=128 时走**退化路径**（validS2Len=128 < topk）：每 token 写 8KB indices（+8KB values），其中 ≥94% 是 -1/-inf 填充；B=32/rv=1 输出 32MB+32MB，B=64 为 64MB+64MB，纯写带宽即数十 µs，是绝对的带宽瓶颈；且 `UIntToFloatReturnValue` 对 2048 个元素全量转换（仅 128 个有效）属纯浪费。
3. 950 上 metadata 必传（`<ROOT>\op_host\lightning_indexer_v2_tiling.cpp:589-591`），形成 metadata op + 主 op **两次 kernel launch**；对 30~80µs 量级的小 kernel，eager 模式下第二次 launch 的 host/流开销占比可达 15~40%。BSND 静态形状下 metadata 是纯标量函数，完全可缓存复用。
4. 主 op workspace 恒定多申请 ~24.6MB（tiling 沿用 arch22 的 `S1_BASE_SIZE=8 / TOPK_MAX_SIZE=8192` 公式，`<ROOT>\op_host\lightning_indexer_v2_tiling.cpp:1216-1218`），而 kernel 实际按 `s1BaseSize=2 × Align(topk,16)` 只需 ≤1.6MB。
5. `ProcessDecode` 中 `SyncAll()`（48 AIV 全栅栏）与 `ICachePreLoad` **无条件执行**（`lightning_indexer_v2_kernel_arch35.h:804-814`），即使本形状没有任何 LD（长 S2 跨核归约）任务。

---

## 1. 调用协议（metadata op → LI v2 op 完整链路）

### 1.1 torch 层调用方式

`<ROOT>\torch_extension\lightning_indexer.py:29-44` 注册两个 schema；`__init__.py` 导出 `lightning_indexer` 与 `lightning_indexer_metadata`。950 上必须**先跑 metadata 算子**，再把结果传给主算子（golden 用例即此写法，`<ROOT>\tests\pytest\lightning_indexer_v2_golden.py:1423-1470`）：

```python
import torch, torch_npu
import cann_ops_transformer.ops as ops   # lightning_indexer, lightning_indexer_metadata

B, S1, S2, N1, D, topk = 32, 128, 128, 64, 128, 64   # g = N1/N2 = 64
q = torch.randn(B, S1, N1, D, dtype=torch.bfloat16, device="npu")     # BSND
k = torch.randn(B, S2, 1,  D, dtype=torch.bfloat16, device="npu")     # BSND, N2=1
w = torch.randn(B, S1, N1,    dtype=torch.float32,   device="npu")    # fp32 权重

# ① 第 1 次 launch：LightningIndexerV2Metadata（独立 ACLNN 算子 aclnnLightningIndexerV2Metadata）
metadata = ops.lightning_indexer_metadata(
    num_heads_q=N1, num_heads_k=1, head_dim=D, topk=topk,
    # cu_seqlens_q/k、seqused_q/k、cmp_residual_k 可选（NPU tensor）
    batch_size=B, max_seqlen_q=S1, max_seqlen_k=S2,
    layout_q="BSND", layout_k="BSND", mask_mode=3, cmp_ratio=1)
# 返回: int32[1024]，NPU tensor（<ROOT>\torch_extension\csrc\lightning_indexer.cpp:31-69）

# ② 第 2 次 launch：LightningIndexerV2（aclnnLightningIndexerV2）
sparse_indices, sparse_values = ops.lightning_indexer(
    q, k, w, topk,
    metadata=metadata,                       # 950 上必传
    max_seqlen_q=S1,                         # TND 必传；BSND 可 -1
    layout_q="BSND", layout_k="BSND",
    mask_mode=3, cmp_ratio=1, return_value=1)
# sparse_indices: int32 [B, S1, N2, topk]；return_value=1 时 sparse_values: fp32 同形，否则 shape (0,)
```

aclnn 原生两段式接口（GetWorkspaceSize + execute）见 `<ROOT>\examples\test_aclnn_lightning_indexer_v2.cpp:186-246`（metadata）与 `:247-330`（主 op）；C++ 侧 `ACLNN_CMD(aclnnLightningIndexerV2Metadata, ...)` 在 `<ROOT>\torch_extension\csrc\lightning_indexer.cpp:65-67`，主 op 在 `:132-134`。

### 1.2 两个算子的输入输出

**LightningIndexerV2Metadata（标量 + seqlen 张量 → int32[1024]）**
- 输入：`num_heads_q(N1), num_heads_k(N2), head_dim(D), topk`，可选 `cu_seqlens_q/k, seqused_q/k, cmp_residual_k`（int32 设备张量），`batch_size, max_seqlen_q/k, layout_q/k, mask_mode, cmp_ratio`（标量）。
- 输出：`int32[1024]`（`<ROOT>\torch_extension\csrc\lightning_indexer.cpp:29,53`）。其 host 源码不在本仓库（预编译 ACLNN 算子），但输出布局由 `<ROOT>\op_kernel\lightning_indexer_v2_metadata.h` 完整定义（见 1.3）。
- 作用：把“逐 batch/逐 gS1 块枚举 + 代价均衡分核”在主 op 之外算好，主 kernel 每核只读十几个 uint32。STC 用例注释（`<ROOT>\tests\pytest\test_lightning_indexer_v2_stc.py:661-680`）表明其调度算法为 AssignByBatch → AssignByRow → AssignByBlock 的代价表 + 尾块处理 + 空核 ForceAssign，并对长 S2 生成 FD/LD 任务。

**LightningIndexerV2（`<ROOT>\op_host\lightning_indexer_v2_def.cpp:22-91`）**
| 槽位 | 名字 | 类型/形状（BSND 主场景） | 说明 |
|---|---|---|---|
| in0 | q | bf16/fp16 [B,S1,N1,D] | D=128，N1≤64 |
| in1 | k | bf16/fp16 [B,S2,N2,D]（BSND）/ [T2,N2,D]（TND）/ PA_BBND | N2=1 |
| in2 | w | fp32 [B,S1,N1] | 每 (b,s1) 的 g 维权重 |
| in3-6 | cu_seqlens_q/k, seqused_q/k | int32 可选 | 实际长度；BSND 下 q 侧不支持 cu_seqlens_q（docs `aclnnLightningIndexerV2.md:424`） |
| in7 | cmp_residual_k | int32[B] 可选 | mask3+cmp_ratio>1 必传，元素值 < cmp_ratio |
| in8 | block_table | int32 可选 | 仅 layout_k=PA_BBND |
| in9 | output_idx_offset | int32 [B,S1,N2] 可选 | 索引偏移，累加到输出 indices |
| in10 | metadata | int32[1024] | **950 必传**（tiling.cpp:589-591, docs:429） |
| out0 | sparse_indices | int32 [B,S1,N2,topk]（TND: [T1,N2,topk]） | 无效位填 -1 |
| out1 | sparse_values | fp32 同形（rv=1）/ shape (0,)（rv=0） | 无效位填 -inf |
| attr | topk/max_seqlen_q/layout_q/layout_k/mask_mode/cmp_ratio/return_value | | 950 上 topk∈[1,8192] |

语义（docs `aclnnLightningIndexerV2.md:28-38`）：`Top-k { (W_{1×g} @ (1_{1×Sk} ⊙ ReLU(Q_index @ K_index^T)) }`——每 token 对 g=64 个 head 的 QK^T 打分，ReLU 后按 g 维加权求和得到该 token 对 S2 的相关性分数，再取 top-k 索引（及分数值）。

### 1.3 metadata[1024] 里装了什么

`<ROOT>\op_kernel\lightning_indexer_v2_metadata.h:24-76`：
- 常量：`AIC_CORE_MAX_NUM=36, AIV_CORE_MAX_NUM=72`（上限；kernel 注释表明所用 950PR 为 **24 AIC + 48 AIV**，`lightning_indexer_v2_kernel_arch35.h:550-553` "vec:0-47 / cube:0-23"），`LI_V2_METADATA_SIZE=8, LD_V2_METADATA_SIZE=8`。
- 前 36×8 个 int32：**每个 AIC 的分核区间**（`LI_V2_*_INDEX`，行 33-40）：
  `[coreEnable, bn2Start, mStart(gS1块起点), s2Start, bn2End, mEnd, s2End, firstLdWorkspaceIdx]`。
  注意 end 是“开区间端点”，kernel 侧 `SplitCoreByAICPU`（`lightning_indexer_v2_kernel_arch35.h:294-366`）会按 s2End→gS1End→bN2End 优先级回退一格转成闭区间，只在跨界分支才需要读 1 次 `GetS1S2ActualSeqLen`（BSND 无 seqused 时 0 次 GM 读）。
- 随后 72×8 个 int32：**每个 AIV 的 LD（长 S2 跨核归约）任务**（行 43-49）：`[coreEnable, bn2Idx, mIdx, workspaceIdx, workspaceNum, mStart, mNum]`。
- 静态断言 36×8×4 + 72×8×4 = 3456B ≤ 4096B（行 71-76）。

对本场景（S2=128 ≤ s2BaseSize=128）：每个 gS1 块只有 1 个 s2 块，metadata 中 s2End 恒 0、无 S2 跨核切分 → **无 LD 任务**，`ldInfo.isLdCoreEnable=false`，`ProcessLD` 不执行；但 `ProcessDecode` 的 `SyncAll()` 仍执行（见瓶颈 5）。

### 1.4 两次 launch 的开销定性

- 主 kernel 本身（B=32）估算只有几十 µs（见 §3），而 metadata op 是一个只读几十个标量/小张量、写 4KB 的小 kernel。eager（aclnn 单算子）路径下每个 op 一次 launch：host 侧 executor 构造 + 下发（~5-15µs 量级）+ 设备端空隙；两次 launch 串行（主 op 数据依赖 metadata），无法在流内重叠。
- 结论：**小形状 eager 场景 metadata launch 占 e2e 的 15~40%**；graph/acl_graph 模式下两个节点同样串行（`<ROOT>\tests\pytest\lightning_indexer_v2_acl_graph.py` 只把主 op 捕进图，metadata 在图外）。BSND + 不传 seqlen 张量时 metadata 输出只依赖标量（形状/attr），**逐 call 完全不变，可缓存**（见优化 1）。

---

## 2. 数据流图（arch35 AIC/AIV 流水）与 v1 差异

### 2.1 分核与基本块（950 上的取值）

`<ROOT>\op_kernel\arch35\lightning_indexer_v2_kernel_arch35.h:89-99` 常量：`M_BASE_SIZE=256(=4×64，g≤32 时), S1_BASE_SIZE=4, S1_BASE_SIZE_SMALL=2, S2_BASE_SIZE=128, HEAD_DIM=128`；cube 侧 `M_BASIC_BLOCK=D_BASIC_BLOCK=S2_BASIC_BLOCK=128`（`lightning_indexer_v2_service_cube_arch35.h:58-66`）。

`InitTilingData`（kernel:206-213）：
```
if (gSize > 32 || topk > 2048) { mBaseSize = 2*g; s1BaseSize = 2; }   // 本场景 g=64 → m=128, s1Base=2
else                          { mBaseSize = 4*g; s1BaseSize = 4; }
s2BaseSize = 128
```
基本块 = (b, n2, gS1块, s2块)：gS1 块含 `s1BaseSize(2) token × g(64) = m=128 行`，s2 块 128 列。
本场景每 (b,n2)：gS1 块数 = ceil(128·64/128) = 64；s2 块数 = 1（mask0：ceil(128/128)；mask3：块 j 的有效列 n=min(2j+2,128)≤128 → 也是 1）。**B=32 → 2048 块；B=64 → 4096 块**；24 AIC → 每 AIC ~85/~171 块（metadata 按代价均衡，mask3 下块代价差异大，均衡有意义）。

cmp_ratio=r>1 时：`actS2Size = actS2SizeOrig/r`（kernel:281-291，`actS2SizeOrig = sequsedK·r + cmpResidualK[b]`，kernel:263-278）；s2 块数 = ceil(actS2Size/128)，mask3 每 token 有效长 `validAllS2Len=(i + actS2SizeOrig - S1 + 1)/r`（vector:413-431，base `GetMaskedS2BaseBlockNum`：<ROOT>\op_kernel\arch35\common\lightning_indexer_v2_kernel_base_arch35.h:37-55）。对 S2=128：块数不变、Mmad 的 n 缩小 r 倍、topk 更容易退化到 valid<topk 路径。

### 2.2 AIC/AIV 流水（每基本块）

以 (AIC_i, AIV_{2i}, AIV_{2i+1}) 为一组（`KERNEL_TYPE_MIX_AIC_1_2`，入口 `lightning_indexer_v2.cpp:49`；AIV 的 `aiCoreIdx = GetBlockIdx()/2`，kernel:549-555）：

```
            ┌────────────────────── AIC i (cube) ──────────────────────┐
 块 j:  WaitCross VC(j%2)×2   ← 两个 AIV 释放 UB 槽位 j%2
        KeyNd2Nz:  GM→L1, n×128×2B (每块重搬!)          [cube:234-249]
        QueryNd2Nz: GM→L1, 128×128×2B=32KB (仅首个s2块) [cube:285-301]
        LoadData L1→L0A/L0B ×2                          [cube:304-338]
        Mmad 128×n×128 → L0C(fp32)                      [cube:341-355]
        Fixpipe L0C→两个 AIV 的 UB（dualDstCtl=1, M/2=64行/核,
                 ROW_MAJOR_UB, SetMMLayoutTransform 开启 CV 直写） [cube:358-396, 401]
        SetCross CV(j%2)×2 → 通知两个 AIV
            └────────────────────────────────────────────────────────┘
                   │UB(64行×n×4B fp32/核, ping-pong×2 = 64KB)
            ┌──────▼─────── AIV_{2i} / AIV_{2i+1} (vector) ───────────┐
 块 j:  WaitCross CV(j%2)
        DataCopyPad w: 1 token × 64 × fp32 = 256B        [vec:330-338]
        BatchMulWeightAndReduceSum: 读 UB 64×n fp32, g 维加权规约
                 → 1 行 n 个 uint32 可排序 score key      [vec:347-349]
        DataCopyPad → scoreGm(GM workspace): 1行×128×4B=512B [vec:360-367]
        SetCross VC(j%2)   ← 释放槽位（在 topk 之前！cube 不被 topk 阻塞）
        ── 若 isLastS2InnerLoop（本场景每块都是）→ ProcessTopK ──
        读 scoreGm 512B → LiTopKVF(直方图 radix-select) 或 退化填充路径
        写 sparse_indices(±sparse_values) 到 GM             [vec:374-717]
            └────────────────────────────────────────────────────────┘
 全部块完成后: AIC 尾部 4×CrossCoreWaitFlag(kernel:779-784);
              AIV: ProcessDecode { InitLDBuffers; ICachePreLoad; SyncAll(); [ProcessLD] } (kernel:804-814)
```

关键流水参数：
- **ping-pong 深度 2**：UB 槽位 `loop%2`（cube:160-163/394，vec:311/345），AIC 最多领先 AIV 1 个块；topk 在 VC 释放之后执行，不阻塞 cube。
- **事件数**：每块 AIC 侧 2 wait + 2 set 跨核 + ~10 个核内 MTE1/MTE2/M/FIX 事件（cube:168-224）；AIV 侧 1 wait + 1 set 跨核 + ~10 个核内事件（vec:328-353, 478-499, 628-630）。
- workspace 布局（kernel:570-586）：`[scoreGm: GetBlockNum×s1Base×Align(S2,128)×4B][ldScoreGm][ldIndexGm]`，AIC/AIV 两侧同公式保证一致。
- `SCORE_T=uint32_t`（入口 `lightning_indexer_v2.cpp:52-53` 的 LIV2Type 实参），score 以可排序 uint32 key 存放，`UIntToFloatReturnValue`（`<ROOT>\op_kernel\arch35\vf\lightning_indexer_v2_vector1.h:23-66`）在输出前转回 float（key==0 → -inf）。

### 2.3 与 v1 的差异（分核方式 / 流水 / 退化点）

| 维度 | v1（`<V1>\op_kernel\arch35\`） | v2（`<ROOT>\op_kernel\arch35\`） | 对小形状的影响 |
|---|---|---|---|
| 分核 | **核内扫描**：每核 `GetTotalBaseBlockNum()` 全量枚举（v1 kernel:232-252）+ 顺序切分（v1 kernel:256-）| **metadata 预计算**：`SplitCoreByAICPU` 只读 ~8-14 个 uint32（kernel:294-404）；核内 `SplitCore` 变为 metadata==nullptr 的 fallback（kernel:561-566），但 950 tiling 强制 metadata 非空 → **fallback 是死代码** | v2 消除了每核 B×(2 个 GM 标量读/有 seqused 时) 的扫描开销，代价是多一次 launch + 强依赖 |
| g=64 的基本块 | `sparseCount>2048` 才降 s1Base=2；g=64/topk≤2048 用 **s1Base=4 (m=256) + splitMFlag**（v1 kernel:177-186，cube:194-228：一个块内切 2 次 m=128 的 Mmad/Fixpipe，**1 组 CV 同步**） | `g>32` 即降 s1Base=2（kernel:206-212） | **B=32 时 v2 块数 2048 vs v1 1024**：跨核同步、KeyNd2Nz、块级循环开销翻倍（Mmad/Fixpipe 总数不变）。这是 v2 明确的小形状退化 |
| mm1 结果通路 | arch22（910B）经 GM workspace mm1ResGm 往返；arch35 与 v2 相同（Fixpipe 直写 AIV UB） | Fixpipe dualDstCtl 直写两个 AIV 的 UB | 950 上两者相同，无差异 |
| 输出/语义 | 无 return_value 之外的 output_idx_offset、TND padding、cmp_ratio | 新增 output_idx_offset、TND padding（DoTndPadding→InitGlobalMemory）、cmp_ratio/cmpResidualK、return_value 完整化 | 新功能带来的固定分支判断，量级可忽略 |
| workspace | 950 分支只给 score 区（v1 tiling:1160-1164 ≈ libApi+48KB） | 950 分支额外加 24.6MB decode 区（tiling:1216-1218） | v2 每次调用多申请 ~24.6MB（见瓶颈 4） |
| LD（长 S2 跨核 topk 归约） | arch35 上 isLDOpen 恒 false，LD 实际不可达 | metadata 可下发 LD 任务（长 S2 时 ProcessTopK 落 ldScoreGm/ldIndexGm，ProcessLD 归约，vector:720-975） | 本场景 S2=128 无 LD；但 `SyncAll()` 恒执行 |
| topk kernel | 同为 VF 直方图 radix-select（v1 `vf_topk_gather.h` vs v2 `vf_topk_gather_v2.h`） | 同左，v2 泛化 topk≤8192 | 无实质差异 |

v2 没有比 v1 更重的“检查”路径进入 kernel（host 侧检查按题目要求忽略）；kernel 内新增判断（outputIdxOffset、needTndPadding、cmpRatio）都是 O(1)。

---

## 3. 量化成本模型（B=32/64 × topk=64/2048）

通用参数（g=64→s1BaseSize=2, mBaseSize=128, s2BaseSize=128；950PR 为 24 AIC + 48 AIV，1 AIC 配 2 AIV）：
- 基本块数：B=32 → 2048（每 AIC ~85.3）；B=64 → 4096（每 AIC ~170.7）。mask0 与 mask3 块数相同（S2=128=S2_BASE）。
- token 数：B×S1×N2 = 4096 / 8192；每 token 的 topk 由**单个 AIV** 完成（`blockId_%2` 行划分，vec:319-320），每 AIV ~85/~171 token。
- mask3 每 (b,n2) 的列和 Σn = Σ_{j=0..63}(2j+2) = 4160（token 有效长总和 8256，块取两 token 最大值多算 64）。

### 3.1 各阶段字节数 / FLOPs / 同步（B=32，rv=1 时 values 列加一倍）

| 阶段 | mask0/topk64 | mask3/topk64 | mask0/topk2048 | mask3/topk2048 |
|---|---|---|---|---|
| Mmad 次数（128×n×128） | 2048 | 2048 | 2048 | 2048 |
| Mmad FLOPs（2·m·n·k） | 8.59 GFLOP | 4.36 GFLOP（50.7%） | 8.59 GFLOP | 4.36 GFLOP |
| Q 读 GM→L1（Nd2Nz，每元素恰 1 次） | 67.1 MB | 67.1 MB（mask3 不省 Q） | 67.1 MB | 67.1 MB |
| K 读 GM→L1（**每块重搬**） | 67.1 MB（张量仅 1.05MB，**64×放大**） | 34.1 MB（32×放大） | 67.1 MB | 34.1 MB |
| Fixpipe L0C→UB（片上） | 134 MB | 68.2 MB | 134 MB | 68.2 MB |
| w 读（AIV） | 1.05 MB | 1.05 MB | 1.05 MB | 1.05 MB |
| scoreGm 写 + 读回（GM workspace） | 2+2 MB | 2+2 MB（恒写满 128 列，vec:362） | 2+2 MB | 2+2 MB |
| **输出写** indices (+values, rv=1) | 1.05 (+1.05) MB | 1.05 (+1.05) MB | **33.6 (+33.6) MB** | **33.6 (+33.6) MB** |
| 跨核同步（AIC 侧 4 ops/块） | 2048×4 = 8192 ops（每 AIC ~341） | 同左 | 同左 | 同左 |
| 核内事件 ops | ~2048×22（两 AIV+AIC 合计/块） | 同左 | 同左 | 同左 |
| per-token topk 指令（见 3.3） | ~150-250 VF 指令 | token0-62 走退化(~30)，63-127 走直方图(~200) | ~30(idx)+~200(rv 转换/填充) | 同左（更早退化） |

B=64：所有行 ×2（块 4096、token 8192、输出 topk2048 为 67.1+67.1 MB）。K 张量 2.1MB，放大倍数不变。

### 3.2 topk 两条路径的 per-token 开销（S2=128）

**路径 A：validS2Len ≥ topk（topk=64, mask0 的全部 token / mask3 的 token 63..127）** —— `vec:482-501` 单轮路径：
- scoreGm→UB：512B（`Align(128,256)=256` 补零，vec:487-497）；
- `LiTopKVF`（`<ROOT>\op_kernel\arch35\vf\vf_topk_gather_v2.h`）：对 ≤256 元素做 4 遍 8bit 直方图（HistogramsFirst/Second/Third/Last，各 1 个 vfLoop=256/256）+ 4 次 FindTargetBin（每次 4×(Arange/Load/Compare/Squeeze/StoreUnAlign)+后处理 ~12 条，`vf_topk_base_v2.h:56-101`）+ FindKth（4 loop，`vf_topk_gather_v2.h:266-314`）+ FindIdxGT/EQ（各 4 loop，`:316-374`）+（rv）FindValueOutput；合计 **~150-250 条 VF 指令/token**；
- UB→UB indices 拷贝 `Align(topk,256)`=256×4B=1KB（`lightning_indexer_v2_topk.h:75-77`）；出 GM 256B(+256B)。

**路径 B（退化）：validS2Len < topk（topk=2048 全部 token；topk=64+mask3 的 token 0..62；valid≤0 时走 vec:438-463 全 -1/-inf 直刷）** —— `vec:595-666`：
- `CreateVecIndex(valid)`（128 元素）+ indices 尾部 [-1] 填充 ~1920 元素（2 次 Duplicate，vec:614-626）；
- rv=1：读 score 512B（`Align(valid,32)×4`）→ **`UIntToFloatReturnValue(topkCountAlign256_=2048)` 全宽转换 ≈ 16 iter × ~10 指令 ≈ 160-190 条（仅 128 个有效，纯浪费，vec:637）** → [-inf] 填充 ~1856 元素（vec:640-653）→ 写 8KB；
- 出 GM：indices 8KB（+values 8KB）——**≥94% 是常量填充（-1/-inf）**。

**每 token 输出写入量**：topk=64 → 256B(+256B)；topk=2048 → 8KB(+8KB)。B=32 共 4096 token：topk2048/rv1 = 64MB 写（**用户提示的 32MB indices + 32MB values**，即 33.6+33.6MB 精确值）。

### 3.3 时间量级估算（假设需 profiling 校准）

假设：聚合 HBM 有效带宽 2.5~3.5 TB/s（950 量级假设）；48 AIV 均分写带宽；每 AIC GM→L1 有效 ~100-140 GB/s；VF 指令发射 ~1 条/cycle（~1.5GHz）。cube Mmad 吞吐按块级搬运（96KB GM/块）而非峰值 FLOPs 估算（128×128×128 的 Mmad 单条指令，L0A/B 装载与 Fixpipe 串在同槽）。

- **B=32/topk=64/mask0**：GM 读 67(Q)+67(K,大概率 L2 命中后实际 ~1)+2+1 ≈ 70-136MB → **~20-55µs**；cube 块流水 85 块×(0.5-1µs) ≈ 42-85µs（含 341 次跨核同步）；AIV 4096 token×(vec1 ~100-400 指令 + topk ~200 指令)/48 ≈ 15-35µs。三者部分重叠（ping-pong 深度 2），**预计 kernel ~40-80µs**。
- **B=32/topk=2048/rv=1**：输出写 67MB → **~19-27µs（纯带宽下限）**，AIV 侧 per-token ~230 指令 + 16KB 写 → ~40-60µs（AIV 成为主瓶颈之一）；cube 侧同 topk=64。**预计 kernel ~50-90µs，输出写+填充占主导**。
- **B=64**：全部 ×2 → ~80-180µs。
- **固定开销**（所有组合叠加在 e2e 上）：metadata launch ~5-20µs（eager）；主 op workspace 24.6MB 申请（首次/缓存失效时明显）；`SyncAll()` 栅栏 ~几 µs。

> 以上绝对时间均为模型推算，用于排相对优先级；建议用 `<ROOT>\tests\pytest\collect_perf_data.py`（读 op_summary 的 Task Duration）实测校准。

---

## 4. 瓶颈结论（按影响排序，附代码证据）

### 瓶颈 1（topk=2048 场景）：退化路径的输出写带宽与全宽填充
- B=32/rv=1 需写 33.6MB indices + 33.6MB values，其中有效数据仅 4096×128×4B=2.1MB（3%）；B=64 翻倍到 134MB。
- `UIntToFloatReturnValue` 按 `topkCountAlign256_=2048` 全宽转换（`lightning_indexer_v2_service_vector_arch35.h:637`），仅 128 有效；-1/-inf 尾部填充逐 token 重复（`:614-626, 640-653`）。
- 输出宽度由语义决定不可缩，但填充/转换方式可优化（优化 7/8）。

### 瓶颈 2（所有场景）：g=64 触发 s1BaseSize=2 → 块数与同步翻倍（v2 相对 v1 的退化）
- `lightning_indexer_v2_kernel_arch35.h:206-212`：g>32 即 s1Base=2/mBase=128；v1 仅 sparseCount>2048 才降，g=64 用 s1Base=4+splitM（`<V1>\op_kernel\arch35\lightning_indexer_kernel_arch35.h:177-186`，`<V1>\...\lightning_indexer_service_cube_arch35.h:194-228`）。
- 后果（B=32）：2048 块 vs 1024 块 → 跨核同步 8192 vs 4096 ops；KeyNd2Nz 2048 vs 1024 次（K 逻辑读 67 vs 34MB）；块级循环/事件开销翻倍；Mmad/Fixpipe 总数不变。
- 深层原因推测：s1Base=4 时 Q 的 L1 双缓冲被一个 256×128 tile 占满（QUERY_BUF_NUM=2×32KB，cube:133），s1Base=2 恢复双缓冲；但 S2≤128（单 s2 块）时双缓冲收益趋近 0，而块数翻倍的代价实打实。

### 瓶颈 3（所有场景）：K 每块重搬（64× 读放大）
- `lightning_indexer_v2_service_cube_arch35.h:167-175, 234-249`：每个基本块都 `KeyNd2Nz` 一次同一 (b,n2) 的 K tile（S2=128 时整段 32KB）。B=32/mask0 逻辑读 67MB vs K 张量 1.05MB。L2 可缓解，但挤占 L2 带宽与功耗；mask3 尾块 n 小时仍按 n 行搬（有改善）。

### 瓶颈 4（所有场景）：两次 kernel launch + 24.6MB workspace 过度申请
- metadata 必传：`<ROOT>\op_host\lightning_indexer_v2_tiling.cpp:589-591`（"Metadata must not be null"）；kernel 的 metadata==nullptr fallback（kernel:561-566）在 950 上不可达。
- workspace：tiling DAV_3510 分支 `lightning_indexer_v2_tiling.cpp:1211-1218`：
  `score区 = 4·ceil(S2/128)·128·4·aicNum`（S2=128, 24 AIC → 48KB，与 kernel 需求 24.6KB 匹配但用的是硬编码 li3510S1Base=4 而非实际 2）；
  `decode区 = 2·8(S1_BASE_SIZE=arch22值!)·2·8192(TOPK_MAX_SIZE!)·4·aicNum` ≈ **24.6MB**，而 kernel 实际 `GetBlockNum·s1Base(2)·Align(topk,16)·2·4·2`（kernel:583-586）在 topk=2048 时仅 ~1.5MB、topk=64 时 ~48KB —— **过度申请 16~500 倍**。v1 的 950 分支没有这个 decode 区（`<V1>\op_host\lightning_indexer_tiling.cpp:1160-1164`）。

### 瓶颈 5（所有场景）：ProcessDecode 的无条件 SyncAll + ICachePreLoad
- `lightning_indexer_v2_kernel_arch35.h:804-814`：48 AIV 全栅栏 + ICache 预取恒执行，即使 metadata 未启用任何 LD 核（本场景必然如此）。S2=128 时纯属浪费。

### 瓶颈 6（topk=64 场景）：per-token topk 固定指令开销与 score 的 GM 往返
- 每 token ~150-250 条 VF 指令（4 遍直方图是固定开销，与 validLen 无关——`Align(valid,256)` 后总是 ≥256 元素参与，`lightning_indexer_v2_service_vector_arch35.h:487`）。
- scoreGm 写 512B + 读回 512B 的 GM 往返（vec:367, 493-497）与 ~8 次核内事件，在单 s2 块场景完全可以在 UB 内直通（写 GM 仅为 LD/多 trunk 设计）。

### 瓶颈 7（mask3 场景）：尾块的微小 Mmad 与 PIPE_M barrier
- mask3 块 j 的 n=2j+2：前 ~15 块 n<32，Mmad 128×n×128 极小且 `(m/16)·(n/16)<10` 触发额外 `PipeBarrier<PIPE_M>`（cube:352-354），Fixpipe nSize=align(n,8)（cube:365）极窄；块数却与 mask0 相同。cube 利用率在 batch 开头塌陷（对总 FLOPs 是减半红利，但对块开销是全价）。

---

## 5. 优化机会清单

| # | 优化点 | 位置（绝对路径:行） | 思路 | 预期收益（本场景） | 风险 | 验证方法 |
|---|---|---|---|---|---|---|
| 1 | **静态形状缓存 metadata，消除第二次 launch** | `<ROOT>\torch_extension\lightning_indexer.py:130-147`（Python 层） | BSND 且不传 seqlen 张量时，metadata 输出仅是 (N1,N2,D,topk,B,S1,S2,layout,mask,cmp) 的纯函数 → 按 key 缓存 tensor 直接复用；kernel 只读 metadata，幂等安全 | 消除每次调用的 metadata launch（eager ~5-20µs，e2e 15-40%） | 极低；需保证形状/seqlen 变化时失效（有 seqlen 张量时不缓存或以张量内容 hash） | 修改前后 NPU profiler 对比 op 数量与 e2e；golden 全量回归（`test_lightning_indexer_v2_single.py`） |
| 2 | **允许 950 上 metadata 可选（小负载用核内 fallback）** | `<ROOT>\op_host\lightning_indexer_v2_tiling.cpp:589-591`；kernel fallback 已存在（kernel:561-566, 439-523） | B×N2 单元数小于阈值（如 ≤ 总核数×2）且无 seqlen 张量时放行 nullptr；核内 SplitCore 对 BSND 静态形状 0 次 GM 读，代价极小 | 动态小形状免一次 launch | fallback 路径目前无 metadata 调度（无代价均衡），mask3 负载不均；需限制在均匀场景 | UT（tests\ut\op_host\arch35）+ 单测补 nullptr 用例；对比分核正确性 |
| 3 | **修正 workspace 过度申请** | `<ROOT>\op_host\lightning_indexer_v2_tiling.cpp:1216-1218` | decode 区按 `aicNum·s1BaseSize(2)·Align(min(topk,8192),16)·2(idx/val)·4B` 计算（与 kernel:583-586 一致），删除 arch22 的 S1_BASE_SIZE=8/TOPK_MAX_SIZE=8192 常量 | workspace 24.6MB → ≤1.6MB；加快分配、降低图模式内存占用 | 低（新公式必须 ≥ kernel 需求，注意 s1BaseSize 在 g>32/topk>2048 时为 2、否则 4，取 max 覆盖两种） | 对比 tilings UT 中的 workspace 断言；跑 topk=8192/长 S2 的 LD 用例（STC TOPK_21-24）确认不越界 |
| 4 | **g=64 且 S2≤128 时恢复 s1BaseSize=4（或块合并）** | `<ROOT>\op_kernel\arch35\lightning_indexer_v2_kernel_arch35.h:206-212`；参考 v1 `<V1>\op_kernel\arch35\lightning_indexer_service_cube_arch35.h:194-228`（splitM） | 条件改为 `g>32 && (topk>2048 \|\| S2>s2BaseSize)` 才降 s1Base=2；短 S2 时用 v1 的 splitM（块内 2×m=128 的 Mmad/Fixpipe，1 组 CV 同步） | B=32 块数/跨核同步/K 搬运次数减半（Mmad/Fixpipe 总数不变）；预计块级开销 -30~50% | 中：UB/L1 布局、dualDstCtl M/2=128 行/AIV（每 AIV 2 token）需按 v1 验证；mask3 下块内两 token 的 valid 窗口取 max 会多算列（S2=128、s1Base=4 时每块多算 ≤2 列，可接受）；LD 长S2 路径仍需 s1Base=2 | 精度：golden 全量（重点 G_13/MASK_25/S2_16）；性能：collect_perf_data 对比 |
| 5 | **K tile 跨 gS1 块复用（L1 缓存）** | `<ROOT>\op_kernel\arch35\lightning_indexer_v2_service_cube_arch35.h:167-175, 224-225` | 同一 (bIdx,n2Idx,s2Idx) 的 K tile 在 L1 保留（记录 valid key），块间复用；metadata 的 AssignByBatch 保证同 batch 的块在同核连续 | K 逻辑读 67MB→1MB（mask0/B=32）；L2 压力解除；cube 块时间 -30~40% | 中：L1 容量与 Q 双缓冲共存（Q 64KB+K 96KB，L1 需 ≥160KB，950 L1 一般 512KB 可行）；跨 batch 边界要失效；PA 布局按 block 拼装需同样处理 | profiler 看 GM 读带宽/L2 命中率；数值不变（纯搬运复用） |
| 6 | **跳过无 LD 时的 SyncAll/ICachePreLoad** | `<ROOT>\op_kernel\arch35\lightning_indexer_v2_kernel_arch35.h:804-814` | 每 AIV 读 `ldInfo.isLdCoreEnable`（或全图无 LD 的 metadata 全局位）；全 false 时直接 return。所有核读同一 metadata，判断一致，无死锁风险 | 消掉 48 核栅栏 + 预取（~几 µs/次） | 低：必须保证“所有 AIV 都跳过”或“都执行”——以 metadata 中任一 LD 核使能为准的同一判定 | LD 用例（TOPK_23/24 长 S2）回归 + 无 LD 用例性能对比 |
| 7 | **退化路径：限制 UIntToFloatReturnValue 宽度** | `<ROOT>\op_kernel\arch35\lightning_indexer_v2_service_vector_arch35.h:637-638` | valid<topk 时只转换 `Align(validS2Len,128)` 个，尾部本就会被 -inf Duplicate 覆盖（`:640-653`） | topk=2048/S2=128 场景 AIV 指令减 ~150-190 条/token（约减半） | 低：需确认 [valid, topk) 区间确实全部被 -inf 覆盖（含 valid%8≠0 的 mask 段）；topk 非 8 倍数边界 | golden TOPK_22（valid<topk）+ 随机 valid 边界用例 |
| 8 | **退化路径：-1/-inf 填充模板化 / GM 端填充** | `lightning_indexer_v2_service_vector_arch35.h:614-626, 640-653`；参照 `CleanInvalidOutput` 已用 `InitGlobalMemory`（`:277-299`） | kernel 初始化时在 UB 预备一段 -1/-inf 模板（或对整段尾部用 InitGlobalMemory 直填 GM），每 token 只 DataCopy 有效前缀 + 一次常量填充 | 省 ~30 条 Duplicate/token 与相应 PipeBarrier；与 7 叠加后 AIV 侧仅剩 GM 写带宽下限 | 低-中：-inf 是 0xFF800000 按 uint32 Duplicate（已有先例 `:287`）；GM 端填充与 MTE3 排序需事件保护 | 同 7；重点 return_value=1 的 -inf 位逐位校验 |
| 9 | **单 s2 块时 score 走 UB 直通，跳过 scoreGm 往返** | `lightning_indexer_v2_service_vector_arch35.h:360-367`（写）与 `:493-497`（读） | `isLastS2InnerLoop && 单 s2 块 && !isNeedLD` 时 ProcessVec1 的输出行留在 UB（每 AIV 1 行×128×4B=512B，ping-pong 额外 1KB），ProcessTopK 直接消费 | 省 2+2MB GM 往返/调用 + 每 token ~6 次核内事件；AIV 关键路径缩短 | 中：UB 槽位与 resMm1 ping-pong 生命周期解耦（score 行须活到 topk，而 resMm1 槽位在 topk 前已释放——需独立小 buffer）；LD/多 trunk 路径不受影响（保留 GM 通路） | 精度回归 + Cycles 对比（重点 topk=64） |
| 10 | **mask3 尾块 Mmad 最小 n 对齐** | `<ROOT>\op_kernel\arch35\lightning_indexer_v2_service_cube_arch35.h:341-355, 365` | n<16 时对齐到 16（多算列由 vector 侧 validS2Len 截断，语义已保证），避免极窄 Mmad/Fixpipe 与额外 PIPE_M barrier 的最差组合 | mask3 batch 开头 ~15 块的发射效率；整体小 | 低：多算列的结果不进入 topk（validS2Len 截断），但需确认 vec1 读 UB 越界列安全（UB 槽位本就按 128 列分配） | golden MASK_25/26 回归 |
| 11 | **（结构性）加深 CV ping-pong 或批处理多块同步** | cube:160-163/227-230，vec:313/370 | 槽位 2→3（UB +32KB）或一次 CV 同步覆盖 2 个块（m=128×2） | 抖动容忍度提升；与 4 部分重叠，二选一 | 高：事件 ID 数量（CROSS_CV_EVENT+parity+AIV0_AIV1_OFFSET≤16）与 UB 预算（topk=2048 时已 ~185KB）受限 | 仅在 4/5 落地后仍见 AIC 等 VC 时考虑（profiler 事件统计） |

优先级建议：1、3、6、7（低风险高确定性，先做）→ 4、5（kernel 主流水，收益最大但需完整回归）→ 2、9、10 → 11。

关于“metadata 与主 op 两次 launch 是否可合并/流水”：
- **可缓存**（优化 1）：BSND 静态形状下等价于合并，收益最大、零风险。
- **可条件放行 fallback**（优化 2）：等价于合并进主 kernel（v1 就是这么做的），仅适合小而均匀的负载。
- **不可流水位**：主 op 逐核读 metadata（`SplitCoreByAICPU`），真数据依赖，同流内无法重叠；跨 step 双流预取属于调用方编排，不建议在算子内做。

---

## 6. 正确性约束（优化时必须保持的语义）

1. **无效索引 -1 / 无效值 -inf**：`INVALID_IDX=-1`（`<ROOT>\op_kernel\arch35\common\lightning_indexer_v2_const_info_common.h:37`）、`NEG_INF_FLOAT=0xFF800000`（`lightning_indexer_v2_common_arch35.h:89`）。出现位置：validS2Len<topk 的尾部填充（vector:614-626, 640-653）、validS2Len≤0 全无效（vector:438-463）、`CleanInvalidOutput`（actS1 不足时的补刷，vector:277-299，kernel_base：`DealActSeqLenIsZero`）、TND padding（`DoTndPadding`→`InitGlobalMemory`，common\lightning_indexer_v2_service_vector_base_arch35.h:24-45）、整图无任务时 `ProcessInvalid`（kernel:707-734，含 rv=1 的 -inf）。任何填充路径改动（优化 7/8/10）都必须逐位保持这两类哨兵值。
2. **mask_mode=3 的 per-token validS2Len**：`cuRealAcSeq = actS2SizeOrig - actS1Size + curAivS1Idx + 1`，`validAllS2Len = (i + cuRealAcSeq)/cmpRatio`（vector:413-431）；块级窗口取块内两 token 的 max（base `GetMaskedS2BaseBlockNum`:37-55，clamp 到 [1, actS2SizeOrig/r]）。S1>S2 时早期 token valid≤0 → 全 -1 路径（STC MASK_26）。改 s1BaseSize（优化 4）会改变块内 max 的多算列数，topk 输入长度必须仍按 per-token valid 截取。
3. **cmp_residual_k 语义**：`actS2SizeOrig = sequsedK·cmpRatio + cmpResidualK[b]`，`actS2Size = actS2SizeOrig/cmpRatio`（kernel:263-291）；每个元素 < cmpRatio；mask3 且 cmp_ratio>1 必传（docs:427）。它只影响窗口长度计算，不参与 score 数值。
4. **score key 语义**：score 以 uint32 可排序 key 存放/归约；key==0 视为无效，输出 value 时映射为 -inf（`lightning_indexer_v2_vector1.h:44-59` 的 zero→NAN→-inf 链）。即“真实加权和恰为 0”也会输出 -inf + 索引仍在（golden 按值排序对比，`lightning_indexer_v2_golden.py:1472`）——这是既定数值语义，勿在优化中改变。
5. **output_idx_offset**：偏移只加到 indices（`IndicesAddOffset`，topk.h:110-112；退化路径 `CreateVecIndex` 起点，vector:596），不加到 values；约束索引和 ≤ INT32_MAX（docs:425）。
6. **return_value=0/1**：rv=0 时 sparse_values 为 shape (0,) 空 tensor（csrc:100-104），topk 内 ISOUTVALUE=false 跳过 FindValueOutput（topk.h:66-74），CleanInvalidOutput/ProcessInvalid/DoTndPadding 均不刷 values；rv=1 时所有无效路径都要补 -inf。
7. **topk 顺序**：输出按 score 降序（测试侧再 sort 校验，等值内部顺序实现自定义）；等值时 GT/EQ 两段填充的索引升序约定不要破坏（`vf_topk_gather_v2.h:316-374`）。
8. **LD/长 S2 语义**（本场景不触发但改动需保护）：多 trunk merge（vector:502-594）、ProcessLD 归约（:720-975）、ldScore/ldIndex workspace 布局（kernel:579-586）。优化 3（workspace 公式）与 4（s1BaseSize）必须覆盖 topk>2048 与 S2>8192 的 STC 用例（TOPK_19-24）。
9. **metadata 一致性**：metadata 必须由参数一致的 LightningIndexerV2Metadata 产出（分核区间含 s1BaseSize 选择结果）；优化 1 缓存 key 必须包含 mask_mode/cmp_ratio/topk/layout/形状/seqlen；优化 2 放行 nullptr 时核内 SplitCore 的分核结果与 metadata 版可以不同（都正确，但性能特征不同）。

## 7. 附：关键文件速查

| 文件 | 作用 |
|---|---|
| `<ROOT>\op_host\lightning_indexer_v2_tiling.cpp:1189-1268` | blockDim(=CalcTschBlockDim(aiv,aic,aiv))、DAV_3510 workspace（:1211-1218）、tilingData/tilingKey |
| `<ROOT>\op_host\lightning_indexer_v2_tiling_info_parser.cpp` | arch35 专用 checker（DAV_3510 强制 :47-50）+ metadata 非空（tiling.cpp:589） |
| `<ROOT>\op_kernel\lightning_indexer_v2_metadata.h` | metadata[1024] 布局（AIC 36×8 + AIV 72×8） |
| `<ROOT>\op_kernel\arch35\lightning_indexer_v2_kernel_arch35.h` | 常量(:85-99)、s1BaseSize 切换(:206-213)、SplitCoreByAICPU(:294-404)、workspace 布局(:570-586)、Process/Main/Decode(:694-814) |
| `<ROOT>\op_kernel\arch35\lightning_indexer_v2_service_cube_arch35.h` | ComputeMm1(:158-231)、KeyNd2Nz(:234-249)、Fixpipe dualDst(:358-396)、事件与 L1/L0 布局(:44-66,128-145) |
| `<ROOT>\op_kernel\arch35\lightning_indexer_v2_service_vector_arch35.h` | ProcessVec1(:309-371)、ProcessTopK(:374-717)、ProcessLD(:720-975)、UB 预算(:156-189) |
| `<ROOT>\op_kernel\arch35\vf\lightning_indexer_v2_topk.h` + `vf_topk_gather_v2.h` + `common\vf\vf_topk_base_v2.h` | 直方图 radix-select topk（4×8bit pass） |
| `<ROOT>\op_kernel\arch35\vf\lightning_indexer_v2_vector1.h` | g 维加权规约、UIntToFloatReturnValue |
| `<ROOT>\torch_extension\lightning_indexer.py` + `csrc\lightning_indexer.cpp` | torch 双入口（metadata + 主 op） |
| `<ROOT>\tests\pytest\test_lightning_indexer_v2_single.py` / `_stc.py` / `lightning_indexer_v2_golden.py` | 用例构造（:1423-1470 metadata→主 op 调用）与覆盖说明 |
| `<V1>\op_kernel\arch35\*` | v1 对照（s1Base=4+splitM、核内 SplitCore、无 decode workspace） |
