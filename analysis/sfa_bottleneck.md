# SparseFlashAttention (SFA) 在 Ascend 950PR (A5/arch35) 上的小形状性能瓶颈分析

> 分析对象：`ops-transformer-9.2.0-attention-sparse_flash_attention` 中 SFA 算子的 A5 (DAV_3510 / arch35 / `__CCE_AICORE__==310`) 实现路线。
> 目标场景：B=32/64，S1=S2=128（每 batch 短序列 prefill），MLA-absorb（attention_mode=2），N1=64 query 头、N2=1（GQA group=G=64），D=512 + Dr=64（总 576），sparseIndices=[B,S1,N2,K]（K=sparseSize≈64/128，token 级，来自 lightning_indexer topk），sparse_mode=3（rightDownCausal），layout_query=BSND / layout_kv=PA_BSND（带 block_table），bf16/fp16。
> 性质：**只读代码分析，未修改任何源码**。所有结论均给出 `文件:行号` 证据；无法从本仓库源码确证的部分（A5 softmax VF 库不在本 checkout 内）已明确标注。

**路径缩写**（下文全部使用绝对路径的前缀缩写，ROOT = `D:\code\op_optimize\ops-transformer-9.2.0-attention-sparse_flash_attention\ops-transformer-9.2.0-attention-sparse_flash_attention\attention\sparse_flash_attention`）：

| 缩写 | 文件 |
|---|---|
| TILING | ROOT\op_host\sparse_flash_attention_tiling.cpp |
| TILING_H | ROOT\op_host\sparse_flash_attention_tiling.h |
| DEF | ROOT\op_host\sparse_flash_attention_def.cpp |
| ACLNN | ROOT\op_host\op_api\aclnn_sparse_flash_attention.cpp（v2 同目录 *_v2.cpp） |
| ENTRY | ROOT\op_kernel\sparse_flash_attention.cpp |
| KEYH | ROOT\op_kernel\sparse_flash_attention_template_tiling_key.h |
| KERNEL | ROOT\op_kernel\arch35\sparse_flash_attention_kernel_mla_arch35.h |
| CUBE | ROOT\op_kernel\arch35\sparse_flash_attention_service_cube_mla_arch35.h |
| VEC | ROOT\op_kernel\arch35\sparse_flash_attention_service_vector_mla_arch35.h |
| KVCACHE | ROOT\op_kernel\arch35\sparse_flash_attention_kvcache.h |
| GOLDEN | ROOT\tests\pytest\sparse_flash_attention_golden.py |
| PROC | ROOT\tests\pytest\batch\sparse_flash_attention_process.py |

---

## 0. 结论摘要（TL;DR）

1. **A5 kernel 的任务粒度 = 1 个 query token**（`qSNumInOneBlock=1`，KVCACHE:98）。每个 (b, s1) 任务独立完成一次「读 sparse_indices → 逐 token gather KV → UB→GM workspace → AIC ND2NZ 进 L1 → bmm1 → softmax → bmm2 → 写回」全流程。B=32/S1=128 时共 4096 个任务，**每个 token 的 KV（≤128×576×2B≈144KB）被完整搬运 4 遍（HBM/L2→UB→GM→L1→L0B），KV 流量相对"每 batch 只装一次 KV cache"放大约 64×**（K=128 时每 batch 平均重读 64.5 遍 KV）。这是该场景第一结构性瓶颈。
2. **gather 循环里逐 token 的标量 GM 读**（`sparseIndicesGm.GetValue` + PA 下 `blockTableGm.GetValue`，VEC:192/198/214）构成串行依赖链：每任务约 2n 次标量 GM load（B=32/K=128 全网约 53 万次），估算占 AIV 侧数百 µs，很可能是单点最大耗时（需 msprof 确认）。
3. **AIC 侧每任务固定搬运远大于计算**：每任务 Q 装载 73.7KB（ND2NZ，CUBE:282-288，Q 为单 buffer L1，不可跨任务复用）+ KV ND2NZ（CUBE:222）+ bmm2 fixpipe 128KB fp32（CUBE:325-338）+ Vec2 的 64KB UB→UB 中转（VEC:559）。按 24 AIC 估算 AIC 每 1 FLOP 要搬 ~16B，AIC 明显搬运受限（估算 B=32/K=128 时 AIC 搬运 ≈694µs vs 纯 MMA 96µs）。
4. **sparseBlockSize 在 A5 上只允许 1**（TILING:793-797），且 kernel 侧硬编码 `constInfo.sparseBlockSize = 1`（KERNEL:417）——**用户关心的 64/128 block 级索引在 A5 当前版本直接被 tiling 校验拒绝**，block 级连续搬运（A2/A3 支持，见 arch22 分支）在 A5 上完全缺失，这是最大的"缺失的优化"。
5. splitKV/FD（flash-decoding）在当前版本**永远不触发**（`splitKVFlag_` 无任何置 true 的代码，TILING:363/434-436；tiling key 的 FLASH_DECODE 恒为 0，TILING:308-310）。S2=128 时本也无需 FD，结论：FD 分支对本场景无影响。
6. 分核方面 **核数能打满**：任务数 = B×S1（B=32→4096 ≥ 24 AIC），`usedCoreNum_=aicNum_`（TILING:391），无核饥饿问题；存在 ±15% 左右的静态不均（任务按 (b,s1) 顺序连续切分，而每任务工作量 n(p)=min(p+1,K) 随 p 递增）。
7. 两个**功能性陷阱**需要注意：(a) `return_softmax_lse=true` 与 `layout_kv=PA_BSND` 互斥（TILING:1992-1996，直接报错）；(b) A5 kernel **完全不读 value 张量**（VEC:688-696 只 set 了 key/blockTable/sparseIndices/keyRope），语义上强制 `value == key[..., :512]`（MLA-absorb 约定，GOLDEN:717 同）。
8. 综合估算（假设见 §3.1）：B=32/K=128 端到端 ≈ **0.6–1.2 ms**，B=64 ≈ 1.2–2.4 ms；而 cube roofline 仅 96/192 µs —— **当前实现距 roofline 约 5–10×**，优化空间主要在 §5 的 O1–O5。

---

## 1. 调用协议

### 1.1 torch 侧（实际调测方式）

`tests/pytest/utils.py::sfa_run_npu` → `batch/sparse_flash_attention_process.py::call_npu`，eager 路径核心代码（PROC:29-57）：

```python
kwargs = {
    "query":   t["query"],            # [B,S1,N1,D]  (BSND)  bf16/fp16
    "key":     t["key_cache"],        # PA_BSND: [block_num, block_size, N2, 512]
    "value":   t["value_cache"],      # 同 key_cache（A5 语义上 V=key[:,:512]，见 §6）
    "sparse_indices": t["sparse_indices"],   # [B,S1,N2,K] int32
    "scale_value": params["scalevalue"],     # 1/sqrt(576)
    "block_table": t["block_table"] if layout_kv=="PA_BSND" else None,  # [B,maxBlockNumPerBatch] int32
    "actual_seq_lengths_query": tensor(params["actual_seq_q"], int32).npu(),
    "actual_seq_lengths_kv":   tensor(params["actual_seq_kv"], int32).npu(),
    "query_rope": t["query_rope"],   # [B,S1,N1,64]
    "key_rope":   t["key_rope_cache"],# PA: [block_num, block_size, N2, 64]
    "sparse_block_size": 1,           # A5 仅支持 1（TILING:793）
    "layout_query": "BSND", "layout_kv": "PA_BSND",
    "sparse_mode": 3,                 # rightDownCausal
    "pre_tokens": (1<<63)-1, "next_tokens": (1<<63)-1,   # 仅支持默认值（TILING:818-836）
    "attention_mode": 2,              # MLA，仅支持 2（TILING:861-865）
    "return_softmax_lse": False,      # true 时 PA_BSND 会被拒（TILING:1992-1996）
    "sinks": t.get("sinks"),          # 仅 A5 支持（ACLNN_V2:56-60）
}
out = torch_npu.npu_sparse_flash_attention(**kwargs)
```

图模式（`call_npu_graph`，PROC:133-175）走 torchair `aclgraph` 静态 shape 编译，同一 kwargs。

### 1.2 aclnn 侧

两段式接口（ACLNN、ACLNN_V2 外层仅做 holder/空指针处理，内部都进 `aclnnInnerSparseFlashAttention`）：

```
aclnnSparseFlashAttention(V2)GetWorkspaceSize(query, key, value, sparseIndices,
    blockTableOpt, actSeqQOpt, actSeqKvOpt, queryRopeOpt, keyRopeOpt, sinksOpt(V2),
    scaleValue, sparseBlockSizeOpt, layoutQueryOpt, layoutKvOpt,
    sparseMode, preTokens, nextTokens, attentionMode, returnSoftmaxLse,
    attentionOut, softmaxMax, softmaxSum, &workspaceSize, &executor)
aclnnSparseFlashAttention(V2)(workspace, workspaceSize, executor, stream)
```

算子注册（DEF:23-93）：输入 `query/key/value`（fp16/bf16，REQUIRED）、`sparse_indices`（int32）、`block_table/actual_seq_lengths_query/actual_seq_lengths_kv/query_rope/key_rope`（OPTIONAL）、`sinks`（fp32，OPTIONAL）；输出 `attention_out`（fp16/bf16）、`softmax_max/softmax_sum`（fp32，REQUIRED——lse=false 时由 aclnn 层 TensorHolder 造 {0} 形状占位，ACLNN:47-87）。ascend950 配置下 key/value/key_rope 允许 0 轴非连续（DEF:104-126，即 PA cache 按 block 散排）。

### 1.3 输入输出清单（目标场景）

| 张量 | shape | dtype | 说明 |
|---|---|---|---|
| query | [32/64, 128, 64, 512] | bf16 | BSND |
| query_rope | [32/64, 128, 64, 64] | bf16 | 必传（TILING:874-878） |
| key | [blockNum, blockSize, 1, 512] | bf16 | PA_BSND，blockSize 16 倍数 ≤1024 |
| key_rope | [blockNum, blockSize, 1, 64] | bf16 | |
| value | 同 key | bf16 | **A5 不读**，语义须等于 key[...,:512] |
| sparse_indices | [32/64, 128, 1, K] | int32 | 有效值在前、-1 补齐在后（doc 约束，aclnnSparseFlashAttentionV2.md:162） |
| block_table | [B, maxBlockNumPerBatch] | int32 | S2 = maxBlockNumPerBatch×blockSize（TILING:2047-2054） |
| actual_seq_lengths_q/kv | [B] | int32 | PA 时两者必传（TILING:955-967） |
| attention_out | [B, 128, 64, 512] | bf16 | 与 query 同 layout |
| softmax_max / softmax_sum | [B, 1, 128, 64] | fp32 | lse=true 才有意义；**PA+true 被拒** |

tiling key：`GET_TPL_TILING_KEY(0, pageAttention, layoutQuery, layoutKV, perfMode==V_TEMPLATE, gSize>64)`（TILING:308-310；KEYH:43-90）。目标场景 gSize=N1/N2=64 → `IS_SPLIT_G=0`（`gSize>64` 不成立，且 TILING:492-493 的 workspace 减半分支也不触发）；sparseBlockSize=1≤4 → `perfMode=V_TEMPLATE`（TILING:329-333），kernel 侧映射为 CFA 模板（KEYH:27-28、KERNEL 模板实参 ENTRY:88-96）。

---

## 2. A5 kernel 整体结构与数据流

### 2.1 编译/分核框架

- 入口按 `__CCE_AICORE__==310` 选 arch35（ENTRY:22-26）；`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`（ENTRY:82）→ **1 AIC : 2 AIV**，`subBlockIdx=GetSubBlockIdx()` 区分同一 AIC 的两个 AIV（KERNEL:133），AIV 的 `aicIdx = aivIdx>>1`（KERNEL:140）。
- tiling 侧 `usedCoreNum_=aicNum_`、`blockDim=CalcTschBlockDim(2*aic, aic, 2*aic)`（TILING:391、531-539）；kernel 侧 `usedCoreNum` 初值取 tiling 值（VEC:816 读 singleCoreParams），任务数 > 核数且非 SPLIT_G 时不再下调（KERNEL:203-214）。
- **任务 = (b, n2, s1 token)**：`InitCalcParamsEach` 以 `actBatchS1×actBatchS2(=1)` 累计 `sfaTotalBaseNum`（KERNEL:187、200），`avgBaseNum=ceil(total/coreNum)` 连续切分（KERNEL:230-251）。B=32 → 4096 任务，24 AIC（假设）→ 每核 171 个。
- 基本块大小**硬编码**：`s1BaseSize=64, s2BaseSize=128`（KERNEL:145-146、CUBE:55-57、VEC:44-45）。tiling 里的 `innerSplitParams.mBaseSize/s2BaseSize`（=gSize/min(512,S2)，TILING:386-389）**A5 kernel 完全不读**（arch35 目录 grep 无 `innerSplitParams`；A2 才读，arch22 kernel:217-218）——即 mBaseSize=gSize=64 只是"每任务 M=G×1 token"的另一种表述。

### 2.2 每 task 的 M/N/K 与 mask

- 每 task：`qSNumInOneBlock=1`（KVCACHE:98，"sfa 不切G轴"），`s1RealSize=1`、`mRealSize=s1RealSize×gSize=64`（KVCACHE:125-131）→ **bmm1=[64(G), 576]×[576, n]，bmm2=[64, n]×[n, 512]**，n=本 token 有效 KV 数。
- 两个 AIV 各负责 M 的一半（halfMRealSize=32；KERNEL:664-689），bmm1/bmm2 的 L0C 经 fixpipe `dualDstCtl` 双目的写分别进两个 AIV 的 UB（CUBE:255-261、325-338）。
- mask（sparse_mode=3，S1=S2 时 nextTokens=0、preTokens=S1，KVCACHE:79-84）：对 query token p，
  `s2LineEnd = clip(p + nextTokens + 1, 0, S2)` 再 `min(·, sparseBlockCount)`（KVCACHE:251-260，注释"当前LI输出的block size只可能是1"），`s2Start=clip(p-preTokens,0,S2)=0`；`kvLoopEndIdx=ceil(s2LineEnd/128)`（KVCACHE:262）→ **K≤128 时每 task 恰好 1 个 s2 loop，n=min(p+1,K)**。golden 语义一致（GOLDEN:624-646：threshold=Δs+p+1，取前 validCount 个索引，≥threshold 的 block 跳过/截断）。
  - 注意：A5 的 gather（VEC:176-225）**不**像 A2（arch22:905-907、884-885）那样在搬运侧按 s2IdLimit 裁剪/压实无效 token；A5 依赖「索引本身在因果窗口内 + 有效值在前/-1 在后（doc:162）+ 计数截断」，列级掩码在 softmax VF（`vselrIndexesBuf`+`negativeFloatScalar`，VEC:503-537）内完成。**VF 源码（common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_sfa.h）不在本 checkout 中，无法直接审计该掩码逻辑**——这是需要在真机上用 -1 padding 用例验证的点（见 §6.4）。

### 2.3 数据流图（单 core，1 AIC + 2 AIV；任务 t 的 5 段流水，跨任务 3 深度）

```
        GM(HBM/L2)                          UB(AIV)                 L1(共享)            L0A/B/C        UB
  ┌──────────────────────────┐   ┌────────────────────────┐  ┌──────────────────┐  ┌───────────┐
  │ sparse_indices[b,p,0..K) │◄──Scalar GetValue×2/pair──┐ │                    │  │           │
  │ block_table[b, idx/bs]   │◄──Scalar GetValue×2/pair──┤ │                    │  │           │
  └──────────────────────────┘   │                      │ │                    │  │           │
  ┌──────────────────────────┐   │ Vec0(AIV×2, task t)  │ │                    │  │           │
  │ key cache(散)  1024B/token│══MTE2: DataCopyPad══════►│stage0Out 16行×576  │  │           │
  │ key_rope(散)   128B/token │══(成对合并,2条指令/pair)═►│×2 ping-pong        │  │           │
  └──────────────────────────┘   │                      │ │                    │  │           │
                                 │      MTE3: 整块搬出   ▼                    │  │           │
  ┌──────────────────────────┐   │            ┌────────────────────┐         │  │           │
  │ v0ResGm workspace        │◄═══════════════╡ UB→GM 压实写(dealRow×1152B)  │  │           │
  │ (每核3槽×128×576×2B)     │   │            └────────────────────┘         │  │           │
  └────────┬─────────────────┘   │  outputL1.SetCrossCore / v0ResGm.SetCrossCore ──► AIC     │
           │                     └────────────────────────────────────────────┴───────────────┘
           │  Bmm1(AIC, task t)
           ├═MTE2+ND2NZ(仅s2Loop==0)═ Q[b,p]: nope[64,512]+rope[64,64] ──► l1Q(单buffer 73.7KB)
           ├═MTE2+ND2NZ═══ KV n×576 ──► l1Right 3槽(147.5KB/槽: K^T NZ + [偏移131072B]P + V=nope NZ 复用)
           │                     MTE1: L1→L0A(Q)/L0B(K^T)          MMA bmm1[64,576]×[576,n]→L0C fp32
           │                     Fixpipe(dualDst): L0C→两AIV的UB bmm1Res(2槽×16KB/AIV) ──SetCrossCore──► AIV
           │  Vec1(AIV×2, task t): softmax VF(3档N分支: ==128/≤64/<128, VEC:501-537)
           │      mmRes[32,n]fp32 → (mask vselr) → max/exp/sum(fp32) → P量化bf16[32,n]
           │      MTE3: P→L1槽内偏移128×512处(strided NZ写, VEC:474-480) ──SetCrossCore──► AIC
           │      (lse=true 且最后loop: CopyFALseToGm 写 softmaxMax/Sum, VEC:489-491,712-742)
           │  Bmm2(AIC, task t): MTE1: L0A(P)/L0B(V=K-nope NZ)
           │      MMA bmm2[64,n]×[n,512]→L0C fp32; Fixpipe(dualDst)→UB bmm2Res(单buffer 64KB/AIV)
           │      ──SetCrossCore──► AIV
           │  Vec2(AIV×2, task t): UB→UB 64KB 中转(VEC:559) → LastDivNew(÷sum, fp32) → Cast bf16
           │      MTE3: DataCopyPad 32块×1KB → attentionOut GM(VEC:586-606)
```

流水与同步（KERNEL:526-611 主循环）：

- 跨任务 **3 深度软件流水**：第 t 轮迭代发射 `IterateBmm1(runInfo[t%3])`+`ProcessVec0(runInfo[(t+1)%3])`，随后 `IterateBmm2(runInfo[(t+2)%3]前一槽)`+`ProcessVec1(...)`，再 `ProcessVec2(...)`（KERNEL:567-607）。启动阶段有 PRE_LOAD_NUM=2 的预热（KERNEL:534-549），即 AIV 比完成侧领先 2 个任务。
- 关键跨核同步（每 task 约 8~10 次 Set/Wait）：Vec0→AIC（L1 槽 + GM 槽 forward，KERNEL:571-577）、AIC→Vec1（bmm1Res UB both）、Vec1→AIC（P L1 forward）、AIC→Vec2（bmm2Res UB both）；另有各引擎内部 MTE2/MTE3/V 事件（如 VEC:425-436 的 ping-pong MTE3_MTE2、CUBE:216-221 的 WaitCrossCore、MatmulK 后的 `mm1ResL0C.Set/Wait` ×2）。
- L1 布局（KERNEL:335-388、CUBE:160-171）：L1 总 512KB = `l1Right` 3 槽×147.5KB（442KB；注释：保存 P 的 L1 必须在第一个 policy 上，与 vec 申请地址一致）+ `l1Q` 单 buffer 73.7KB；L0A/L0B 各 64KB、L0C 256KB；mmL0A/B/C 以 16K/32K/128K 双 buffer 管理。UB 侧 `bmm1Res` 双槽×16KB/AIV、`bmm2Res` 单 64KB/AIV（KERNEL:316-334）；`stage0Out` 16 行×576×2B×2 ping-pong（VEC:769-772）、`stage1OutQue` 33×128×2B×2（VEC:774-775）。
- **启动同步**：AIV 将 tiling→CVSharedParams 写入 ssbuf 后 `CrossCoreSetFlag<…>(15)`，AIC `CrossCoreWaitFlag<…>(15)` 后逐字拷回（KERNEL:165-171）；`needInit` 时全体 AIV 先对输出全量 memset 并 `SyncAll`（VEC:618-682、KERNEL:471-473 附近）——本场景（act_q 全 128）不触发，但变长 batch 会触发（见 §4-B6）。

### 2.4 workspace 结构（问题 6）

A5 分支（TILING:485-495）：

```
workspaceSize = libapiSize + 128(S2_BASE) × 576(D) × 2B(dtype) × 3(槽) × aicNum
             = libapiSize + 442,368B × aicNum        （gSize>64 时 aicNum 减半）
```

- 24 AIC 假设下 ≈ **10.6MB**（+ libapi 通用空间）；每核 3 槽 × 144KB，即 §2.3 的 `v0ResGm`。每核基址 `workspace + aicIdx×3×144KB`（KERNEL:373-387，SPLIT_G 时 `aicIdx>>1`）。
- **FD/splitKV workspace 恒为 0**：`splitKVFlag_` 从未被置 true（grep 全文件仅在 363/465/469 判断），`FillTilingSplitKVMla` 最终 `set_s2(0)`（TILING:434-436），`NormalCalcFDWorkSpace` 不累加（TILING:463-475）。A2 分支的 mmRes/bmm2Res/topk 聚合（`4×512×576×2×actCoreNum` 等，TILING:509-525）A5 全部不需要——A5 的中间结果都在 UB/L1 里，只有 KV gather 借道 GM。

---

## 3. 量化成本模型

### 3.1 假设参数（结论对这些参数敏感，建议用 msprof 实测替换）

| 参数 | 取值 | 依据/说明 |
|---|---|---|
| AIC 数 A | 24（950PR，假设值） | tiling 运行时取 `GetCoreNumAic()`；AIV=48。若实际为 20/28，按比例缩放 |
| 单 AIC cube 峰值 P_c | 16 TFLOPS(bf16) | 量级假设（16×16×16 MMA/周期@~2GHz）；请以规格书替换 |
| 单核 MTE2/MTE3/fixpipe 有效带宽 B_m | 100 GB/s | 保守经验值；L1→L0(MTE1) 取 200 GB/s |
| 标量 GM load（L2 命中）延迟 | 50–100 ns | gather 循环内串行依赖链 |
| HBM 带宽 | ~1.2 TB/s | 本场景 HBM 不是第一瓶颈（见下），影响小 |
| Σn（每 batch 有效 KV 行数） | K=128 → 8256；K=64 → 6176 | n(p)=min(p+1,K)，S1=S2=128，sparse_mode=3 |
| 任务数 | B×128 | B=32→4096（171/核），B=64→8192（341/核） |
| FLOP/task | 139,264×n（bmm1 73,728n + bmm2 65,536n） | M=64，K1=576，N=n；K2=n，N2=512 |
| 字节/task | KV 三跳各 1,152×n B；Q 73,728 B；MTE1 73,728+2,304n；fixpipe 256n+131,072；输出 65,536 | 见 §2.3 数据流 |

### 3.2 各维度总量（bf16）

**每 batch（S1=S2=128）**：

| 量 | K=128 | K=64 |
|---|---|---|
| FLOPs | 1.149 GFLOP | 0.860 GFLOP |
| KV gather 读（AIV MTE2） | 9.50 MB | 7.11 MB |
| KV workspace 写（AIV MTE3） | 9.50 MB | 7.11 MB |
| KV GM→L1（AIC MTE2, ND2NZ） | 9.50 MB | 7.11 MB |
| Q 装载（AIC MTE2, ND2NZ，每 token 固定） | 9.44 MB | 9.44 MB |
| MTE1（L1→L0） | 28.5 MB | 23.7 MB |
| fixpipe（L0C→UB fp32） | 18.9 MB（其中 bmm2 固定 16.8MB） | 18.4 MB |
| 输出写回 | 8.39 MB | 8.39 MB |
| 标量 GM load（idx+blockTable） | 2×8256=16.5K 次 | 12.4K 次 |
| KV 放大倍数（相对 147.5KB 全量 cache） | 64.5× | 48.3× |

**整算子（B=32 / B=64）**：

| 量 | B=32,K=128 | B=32,K=64 | B=64,K=128 | B=64,K=64 |
|---|---|---|---|---|
| 任务数 | 4096 | 4096 | 8192 | 8192 |
| FLOPs | 36.8 G | 27.5 G | 73.6 G | 55.0 G |
| KV 单跳流量（×3 跳） | 304 MB | 228 MB | 608 MB | 455 MB |
| Q 装载 | 302 MB | 302 MB | 604 MB | 604 MB |
| 输出写 | 268 MB | 268 MB | 537 MB | 537 MB |
| MTE1 总量 | 911 MB | 757 MB | 1821 MB | 1515 MB |
| fixpipe 总量 | 605 MB | 588 MB | 1209 MB | 1175 MB |
| 标量 load 总数 | 528K | 395K | 1.06M | 791K |

注意两个结构性事实：**Q+输出 = 570MB（B=32）与 K 无关**，是该 shape 的下限流量；K=64 时 KV gather（228MB）甚至小于 Q 装载（302MB）——「稀疏」省下的计算没有省下固定搬运。

### 3.3 估算耗时分解（B=32，K=128；24 AIC/48 AIV）

| 阶段 | 每核负载 | 估算耗时 | 说明 |
|---|---|---|---|
| cube MMA（roofline） | 1.53 GFLOP | **96 µs** | 36.8G/24核/16TF |
| AIC MTE2（Q+KV ND2NZ） | 25.2 MB | ~252 µs | @100GB/s；与 MMA 在现设计里基本串行 |
| AIC MTE1（L1→L0） | 37.9 MB | ~190 µs | @200GB/s |
| AIC fixpipe（bmm1+bmm2） | 25.2 MB | ~252 µs | @100GB/s |
| **AIC 合计（搬运为主）** | — | **~500–700 µs**（若完全串行 694µs；理想重叠下限 ~252µs） | MMA 占比 <20% |
| AIV Vec0 标量链 | 11.0K 次 load | **~0.55–1.1 ms** | 50–100ns/次，串行依赖（见 §4-B2） |
| AIV Vec0 MTE2/MTE3 | 各 6.3 MB | 各 ~63 µs | |
| AIV Vec1 softmax | 每 task 32×n 元素 | ~0.3–1 µs/task → 50–170 µs/核 | exp/规约/量化+P→L1 |
| AIV Vec2（UB 中转+除+cast+写出） | 11.2 MB UB拷贝 + 5.6 MB 输出 | ~170 µs/核 | UB→UB 64KB/task 是纯浪费（§5-O6） |
| 端到端估计 | — | **~0.6–1.2 ms** | 受 max(AIC 链, Vec0 标量链) 限制 |

B=64 线性翻倍 ≈ 1.2–2.4 ms；K=64 时 FLOP −25%、KV −25%，但 Q/输出/fixpipe 固定部分不变，估计 −15~20%。

**与 roofline 的差距**：96 µs vs 0.6–1.2 ms → **6–12×**。即使 AIC 各引擎完美重叠（下限 ~252 µs），Vec0 标量链（≥550 µs）仍单独超标 5×以上——这就是小形状场景"算得少、搬得多、等得多"的量化画像。

---

## 4. 瓶颈结论（按影响排序，附证据）

**B1（结构性，★★★）每 token 一个任务的 KV gather 流水，KV 数据被搬运 4 遍、跨任务零复用**
- 任务粒度 1 token：KVCACHE:98（`qSNumInOneBlock=1`）、KERNEL:200（任务数=Σ actS1）。
- 每 task 全套 gather：VEC:404-450（ProcessSparseKv）、VEC:342-348（UB→GM workspace）、CUBE:222-223（GM→L1 ND2NZ，每 task）。
- 后果：KV 放大 48–64×（§3.2）；B=32/K=128 时仅 KV 三跳就有 912MB 片内/互连流量。而每 batch 的全量 KV 只有 147.5KB，L1（512KB）甚至能整批驻留。
- 每 task 固定开销被 ×4096/×8192：Q ND2NZ 73.7KB（CUBE:282-288，`s2LoopCount==0` 每 task 必发；l1Q 单 buffer，KERNEL:346-350、CUBE:160-161）、bmm2 fixpipe 128KB fp32（CUBE:325-338）、~10 次跨核 Set/Wait（§2.3）、5 段串行链 Vec0→bmm1→Vec1→bmm2→Vec2。

**B2（单点最大嫌疑，★★★，需 profiling 定量）gather 循环的标量 GM 读串行链**
- VEC:192/198（`sparseIndicesGm.GetValue`×2/对）、VEC:214（`blockTableGm.GetValue`/token）：每对 token 4 次标量 GM load，且 idx→blockTable→DataCopyPad 地址计算构成**串行依赖**（VEC:424-436 内层循环），无法靠 MTE 异步掩盖。
- 每 AIV 每 task ≈ n 次 load（B=32/K=128 全网 528K 次）；50–100ns/次 → 每 AIV 0.55–1.1 ms（§3.3）。**这是最先该用 msprof（AIV Scalar/MTE 队列占比）验证的数字。**
- 对比：A2 同样用 GetValue（arch22:837/852），但 A2 场景 sparseBlockSize 可 >1，load 次数按块数缩。

**B3（★★★）AIC 侧搬运/计算比严重失衡**
- 每 task AIC 搬运 = Q 73.7KB + KV 1,152n B（MTE2）+ MTE1 73.7KB+2,304n + fixpipe 256n+131KB ≈ 345KB（n=64.5），而 MMA 仅 9.0 MFLOP（0.56µs@16TF）→ 每 FLOP 搬 ~16B；AIC 各引擎时间估算 694µs vs MMA 96µs（§3.3）。
- 证据行号见 B1/B4 各条；Q 单 buffer 使下一 task 的 Q 拷贝必须等本 task bmm1 的 MTE1 释放（CUBE:273-291 的 Wait/Set 配对）。
- bmm2 的 fixpipe 128KB fp32/task 是最大单项：L0C[64,512] 每 task 全量落 UB，随后 Vec2 再 UB→UB 拷贝一次（VEC:558-559）——同一数据在 AIV 侧被触碰两次。

**B4（★★）token 级小搬运的访存形态**
- 修正一个直觉：A5 的 token 级 gather **不是**逐 token 32B 搬运。`CopyInKvSparse`（VEC:260-302）把相邻两个索引合并成 1 条 `DataCopyPad`（blockCount=2）：K-nope 每块 **1024B** 连续（512×2B，VEC:282-286）、K-rope 每块 **128B**（VEC:291-299）；两索引相邻时 srcStride=0 变成 2KB 连续。16 行一批经 UB ping-pong（VEC:424-436）再整块 MTE3 出 GM（VEC:342-348）。
- 残余问题：(a) rope 的 128B 块太小（4×32B），但 rope 张量紧凑（每 batch 16KB），L2 命中率高，属次要；(b) 索引随机时 srcStride 巨大但仍是一条指令，HBM 侧 1KB 随机读效率依赖 L2 聚合——需 profiling 看 MTE2 实际带宽；(c) **每 task 都从 GM 重新读索引**（512B 行），跨 task 无预取。
- block 级（sparseBlockSize=64/128）在 A5 **不可用**：TILING:793-797 直接报错；kernel 亦硬编码 `constInfo.sparseBlockSize=1`（KERNEL:417）。若可用（如 A2/A3，arch22:887-897 的 `blockLen=sparseBlockSize×headDim` 连续拷贝），每索引一次 73.7KB/147KB 连续搬运、标量 load 次数 ÷64~128——对本场景是数量级差异（§5-O3）。

**B5（★★）小 M/短 K 的 cube 效率 + bmm2 fixpipe 瓶颈**
- bmm1：M=64（单 token 的 G 头），N=n（1~128，均值 48–64.5）——早期 token N 极小但每 task 固定开销不变；bmm2：K=n（均值 48–64.5）短累加、N=512。
- bmm2 每 task L0B(V) 搬 1,024n B 只做 65,536n FLOP，且 L0C 输出 128KB fp32（16K 元素）对应 4.2 MFLOP → fixpipe 侧 31 FLOP/B，低于机器平衡点，fixpipe-bound。
- MMA 以 16×16×16 分形计：bmm1 4×(n/16)×36 条、bmm2 4×32×(n/16) 条，每条之间的 MTE1 装载/等待占比高（L0A/B 双 buffer 只能部分掩盖）。

**B6（★，场景陷阱）needInit 全量 memset 与 PA+LSE 互斥**
- 变长 batch（任一 batch act_q<S1，BSND）触发 `needInit` → **全体 AIV memset 整个 attention_out（B=32 时 268MB）+ softmax 缓冲 + SyncAll**（VEC:618-682、KERNEL:197-199）。本场景全 128 不触发，但混合变长时会淹没一切优化。
- `return_softmax_lse=true && layout_kv=PA_BSND` 被 tiling 拒绝（TILING:1992-1996）——目标场景若需 LSE 只能退 BSND。

**B7（★）静态连续分核的负载不均**
- 任务按 (b,s1) 顺序连续切（KERNEL:230-251），而工作量 n(p)=min(p+1,K) 随 p 增长：24 核分 4096 任务时核间 Σn 波动约 ±15%（K=128，B=32）。无 work-stealing。次要。

**B8（信息性）核数与 FD**
- B=32/64 任务数（4096/8192）≥ AIC 数 → 核打满（`usedCoreNum_=aicNum_`，TILING:391；kernel 侧不再下调，KERNEL:203-214）。**"核数打不满"不是本场景问题**。
- splitKV/FD 永不触发（`splitKVFlag_` 无置 true 路径，TILING:363/434-436；tiling key FLASH_DECODE 恒 0，TILING:308-310）。S2=128 本无 FD 需求；但若上游把 S2 拉长（如 prefill 累积上下文），FD 缺失会成为新瓶颈（预留风险）。

**B9（次要）softmax 向量化**
- 每 task 每 AIV 处理 [32,n] fp32（≤16KB），VF 按 N 三档特化（VEC:501-537）已尽量避免分支；exp/规约/量化本身 ~0.3–1µs/task，非主要矛盾。掩码（vselr）与无效列处理的开销无法在本仓库评估（VF 源码缺失，见 §6.4）。

---

## 5. 优化机会清单

> 按 预期收益×可行性 排序。均未实施，仅给方案。

**O1. gather 索引批量化：每 task 一次 DataCopy 读整行索引 + blockTable 行缓存（收益 ★★★，风险低）**
- 位置：VEC:176-225（GetRealCmpS2Idx/GetkeyOffset）、VEC:404-450（ProcessSparseKv）。
- 思路：任务开始时把 `sparse_indices[b,p,0,n)`（≤512B）与该 batch 的 `block_table` 行一次性 DataCopy 进 UB，内层循环改读 UB 标量（UB 访存 ~ns 级），消除 2n 次串行 GM 标量 load。可进一步用向量指令（gather/按 16 展开）把 idx→(blockIdx,offset) 计算向量化。
- 预期：Vec0 标量链 0.55–1.1ms → 数十 µs 量级；若 B2 成立，端到端 **~1.5–3×**。
- 风险：低；-1 哨兵与计数截断语义保持不变即可。
- 验证：现有 pytest 精度回归（含 -1 padding 用例）+ msprof 对比 AIV scalar pipe 占用。

**O2. 小 S2 场景的"整批 KV 驻留 + 列掩码"路径（收益 ★★★，风险中高）**
- 位置：KERNEL:526-611（任务循环）、KVCACHE:94-120（qSNumInOneBlock/任务定义）、CUBE:268-295（Q 装载）。
- 思路：当 S2（或每 batch 有效 KV 窗口）≤128（更一般：batch KV ≤ L1 预算）时，每 (b,n2) **只 gather 一次全窗口 KV**（147.5KB，ND2NZ 进 L1 一份），该 batch 的所有 S1 任务直接复用；每 token 的 topk 子集用 softmax 列掩码（vselr 机制已存在，VEC:503-537）表达——即退化为"带每 token 动态列掩码的 dense 小 FA"。Q 可按 4~8 token 合批（M=256/512）提升 cube 效率，fixpipe/Q 装载按批摊薄。
- 量化：KV 流量 912MB → 4.7MB（×B）；FLOP 至多 ×2（128 列 vs 均值 64.5）而 cube 当前利用率 <20%，净赚；AIC MTE2 从 606MB → 302MB（只剩 Q）。
- 预期：本 shape 端到端 **3–6×**（趋向 cube/带宽 roofline ~150–250µs@B=32）。
- 风险：中高——列掩码必须精确复刻「前 min(p+1,K) 个索引 + 因果阈值 + -1 截断」语义（GOLDEN:624-646 是对齐基准）；softmax VF 需要每 token 有效列数/位置比较（vselrIndexesBuf 的填充协议需先逆向清楚，见 §6.4）；跨核同步结构大改。
- 验证：golden 全量对比（随机索引+变 K+变 S1/S2）+ 目标 shape 性能基线。

**O3. A5 开放 block 级 sparseBlockSize（收益 ★★★（对 S2≫K 的场景），风险中）**
- 位置：TILING:793-797（放开 DAV_3510 限制）、KERNEL:417（`sparseBlockSize=1` 硬编码）、VEC:275-278（>1 时退化双 SingleKv，需改成 blockLen=sparseBlockSize×dims 的整块拷贝，参照 arch22:887-897）、KVCACHE:241-265（s2LineEnd 逻辑需按块数计）。
- 思路：让 lightning_indexer 输出 block 级索引（topk 后聚合/上取整到 64/128 块）。block=128 且 S2=128 时等价于 O2 的整窗驻留；S2≫K 时每索引一条 73.7~147KB 连续搬运，标量 load 与指令数 ÷64~128，HBM 访存连续。
- 预期：S2=128 场景与 O2 合流；更长期对长上下文 sparse 场景是主路径。
- 风险：块边界因果截断（arch22:884-885 的 validS2Count 裁剪需移植）；LI 侧索引转换精度（块化会引入额外计算量，需端到端评估）；tiling/kernel 双侧改动。
- 验证：A2/A3 已有 block 路径可作行为对照；golden 块化索引用例。

**O4. 去掉 KV 的 GM workspace 中转（收益 ★★，风险中）**
- 位置：VEC:342-348（UB→GM）、CUBE:222-223（GM→L1 ND2NZ）。
- 思路：AIV 直接把 gather 结果按 NZ 分形写入 L1 槽（P 的 strided 写 VEC:474-480 证明 AIV→L1 分形写可行），省掉 MTE3 出 GM 与 AIC MTE2 回读两跳。KV 触碰次数 4→2。
- 预期：AIC MTE2 的 KV 部分（~126µs/核@B=32,K=128）与 AIV MTE3（63µs）转为 L1 直写；端到端 ~10–20%。
- 风险：AIV 写 NZ 的 32B 粒度碎写模式效率、L1 槽同步从 GM 槽改为 L1 槽的 hazard 管理。
- 验证：L1 布局 dump + 精度回归 + MTE 队列 profiling。

**O5. Q 装载合批（依赖 O2/O3）（收益 ★★，风险中）**
- 位置：CUBE:282-288（每 task Q ND2NZ）、KERNEL:145（s1BaseSize=64）、CUBE:160-161（l1Q 单 buffer 73.7KB）。
- 思路：O2/O3 使多 token 共享同一 KV 窗口后，把 qSNumInOneBlock 提到 4~8（M=256~512），Q 每批一次装载，bmm1 M 维扩大 4–8×，fixpipe/Q 摊薄同倍数。注意 L1 预算：KV 3 槽 442KB + Q 147KB(M=128) 已近 512KB 上限，需减为 2 槽或 Q 走 GM。
- 预期：cube 效率显著提升（M=64→256 的 MMA/MTE1 比改善 ~4×）。

**O6. Vec2 免 UB→UB 中转 + 融合（收益 ★，风险低）**
- 位置：VEC:558-559（64KB UB→UB DataCopy）、VEC:541-584。
- 思路：单 s2 块场景（本场景全部任务）直接在 bmm2ResBuf 上原地 LastDiv+Cast，去掉中转拷贝；16.8MB×B/48AIV ≈ 11.2MB/AIV 的 UB 带宽（~110µs@B=32）白省。
- 风险：低（bmm2ResBuf 为 CROSS_CORE_SYNC_BOTH，Wait 后只读安全）。

**O7. needInit memset 精确化（收益 ★（变长 batch 时 ★★★），风险低）**
- 位置：VEC:618-667（InitOutputSingleCore 全量 memset）。
- 思路：只 memset 各 batch 的 [actS1, S1) 尾部区域与 softmax 尾部，而非全张量；变长 batch 场景 268MB → 0。
- 验证：变长 golden 用例（paramset "bsnd_multi_batch" 已有该形态）。

**O8. 索引上游排序 / 连续 run 合并（收益 ★~★★，风险低）**
- 位置：VEC:260-302（CopyInKvSparse 目前固定成对合并）；上游 LI（其 golden 按分数降序排，lightning_indexer_v2 测试golden:1472）。
- 思路：索引按位置升序后，相邻 run 可合并为单条大 blockCount 的 DataCopyPad（dstStride 已支持 576 行距），MTE 指令数与 HBM 突发都改善；LI 侧加一个 argsort（S1×K×logK，很小）或 kernel 侧做 run 检测。
- 风险：低（语义不变）；需评估 LI 侧排序成本。

**O9. rope 与 nope 融合 cache（长期，生态改动）**
- 位置：上游 KV cache 布局（测试侧 GOLDEN:868-893 分列两个 cache）。
- 思路：PA cache 融合为 [block, token, 576] 后每 token 一条 1152B 拷贝（指令减半、局部性更好）。涉及 cache 生产者契约，风险高。

**不建议的方向**：FD/splitKV（S2=128 无收益）；调大 `sInnerSize_`（A5 kernel 不读该 tiling 字段，改了无效——见 §2.1）；减少 usedCoreNum（核已打满）。

---

## 6. 正确性约束（优化必须保持的语义）

1. **数值链路**（GOLDEN:777-803 是对齐基准）：
   - bmm1/bmm2 均以 fp32 累加（模板 T=float，ENTRY:88-96；MMParam/CUBE:230-239）；scale 在 softmax 内施加（`softmaxScale`，VEC:504 等）。
   - softmax 为 online/max-subtract 形式；**P（exp 结果）先量化为 Q_T（bf16/fp16）再进 bmm2**（VEC:471-480 的 cast；GOLDEN:786-799 `quantize_exp` 对齐）；最终 `LastDivNew` 以 fp32 sum 归一后再 cast 到输出 dtype（VEC:541-606）。任何融合/重排不得改变该量化次序，否则双千分位对比会退。
   - sinks（A5 独有）：首 s2 块以 sinks 初始化 max/sum（VEC:499-524、486-488 的 SFAUpdateExpSumAndExpMax）。
2. **lse 语义**：`softmax_max`=scale 后的行 max、`softmax_sum`=Σexp(x−max)，fp32，BSND [B,N2,S1,G] / TND [N2,T,G]（GOLDEN:724-734；VEC:712-742），只在最后一个 s2 loop 写；`return_softmax_lse=false` 时输出为占位。**PA_BSND + lse=true 当前直接报错（TILING:1992-1996），优化不得悄悄放开语义**。
3. **mask / 索引语义**（sparse_mode=3）：
   - 每 query token p 只处理 sparse_indices 行的**前 min(clip(p+Δs+1,0,S2), K) 个 entry**（KVCACHE:251-260）；Δs=S2−S1（nextTokens 由 actual lens 推导，KVCACHE:79-84；pre/next 属性仅允许默认 INT64_MAX，TILING:818-836）。
   - 合同：**有效索引必须在行前部、-1 补齐在后**（aclnnSparseFlashAttentionV2.md:162）；-1（加 s2Start 后仍为 −1，VEC:186-199）终止 gather；单侧 -1 时另一侧照常拷贝并压实（VEC:260-302 返回值只计有效行）。
   - golden 对「索引值 ≥ threshold」的容错（GOLDEN:641-646 skip/截断）在 A5 kernel 侧**没有**对应的逐位置检查——正确性依赖 LI 只产因果窗口内索引。优化 O2 的列掩码必须显式补上这个逐位置判断，否则比现状更脆。
   - sparse_mode=0：threshold=actSeqKV，全索引有效（GOLDEN:624-625）。
4. **无法在本仓库审计的点**（优化前必须逆向/实测确认）：A5 softmax VF（`vf_mul_sel_softmaxflashv2_cast_nz_sfa.h`，VEC:24 引用，源码不在 checkout）中 `vselrIndexesBuf` 的填充者与掩码规则——特别是「gather 提前终止时 GM workspace 尾部行是 3 槽前的陈旧数据，bmm1 的 N 仍取 s2RealSize（KERNEL:694-701、CUBE:230-236）」时，无效列如何被置 −inf（`negativeFloatScalar` 的用途）。建议用受控 -1 padding 用例（K 大、p 小）在真机上做差分测试确认现状行为，再做 O2/O4。
5. **布局/形状硬约束**：D=512、Dr=64（TILING:1463-1485）；N2=1（TILING:1429）；A5 gSize∈[1,128]（TILING:1444-1451）；blockSize 为 16 倍数 ≤1024 且整除 sparseBlockSize（TILING:1538-1558 一带）；attention_mode 仅 2 且 rope 必传（TILING:851-878，README:213）；TND 需 actual_seq_lengths（TILING:1343-1347）；**A5 sparseBlockSize 仅 1**（TILING:793）。
6. **value 张量被忽略**：A5 kernel 无任何 valueGm（VEC:688-696 grep 证据，§1.3）——语义等价 V=key[...,:512]（GOLDEN:717）。优化时保持该约定（或借此机会显式校验 value 与 key 的一致性以防误用）。
7. **无效/边界输入**：actS1=0 的 batch（golden 直接置零）；`needInit` 路径必须保留（变长 batch 的输出尾部清零语义）；sparse_indices dtype int32、block_table int32（TILING:60-61）。

---

## 7. 建议的验证顺序（perf 侧）

1. 基线：目标 4 组 shape（B∈{32,64}×K∈{64,128}，PA_BSND、bf16、sparse_mode=3、act=128）用 `test_run.sh single`（ROOT\tests\pytest\test_run.sh:26-34）+ msprof 采集，重点看：AIV Scalar/MTE2/MTE3 队列占比、AIC MTE1/MTE2/fixpipe/MMA 占比、每 task 平均周期（验证 §3.3 的 B2/B3 权重）。
2. 先做 O1（低风险）复测 → 若 B2 命中，直接拿到 1.5–3×。
3. 并行预研 O2 的列掩码协议（§6.4 的逆向 + 差分用例），再决定 O2/O3 路线。
4. O6/O7 随任意一次内核改动顺手带上。
