# Mock HCCL Server Framework

## 概览

本框架提供了一个**通用的HCCL Server端模拟实现**，用于在单卡环境下替代真实的CCU Server / HCCL通信后端。它与具体的MC2算子无关—任何通过workspace消息协议与HCCL Server交互的算子（如AllGatherMatmul、MatmulReduceScatter等）都可以使用本框架进行单卡测试。

### 背景：hcclClient / Server架构

在真实的多卡环境中，MC2类算子的通信流程如下：

```
Device (kernel)                      CCU (HCCL Server)
  │                                    │
  │  1. 填sendMsgs[slot]              │
  │  2. 写commitTurnCnt[slot]         │
  │  ──────── workspace ──────────→    │
  │                                    │  3. 读commitTurnCnt,发现有效
  │                                    │  4. 读sendMsgs,执行集合通信
  │                                    │  5. 写finishedTurnCnt[slot]
  │  ←──────── workspace ─────────     │
  │  6. 轮询finishedTurnCnt,继续     │
  │                                    │
```

kernel通过workspace中的消息协议向Server提交通信请求，Server执行实际的跨rank数据搬移，完成后通过workspace回写完成标志。这个 **workspace消息协议是算子无关的**—不同的MC2算子（AllGather、ReduceScatter等）使用相同的协议格式，只是commType字段不同。

**本框架就是在host侧用一个轮询线程模拟这个Server的行为。**

```
目录结构:
tests/mc2_mock_test_frame/              # 本目录—通用mock框架
├── mock_framework.h                    # C++ 核心(MockContextBuilder + MultiRankMockContext + MockHcclServer)
├── mock_framework_test.cpp             # C++ 单元测试(协议交互验证)
├── mock_framework.cpp                   # pybind11 torch extension (Python binding)
└── README.md                           # 本文档

tests/torch_extension_tests/mc2/all_gather_matmul_v3/  # V3算子测试
├── conftest.py                             # mock测试公共工具
├── test_all_gather_matmul_v3_mock.py       # 单rank mock测试
├── test_all_gather_matmul_v3_mock_multirank.py  # 多rank多流mock测试
└── test_all_gather_matmul_v3.py            # 真实多卡测试(torchrun)
```

---

## 1. Workspace消息协议

MockHcclServer模拟的核心是workspace消息协议。这是HCCL定义的kernel ↔ Server通信接口，与具体算子无关。

### 1.1 Workspace布局

workspace需要512B对齐，包含4个区域：

```
Workspace (512B对齐):
┌──────────────────────────────────────┐
│ sendMsgs[64]      (each 112B)        │ +0x0000   kernel → server的通信请求
├──────────────────────────────────────┤
│ recvMsgs[64]      (each 112B)        │ +0x1C00   server → kernel的响应(预留)
├──────────────────────────────────────┤
│ commitTurnCnt[64] (each 64B)         │ +0x5800   kernel提交标志
├──────────────────────────────────────┤
│ finishedTurnCnt[64](each 64B)        │ +0x6800   server完成标志
└──────────────────────────────────────┘
```

### 1.2 消息结构

**HcclMsg (112 bytes)**— kernel填写的通信请求：

| 偏移 | 字段 | 说明 |
|------|------|------|
| +0x00 | commType (u32) | 通信类型: AllGather=6, ReduceScatter=7 |
| +0x04 | opType (u32) | reduce操作类型 |
| +0x08 | sendBuffer (u64) | device源地址 |
| +0x10 | recvBuffer (u64) | device目标地址 |
| +0x18 | dataCnt (u64) | 元素数 |
| +0x20 | strideCount (u64) | recvBuf中rank间步长 |
| +0x28 | msgValid (u32) | HCCL_MSG_VALID_MASK (0x5CDF123A) |
| +0x2C | hcclDataType (u32) | 数据类型: FP32=0, FP16=1, BF16=5, ... |
| +0x30 | rest[64] | 剩余字段 |

**TurnCnt (64 bytes)**—提交/完成计数器：

| 偏移 | 字段 | 说明 |
|------|------|------|
| +0x00 | valid (u64) | COMMIT_VALID_MASK (987654321)表示有效 |
| +0x08 | cnt (u64) | 提交/完成计数 |
| +0x10 | reserved[6] | 填充至cache line对齐 |

### 1.3 协议流程

**常规通信（AllGather/ReduceScatter）：**

```
kernel:
  1. 填写sendMsgs[slot] (commType, sendBuf, recvBuf, dataCnt, ...)
  2. 写commitTurnCnt[slot].valid = COMMIT_VALID_MASK

server (MockHcclServer):
  3. 轮询commitTurnCnt,发现valid → 读sendMsgs[slot]
  4. 根据commType执行操作(单卡: D2D memcpy sendBuf → recvBuf)
  5. 写finishedTurnCnt[slot].cnt = commitCnt
  6. 清除commitTurnCnt[slot].valid

kernel:
  7. 轮询finishedTurnCnt[slot].cnt >= 期望值 → 继续
```

**Finalize（通信结束握手）：**

```
kernel Finalize():
  1. 提交finalize commit (最后一条消息)

server:
  2. 检测到finalize commit
  3. 写finishedTurnCnt[slot].cnt = FINALIZE_FINISH_CNT (1234567899999999999)

kernel:
  4. 检测到FINALIZE_FINISH_CNT → 通信流程结束
```

---

## 2. 框架组件

### 2.1 MockHcclServer —通用HCCL Server模拟

**核心职责：** 在host端启动一个轮询线程，监听workspace中的通信请求，并使用传入的各rank输入tensor模拟真实的集合通信语义。

构造时需传入每个rank的输入tensor（device指针），server在处理通信请求时读取这些tensor，模拟从远端rank获取数据。

**支持的通信类型：**

| commType | 操作 | Mock行为 |
|----------|------|-----------|
| 6 (AllGather) | 各rank chunk拼接 | rankInputs[r] → recvBuf[r * stride]，本卡用sendBuf |
| 7 (ReduceScatter) | reduce + scatter | D2H读各rank → host端element-wise reduce → 取chunk[localRank] H2D写回 |
| 2 (AllReduce) | reduce全量 | D2H读各rank → host端element-wise reduce → 全量H2D写回 |
| 12 (AlltoAll) | 块重排 | 各rank的block[localRank] → recvBuf[r] |

**Reduce操作：** 支持SUM(0) / PROD(1) / MAX(2) / MIN(3)，host侧通过FP16/BF16 ↔ float转换后计算。

### 2.2 MockContextBuilder —单rank HCCL Context构造

在device侧构造kernel需要的`HcclA2CombineOpParam`结构（即commContext）。所有rank的`windowsIn[i]`指向同一块device内存，使得910B AIV模式的flag同步机制自洽。

### 2.3 MultiRankMockContext —多rank Context构造

分配N个独立window（184MB每个）+ N个context，context中windowsIn[] 互相交叉引用。配合多stream并发kernel实现真实多rank通信。

### 2.4 Python Binding (mock_framework.cpp)

通过pybind11暴露给Python：

```python
import mock_hccl_ext

# 单rank测试
ctx = mock_hccl_ext.MockContext(rank_num=2, rank_id=0, device_id=0)
ctx.build()
server = mock_hccl_ext.MockServer(workspace_ptr=ctx.workspace_ptr(),
                                   rank_inputs=[t0, t1], local_rank_id=0)
server.start()
# ... 调用算子 ...
server.wait_for_finalize(slot=0, timeout_ms=10000)
server.stop()

# 多rank测试
mctx = mock_hccl_ext.MultiRankContext(rank_num=2, device_id=0)
mctx.build()
ctx_r0 = mctx.context_tensor(0)
ctx_r1 = mctx.context_tensor(1)
# Launch on separate streams...
```

---

## 3. 单元测试

`mock_framework_test.cpp`验证mock server的协议正确性（与具体算子无关）：

| Test | 验证点 |
|------|--------|
| TestContextBuilder | context H2D/D2H，字段readback正确 |
| TestServerProtocol | 手写commitTurnCnt → server响应finishedTurnCnt |
| TestServerDataCopy | AllGather msg → server执行D2D copy → recvBuf内容正确 |
| TestFinalizeProtocol | regular msg + finalize commit → FINALIZE_FINISH_CNT响应 |

编译运行：

```bash
ASCEND_HOME=~/Ascend/ascend-toolkit/latest
g++ -std=c++17 -O2 -I${ASCEND_HOME}/include \
  mock_framework_test.cpp \
  -L${ASCEND_HOME}/lib64 -lascendcl -lpthread \
  -Wl,-rpath,${ASCEND_HOME}/lib64 \
  -o /tmp/mock_framework_test
/tmp/mock_framework_test
```

---

## 4. 如何为新算子添加测试

本框架与算子无关，为新的MC2算子添加测试只需：

1. **构造mock context**— `MockContextBuilder.Build(rankNum, rankId)`
2. **启动mock server**— `MockHcclServer(workspace).Start()`
3. **调用算子**—将mock context作为commContext输入传给算子
4. **等待完成**— `WaitForFinalize()`等待kernel的Finalize握手
5. **验证结果**—检查算子的计算输出是否正确

---

## 5. 已知限制

| 限制 | 说明 |
|------|------|
| 单卡only | 多卡需要真实HCCL或多device mock |
| D2D memcpy代替集合通信 | 单卡无跨rank数据，AllGather/ReduceScatter退化为memcpy |
| Mock context无RDMA | IbVerbsData / aiRMAInfo为0，不支持跨节点场景 |
| Server轮询延迟 | host D2H/H2D轮询有 ~50us级延迟，不反映真实通信时序 |
| Flag自解不验证时序 | 单卡mock的flag写后立即可读，无法验证多卡竞争条件 |
