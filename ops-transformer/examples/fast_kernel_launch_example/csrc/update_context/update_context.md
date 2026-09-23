# MC2 Context创建流程

## Context结构

```cpp
struct Mc2ContextStru {
    uint64_t epRankId;           // 当前rank ID
    uint64_t kfcContextAddr;     // KFC通信上下文地址
    uint64_t epHcclBuffer[1024]; // 各rank的HCCL Buffer地址数组
};
```

## 创建流程

```
update_context(group_ep, ep_world_size)
    │
    ▼
GetMc2Context(mc2ContextHost, epWorldSize, bufferSize, groupEpStr)
    │
    ├─► HcclKfcAllocOpArgs(&opArgs)           // 1. 分配通信配置对象
    │
    ├─► HcclKfcOpArgsSetCommEngine(opArgs, AIV) // 2. 设置通信引擎为AIV
    │
    ├─► CreateHcclContext(comm, opArgs, worldSize, groupName)
    │       │
    │       ├─► HcclKfcOpArgsSetAlgConfig(opArgs, algConfig)  // 设置算法配置
    │       ├─► HcclCommGetHandleWithName(groupName, &comm)   // 获取通信句柄
    │       ├─► HcclCreateOpResCtx(comm, opType, opArgs)      // 创建操作资源上下文
    │       ├─► HcclGetRankSize(comm, &worldSize)             // 获取world size
    │       └─► HcclGetRankId(comm, &rankId)                  // 获取rank ID
    │
    ├─► CreatMc2Context(comm, worldSize, bufferSize, mc2Context)
    │       │
    │       ├─► HcclGetRankId(comm, &rankId)   // 获取当前rank
    │       │
    │       └─► for (remoteRankId = 0; remoteRankId < worldSize; remoteRankId++)
    │               │
    │               ├─► rankId == remoteRankId:
    │               │   HcclGetHcclBuffer(comm, &addr, &size)     // 本地buffer
    │               │
    │               └─► rankId != remoteRankId:
    │                   HcclGetRemoteIpcHcclBuf(comm, remoteRankId, &addr, &size)  // 远程IPC buffer
    │               │
    │               └─► mc2Context.epHcclBuffer[remoteRankId] = addr  // 存入结构体
    │
    ├─► HcclKfcFreeOpArgs(opArgs)             // 3. 释放通信配置对象
    │
    ▼
at::Tensor output = copy mc2ContextHost to NPU
return (output, bufferSize)
```

### 两阶段函数说明

创建流程分为两个核心函数：

#### CreateHcclContext - 建立HCCL通信框架

**职责**: 创建HCCL操作资源上下文，建立通信框架

| API | 作用 |
|-----|------|
| HcclKfcOpArgsSetAlgConfig | 设置All-to-All算法配置 |
| HcclCommGetHandleWithName | 根据group名称获取通信句柄 |
| HcclCreateOpResCtx | 创建操作资源上下文 |
| HcclGetRankSize / HcclGetRankId | 获取world size和rank ID，验证参数 |

**产出**: `HcclComm commHandle` - 有效的通信句柄

#### CreatMc2Context - 填充具体通信地址

**职责**: 基于已建立的通信框架，填充Mc2ContextStru结构体

| API | 作用 |
|-----|------|
| HcclGetRankId | 获取当前rank ID → `mc2Context.epRankId` |
| HcclGetHcclBuffer | 获取本地buffer地址和大小 |
| HcclGetRemoteIpcHcclBuf | 获取远程rank的IPC buffer地址 |

**产出**: 完整的`Mc2ContextStru`，包含所有rank的buffer地址

#### 逻辑关系

- **顺序依赖**: `CreateHcclContext`先执行建立通信框架，产出`commHandle`；`CreatMc2Context`后执行，依赖`commHandle`查询buffer地址

## 缓存机制

```cpp
static bool isInit = false;
static std::string last_group_ep_str = "";

// group未变化时直接返回缓存结果，以适应重复调用
if (isInit && new_group_ep_str == last_group_ep_str) {
    return std::make_tuple(output, bufferSize);
}
```

## Context的作用

1. **预获取通信地址**: 初始化时一次性获取所有rank的HCCL buffer地址，避免每次kernel调用时重复HCCL API查询
2. **支持零拷贝**: Kernel通过`epHcclBuffer[rank_id]`直接访问目标rank的buffer地址
3. **通信资源缓存**: 同一group重复调用时直接返回已创建的context
