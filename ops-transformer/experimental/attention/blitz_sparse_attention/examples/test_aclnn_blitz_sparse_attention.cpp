/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_blitz_sparse_attention.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_blitz_sparse_attention.h"
#include "securec.h"
#include<unistd.h>
 
using namespace std;

namespace {
 
#define CHECK_RET(cond) ((cond) ? true :(false))
 
#define LOG_PRINT(message, ...)     \
  do {                              \
    (void)printf(message, ##__VA_ARGS__); \
  } while (0)
 
int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}
 
int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  if (!CHECK_RET(ret == ACL_SUCCESS)) { 
    LOG_PRINT("aclInit failed. ERROR: %d\n", ret); 
    return ret;
  }
  ret = aclrtSetDevice(deviceId);
  if (!CHECK_RET(ret == ACL_SUCCESS)) { 
    LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); 
    return ret;
  }
  ret = aclrtCreateStream(stream);
  if (!CHECK_RET(ret == ACL_SUCCESS)) { 
    LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); 
    return ret;
  }
  return 0;
}
 
template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) { 
    LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); 
    return ret;
  }
  
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (!CHECK_RET(ret == ACL_SUCCESS)) { 
    LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); 
    return ret;
  }
 
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
 
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

struct TensorResources {
    void* queryDeviceAddr = nullptr;
    void* keyDeviceAddr = nullptr;
    void* valueDeviceAddr = nullptr;
    void* attenDeviceAddr = nullptr;
    void* sabiDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* lseDeviceAddr = nullptr;
    aclTensor* queryTensor = nullptr;
    aclTensor* keyTensor = nullptr;
    aclTensor* valueTensor = nullptr;
    aclTensor* attenTensor = nullptr;
    aclTensor* sabiTensor = nullptr;
    aclTensor* outTensor = nullptr;
    aclTensor* lseTensor = nullptr;
    aclIntArray* actualSeqLengths = nullptr;
    aclIntArray* actualSeqLengthsKv = nullptr;
};

int InitializeTensors(TensorResources& resources) {
    // Parameters matching the Python test suite (N=8, S=512, D=128, B=1).
    // Constraint: N*S/128 > coreNum (~20) so the tiling keeps sOuterFactor=128
    // (the only tile size the BSA kernel is compiled for).  With N=2 or D=16 the
    // tiling degrades to sOuterFactor=32 which has no matching compiled template
    // and causes ADD_TO_LAUNCHER_LIST_AICORE to fail -> 561103.
    static constexpr int64_t B = 1, N = 8, S = 512, D = 128;
    static constexpr int64_t Q_TILES  = S / 128;   // 4
    static constexpr int64_t KV_TILES = S / 128;   // 4

    std::vector<int64_t> queryShape = {B, N, S, D};
    std::vector<int64_t> keyShape   = {B, N, S, D};
    std::vector<int64_t> valueShape = {B, N, S, D};
    // sabi: [B, N, Q_tiles, KV_tiles_kept] – dense: keep all KV tiles
    std::vector<int64_t> sabiShape  = {B, N, Q_TILES, KV_TILES};
    std::vector<int64_t> outShape   = {B, N, S, D};

    int64_t queryShapeSize = GetShapeSize(queryShape);
    int64_t sabiShapeSize  = GetShapeSize(sabiShape);
    int64_t outShapeSize   = GetShapeSize(outShape);

    // fp16 data: 0x3C00 = 1.0 in IEEE 754 half-precision.
    // Using uint16_t avoids the float/fp16 size mismatch in CreateAclTensor.
    std::vector<uint16_t> queryHostData(queryShapeSize, 0x3C00u);
    std::vector<uint16_t> keyHostData  (queryShapeSize, 0x3C00u);
    std::vector<uint16_t> valueHostData(queryShapeSize, 0x3C00u);
    std::vector<uint16_t> outHostData  (outShapeSize, 0u);

    // sabi values: KV tile indices 0..KV_TILES-1, cycling per Q-tile
    std::vector<uint16_t> sabiHostData(sabiShapeSize);
    for (int64_t i = 0; i < sabiShapeSize; i++) {
        sabiHostData[i] = static_cast<uint16_t>(i % KV_TILES);
    }

    int ret = CreateAclTensor(queryHostData, queryShape, &resources.queryDeviceAddr,
                              aclDataType::ACL_FLOAT16, &resources.queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      LOG_PRINT("Create query tensor failed. ERROR: %d\n", ret);
      return ret;
    }

    ret = CreateAclTensor(keyHostData, keyShape, &resources.keyDeviceAddr,
                          aclDataType::ACL_FLOAT16, &resources.keyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      LOG_PRINT("Create key tensor failed. ERROR: %d\n", ret);
      return ret;
    }

    ret = CreateAclTensor(valueHostData, valueShape, &resources.valueDeviceAddr,
                          aclDataType::ACL_FLOAT16, &resources.valueTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      LOG_PRINT("Create value tensor failed. ERROR: %d\n", ret);
      return ret;
    }

    // No attention mask – sparseMode=0 doesn't require one
    resources.attenTensor = nullptr;

    ret = CreateAclTensor(sabiHostData, sabiShape, &resources.sabiDeviceAddr,
                          aclDataType::ACL_UINT16, &resources.sabiTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      LOG_PRINT("Create sabi tensor failed. ERROR: %d\n", ret);
      return ret;
    }

    ret = CreateAclTensor(outHostData, outShape, &resources.outDeviceAddr,
                          aclDataType::ACL_FLOAT16, &resources.outTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      LOG_PRINT("Create output tensor failed. ERROR: %d\n", ret);
      return ret;
    }

    // LSE output: [B, N, S] float32
    std::vector<int64_t> lseShape = {B, N, S};
    int64_t lseShapeSize = GetShapeSize(lseShape);
    std::vector<float> lseHostData(lseShapeSize, 0.0f);
    ret = CreateAclTensor(lseHostData, lseShape, &resources.lseDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.lseTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Create LSE tensor failed. ERROR: %d\n", ret);
        return ret;
    }

    LOG_PRINT("Tensor shapes:\n");
    LOG_PRINT("  query:  [%ld, %ld, %ld, %ld] (B, N, S, D) fp16\n", B, N, S, D);
    LOG_PRINT("  key:    [%ld, %ld, %ld, %ld] (B, N, S, D) fp16\n", B, N, S, D);
    LOG_PRINT("  value:  [%ld, %ld, %ld, %ld] (B, N, S, D) fp16\n", B, N, S, D);
    LOG_PRINT("  sabi:   [%ld, %ld, %ld, %ld] (B, N, Q_tiles, KV_tiles) uint16\n",
              B, N, Q_TILES, KV_TILES);
    LOG_PRINT("  out:    [%ld, %ld, %ld, %ld] (B, N, S, D) fp16\n", B, N, S, D);
    LOG_PRINT("  lse:    [%ld, %ld, %ld]       (B, N, S)    float32\n", B, N, S);

    std::vector<int64_t> actualSeqlenVector   = {S};
    resources.actualSeqLengths  = aclCreateIntArray(actualSeqlenVector.data(),
                                                     actualSeqlenVector.size());

    std::vector<int64_t> actualSeqlenKvVector = {S};
    resources.actualSeqLengthsKv = aclCreateIntArray(actualSeqlenKvVector.data(),
                                                      actualSeqlenKvVector.size());

    return ACL_SUCCESS;
}

int ExecuteBlitzSparseAttention(TensorResources& resources, aclrtStream stream, 
                               void** workspaceAddr, uint64_t* workspaceSize) {
    int64_t numHeads = 8;
    int64_t numKeyValueHeads = 8;
    float scaleValue = static_cast<float>(1.0 / sqrt(128.0));
    int64_t preTokens = 65535;     // Large value to include all previous tokens
    int64_t nextTokens = 65535;    // Large value to include all next tokens
    int64_t sparseMode = 0;        // Basic sparse mode
    int64_t innerPrecise = 1;      // Standard precision
    bool    softmaxLseFlag = true; // Request LSE output

    // block_shape attr (index 9): [BLOCK_SIZE_Q, BLOCK_SIZE_KV]. Supported pairs:
    // (128,128) and (128,512). Default in this example matches the sabi shape
    // computed above (Q_TILES = KV_TILES = S/128 → sabi at 128 granularity).
    std::vector<int64_t> blockShapeVec = {128, 128};
    aclIntArray* blockShape = aclCreateIntArray(blockShapeVec.data(), blockShapeVec.size());

    // Using BNSD layout as it's common for attention operations
    constexpr const char LAYER_OUT_STR[] = "BNSD";
    constexpr size_t LAYER_OUT_LEN = sizeof(LAYER_OUT_STR);
    char layerOut[LAYER_OUT_LEN];
    memcpy(layerOut, LAYER_OUT_STR, LAYER_OUT_LEN);

    aclOpExecutor* executor;

    // Function call with all parameters in correct order
    int ret = aclnnBlitzSparseAttentionGetWorkspaceSize(
        resources.queryTensor,           // const aclTensor   *query,
        resources.keyTensor,             // const aclTensor   *key,
        resources.valueTensor,           // const aclTensor   *value,
        nullptr,                         // const aclTensor   *pseShift,
        resources.attenTensor,           // const aclTensor   *attenMask,
        resources.sabiTensor,            // const aclTensor   *sabi,
        resources.actualSeqLengths,      // const aclIntArray *actualSeqLengths,
        resources.actualSeqLengthsKv,    // const aclIntArray *actualSeqLengthsKv,
        nullptr,                         // const aclTensor   *deqScale1,
        nullptr,                         // const aclTensor   *quantScale1,
        nullptr,                         // const aclTensor   *deqScale2,
        nullptr,                         // const aclTensor   *quantScale2,
        nullptr,                         // const aclTensor   *quantOffset2,
        numHeads,                        // int64_t            numHeads,
        scaleValue,                      // float              scaleValue,
        preTokens,                       // int64_t            preTokens,
        nextTokens,                      // int64_t            nextTokens,
        layerOut,                        // char              *inputLayout,
        numKeyValueHeads,                // int64_t            numKeyValueHeads,
        sparseMode,                      // int64_t            sparseMode,
        innerPrecise,                    // int64_t            innerPrecise,
        softmaxLseFlag,                  // bool               softmaxLseFlag,
        blockShape,                      // const aclIntArray *blockShape,
        resources.outTensor,             // const aclTensor   *attentionOut,
        resources.lseTensor,             // const aclTensor   *softmaxLse,
        workspaceSize,                   // uint64_t          *workspaceSize,
        &executor);                      // aclOpExecutor     **executor

    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnBlitzSparseAttentionGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnBlitzSparseAttention(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnBlitzSparseAttention failed. ERROR: %d\n", ret);
        aclDestroyIntArray(blockShape);
        return ret;
    }

    aclDestroyIntArray(blockShape);
    return ACL_SUCCESS;
}

int ProcessResults(TensorResources& resources, const std::vector<int64_t>& outShape) {
    auto size = GetShapeSize(outShape);
    // Output tensor is ACL_FLOAT16 (uint16_t, 2 bytes per element)
    std::vector<uint16_t> resultData(size, 0);

    int ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         resources.outDeviceAddr, size * sizeof(resultData[0]),
                         ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
    }

    LOG_PRINT("Output results (first 10 values as raw fp16 hex):\n");
    for (int64_t i = 0; i < std::min(int64_t(10), size); i++) {
        LOG_PRINT("output[%ld] = 0x%04X\n", i, resultData[i]);
    }

    // LSE output: [B, N, S] float32
    // outShape = {B, N, S, D}; lse shape = {B, N, S}
    int64_t lseSize = outShape[0] * outShape[1] * outShape[2];
    std::vector<float> lseData(lseSize, 0.0f);
    ret = aclrtMemcpy(lseData.data(), lseData.size() * sizeof(float),
                      resources.lseDeviceAddr, lseData.size() * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy LSE from device to host failed. ERROR: %d\n", ret);
        return ret;
    }

    // For all-ones input: score = D*(1/sqrt(D)) = sqrt(D), so
    // LSE = log(S * exp(sqrt(D))) = log(S) + sqrt(D)
    float expectedLse = std::log(static_cast<float>(outShape[2]))
                      + std::sqrt(static_cast<float>(outShape[3]));
    LOG_PRINT("LSE results (first 10 values, expect ~%.4f for all-ones input):\n", expectedLse);
    for (int64_t i = 0; i < std::min(int64_t(10), lseSize); i++) {
        LOG_PRINT("lse[%ld] = %f\n", i, lseData[i]);
    }

    return ACL_SUCCESS;
}

void CleanupResources(TensorResources& resources, void* workspaceAddr, 
                     aclrtStream stream, int32_t deviceId) {
    // Cleanup tensors
    if (resources.queryTensor) aclDestroyTensor(resources.queryTensor);
    if (resources.keyTensor) aclDestroyTensor(resources.keyTensor);
    if (resources.valueTensor) aclDestroyTensor(resources.valueTensor);
    if (resources.attenTensor) aclDestroyTensor(resources.attenTensor);
    if (resources.sabiTensor) aclDestroyTensor(resources.sabiTensor);
    if (resources.outTensor) aclDestroyTensor(resources.outTensor);
    if (resources.lseTensor) aclDestroyTensor(resources.lseTensor);

    // Cleanup arrays
    if (resources.actualSeqLengths) aclDestroyIntArray(resources.actualSeqLengths);
    if (resources.actualSeqLengthsKv) aclDestroyIntArray(resources.actualSeqLengthsKv);

    // Cleanup device memory
    if (resources.queryDeviceAddr) aclrtFree(resources.queryDeviceAddr);
    if (resources.keyDeviceAddr) aclrtFree(resources.keyDeviceAddr);
    if (resources.valueDeviceAddr) aclrtFree(resources.valueDeviceAddr);
    if (resources.attenDeviceAddr) aclrtFree(resources.attenDeviceAddr);
    if (resources.sabiDeviceAddr) aclrtFree(resources.sabiDeviceAddr);
    if (resources.outDeviceAddr) aclrtFree(resources.outDeviceAddr);
    if (resources.lseDeviceAddr) aclrtFree(resources.lseDeviceAddr);
    
    // Cleanup workspace
    if (workspaceAddr) aclrtFree(workspaceAddr);
    
    // Cleanup stream
    if (stream) aclrtDestroyStream(stream);
    
    // Finalize device
    aclrtResetDevice(deviceId);
    aclFinalize();
}

} // namespace
 
int main() {
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    TensorResources resources = {};
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<int64_t> outShape = {1, 8, 512, 128};
    int ret = ACL_SUCCESS;

    LOG_PRINT("Initializing ACL...\n");
    
    // 1. Initialize device and stream
    ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    LOG_PRINT("Initializing tensors...\n");
    
    // 2. Initialize tensors
    ret = InitializeTensors(resources);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Initialize tensors failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    LOG_PRINT("Executing BlitzSparseAttention...\n");
    
    // 3. Execute the operation
    ret = ExecuteBlitzSparseAttention(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Execute BlitzSparseAttention failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    LOG_PRINT("Synchronizing stream...\n");
    
    // 4. Synchronize stream
    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    LOG_PRINT("Processing results...\n");
    
    // 5. Process results
    ret = ProcessResults(resources, outShape);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Process results failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    LOG_PRINT("Cleaning up resources...\n");
    
    // 6. Cleanup resources
    CleanupResources(resources, workspaceAddr, stream, deviceId);
    
    LOG_PRINT("Test completed successfully!\n");
    return 0;
}
