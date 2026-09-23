/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdio.h>
#include <iostream>
#include <tuple>
#include <vector>
#include "acl/acl.h"
//#include "aclnnop/aclnn_quant_sals_indexer_metadata.h"
#include "../../quant_sals_indexer/op_kernel/quant_sals_indexer_metadata.h"
#include "../op_host/aclnn_quant_sals_indexer_metadata.h"

static const uint32_t batchSize = 3;
static const uint32_t kvSeqSize = 10240;
static const uint32_t kvHeadNum = 4;
static const uint32_t headDim = 128;
static const uint32_t sparseBlockSize = 16;
static const double sparseRatio = 0.25f;
static const uint32_t fixedTailCount = 16;
static std::string layoutKey = "BSND";

static const std::vector<int32_t> actSeqLenKV = {10240, 10240, 10240};
static const std::vector<int64_t> actSeqLenKVShape = {batchSize};
static const std::vector<int64_t> actSeqLenKVStride = {1};
static const std::vector<int64_t> metadataShape = {optiling::QSI_META_SIZE};
static const std::vector<int64_t> metadataStride = {1};

static const bool enableActLenKV = true;

std::tuple<aclTensor*, void*> CreateTensor(size_t size,  // in bytes
                                           std::vector<int64_t> shape,
                                           std::vector<int64_t> stride,
                                           aclDataType dType,
                                           const void* hostData = nullptr) {
  void* devicePtr = nullptr;
  auto ret = aclrtMalloc(&devicePtr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) {
    printf("aclrtMalloc %d\n", ret);
    return {nullptr, nullptr};
  }

  aclTensor* tensor = aclCreateTensor(&shape[0], shape.size(), dType,
                                      &stride[0], 0, aclFormat::ACL_FORMAT_ND,
                                      &shape[0], shape.size(), devicePtr);
  if (tensor == nullptr) {
    aclrtFree(devicePtr);
    return {nullptr, nullptr};
  }

  if (hostData != nullptr) {
    aclrtMemcpy(devicePtr, size, hostData, size, ACL_MEMCPY_HOST_TO_DEVICE);
  }
  return {tensor, devicePtr};
}

static void DumpMeta(void* data) {
  optiling::detail::QsiMetaData* metaDataPtr =
      (optiling::detail::QsiMetaData*)data;
  printf("usedCoreNum: %d \n", metaDataPtr->usedCoreNum);
  for (uint32_t i = 0; i < metaDataPtr->usedCoreNum; i++) {
    printf("bN2End[%d]: %d \n", i, metaDataPtr->bN2End[i]);
    printf("gS1End[%d]: %d \n", i, metaDataPtr->gS1End[i]);
    printf("s2End[%d]: %d \n", i, metaDataPtr->s2End[i]);
  }
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  aclError ret = 0;

  aclTensor* kvSeqLenTensor = nullptr;
  void* kvSeqLenDevPtr = nullptr;

  aclTensor* metadataTensor = nullptr;
  void* metadataDevPtr = nullptr;
  aclOpExecutor* executor = nullptr;
  uint64_t workspaceSize = 0;
  void* workspace = nullptr;

  ret = aclInit(nullptr);
  if (ret != ACL_SUCCESS) {
    printf("aclInit %d\n", ret);
    return -1;
  }

  ret = aclrtSetDevice(deviceId);
  if (ret != ACL_SUCCESS) {
    printf("aclrtSetDevice %d\n", ret);
    return -1;
  }

  ret = aclrtCreateStream(&stream);
  if (ret != ACL_SUCCESS) {
    printf("aclrtCreateStream %d\n", ret);
    return -1;
  }

  if (enableActLenKV) {
    std::tie(kvSeqLenTensor, kvSeqLenDevPtr) = CreateTensor(
        actSeqLenKV.size() * sizeof(actSeqLenKV[0]), actSeqLenKVShape,
        actSeqLenKVStride, aclDataType::ACL_INT32, &actSeqLenKV[0]);
    if (kvSeqLenTensor == nullptr) {
      return -1;
    }
  }

  std::tie(metadataTensor, metadataDevPtr) =
      CreateTensor(sizeof(int32_t) * optiling::QSI_META_SIZE, metadataShape,
                   metadataStride, aclDataType::ACL_INT32);
  if (metadataTensor == nullptr) {
    return -1;
  }

  ret = aclnnQuantSalsIndexerMetadataGetWorkspaceSize(
      kvSeqLenTensor, batchSize, kvSeqSize, kvHeadNum, headDim,
      sparseBlockSize, sparseRatio, fixedTailCount, &layoutKey[0], metadataTensor, &workspaceSize,
      &executor);
  if (ret != ACL_SUCCESS) {
    printf("aclnnQuantSalsIndexerMetadataGetWorkspaceSize %d\n", ret);
    return -1;
  }

  ret =
      aclnnQuantSalsIndexerMetadata(workspace, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    printf("aclnnQuantSalsIndexerMetadata %d\n", ret);
    return -1;
  }

  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    printf("aclrtSynchronizeStream %d\n", ret);
    return -1;
  }

  std::vector<int32_t> metdataHost(optiling::QSI_META_SIZE);
  ret = aclrtMemcpy(metdataHost.data(),
                    metdataHost.size() * sizeof(metdataHost[0]), metadataDevPtr,
                    optiling::QSI_META_SIZE * sizeof(int32_t),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != ACL_SUCCESS) {
    printf("aclrtMemcpy %d\n", ret);
    return -1;
  }

  DumpMeta(&metdataHost[0]);

  aclDestroyTensor(kvSeqLenTensor);
  aclDestroyTensor(metadataTensor);

  aclrtFree(kvSeqLenDevPtr);
  aclrtFree(metadataDevPtr);
  aclrtFree(workspace);

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
