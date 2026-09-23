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
//#include "aclnnop/aclnn_sparse_flash_attention_antiquant_metadata.h"
#include "../../sparse_flash_attention_antiquant/op_kernel/sparse_flash_attention_antiquant_metadata.h"
#include "../op_host/aclnn_sparse_flash_attention_antiquant_metadata.h"

static const uint32_t batchSize = 4;
static const uint32_t querySeqSize = 3;
static const uint32_t queryHeadNum = 10;
static const uint32_t kvSeqSize = 10240;
static const uint32_t kvHeadNum = 1;
static const uint32_t headDim = 128;
static const uint32_t topKSize = 128;
static const uint32_t sparseBlockSize = 16;
static std::string layoutQuery = "BSND";
static std::string layoutKV = "BSND";
static const uint32_t sparseMode = 0;
static const uint32_t attentionMode = 0;
static const uint32_t ropeHeadDim = 64;
static const uint32_t sparseSharedSize = 3;

static const std::vector<int32_t> actSeqLenQuery = {3, 6, 9, 12};
static const std::vector<int32_t> actSeqLenKV = {10240, 10240, 10240, 10240};
static const std::vector<int32_t> sparseSeqLenKV = {2560, 2560, 2560, 2560};
static const std::vector<int64_t> actSeqLenQShape = {batchSize};
static const std::vector<int64_t> actSeqLenKVShape = {batchSize};
static const std::vector<int64_t> sparseSeqLenKVShape = {batchSize};
static const std::vector<int64_t> actSeqLenQStride = {1};
static const std::vector<int64_t> actSeqLenKVStride = {1};
static const std::vector<int64_t> sparseSeqLenKVStride = {1};
static const std::vector<int64_t> metadataShape = {optiling::SFA_META_SIZE};
static const std::vector<int64_t> metadataStride = {1};

static const bool enableActLenQuery = true;
static const bool enableActLenKV = true;
static const bool enablesparseSeqLenKV = true;

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
  optiling::detail::SfaMetaData* metaDataPtr =
      (optiling::detail::SfaMetaData*)data;
  printf("mBaseSize: %d \n", metaDataPtr->mBaseSize);
  printf("s2BaseSize: %d \n", metaDataPtr->s2BaseSize);
  printf("gS1BaseSizeOfFd: %d \n", metaDataPtr->gS1BaseSizeOfFd);
  printf("usedCoreNum: %d \n", metaDataPtr->usedCoreNum);
  printf("numOfFdHead: %d \n", metaDataPtr->numOfFdHead);
  printf("usedVecNumOfFd: %d \n", metaDataPtr->usedVecNumOfFd);
  for (uint32_t i = 0; i < metaDataPtr->usedCoreNum; i++) {
    printf("bN2End[%d]: %d \n", i, metaDataPtr->bN2End[i]);
    printf("gS1End[%d]: %d \n", i, metaDataPtr->gS1End[i]);
    printf("s2End[%d]: %d \n", i, metaDataPtr->s2End[i]);
    printf("s2SplitStartIdxOfCore[%d]: %d \n", i,
           metaDataPtr->fdRes.s2SplitStartIdxOfCore[i]);
  }

  for (uint32_t i = 0; i < metaDataPtr->numOfFdHead; i++) {
    printf("bN2IdxOfFdHead[%d]: %d \n", i,
           metaDataPtr->fdRes.bN2IdxOfFdHead[i]);
    printf("gS1IdxOfFdHead[%d]: %d \n", i,
           metaDataPtr->fdRes.gS1IdxOfFdHead[i]);
    printf("s2SplitNumOfFdHead[%d]: %d \n", i,
           metaDataPtr->fdRes.s2SplitNumOfFdHead[i]);
    printf("gS1SplitNumOfFdHead[%d]: %d \n", i,
           metaDataPtr->fdRes.gS1SplitNumOfFdHead[i]);
    printf("gS1LastPartSizeOfFdHead[%d]: %d \n", i,
           metaDataPtr->fdRes.gS1LastPartSizeOfFdHead[i]);
  }

  for (uint32_t i = 0; i < metaDataPtr->usedVecNumOfFd; i++) {
    printf("gS1IdxEndOfFdHead[%d]: %d \n", i,
           metaDataPtr->fdRes.gS1IdxEndOfFdHead[i]);
    printf("gS1IdxEndOfFdHeadSplit[%d]: %d \n", i,
           metaDataPtr->fdRes.gS1IdxEndOfFdHeadSplit[i]);
  }
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  aclError ret = 0;
  aclTensor* qSeqLenTensor = nullptr;
  void* qSeqLenDevPtr = nullptr;
  aclTensor* kvSeqLenTensor = nullptr;
  void* kvSeqLenDevPtr = nullptr;
  aclTensor* spSeqLenTensor = nullptr;
  void* spSeqLenDevPtr = nullptr;
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

  if (enableActLenQuery) {
    std::tie(qSeqLenTensor, qSeqLenDevPtr) = CreateTensor(
        actSeqLenQuery.size() * sizeof(actSeqLenQuery[0]), actSeqLenQShape,
        actSeqLenQStride, aclDataType::ACL_INT32, &actSeqLenQuery[0]);
    if (qSeqLenTensor == nullptr) {
      return -1;
    }
  }

  if (enableActLenKV) {
    std::tie(kvSeqLenTensor, kvSeqLenDevPtr) = CreateTensor(
        actSeqLenKV.size() * sizeof(actSeqLenKV[0]), actSeqLenKVShape,
        actSeqLenKVStride, aclDataType::ACL_INT32, &actSeqLenKV[0]);
    if (kvSeqLenTensor == nullptr) {
      return -1;
    }
  }

  if (enablesparseSeqLenKV) {
    std::tie(spSeqLenTensor, spSeqLenDevPtr) = CreateTensor(
        sparseSeqLenKV.size() * sizeof(sparseSeqLenKV[0]), sparseSeqLenKVShape,
        sparseSeqLenKVStride, aclDataType::ACL_INT32, &sparseSeqLenKV[0]);
    if (spSeqLenTensor == nullptr) {
      return -1;
    }
  }

  std::tie(metadataTensor, metadataDevPtr) =
      CreateTensor(sizeof(int32_t) * optiling::SFA_META_SIZE, metadataShape,
                   metadataStride, aclDataType::ACL_INT32);
  if (metadataTensor == nullptr) {
    return -1;
  }

  ret = aclnnKvQuantSparseFlashAttentionMetadataGetWorkspaceSize(
      qSeqLenTensor, kvSeqLenTensor, spSeqLenTensor, batchSize, querySeqSize,
      queryHeadNum, kvSeqSize, kvHeadNum, headDim, topKSize, sparseBlockSize,
      &layoutQuery[0], &layoutKV[0], sparseMode,
      attentionMode, ropeHeadDim, sparseSharedSize, metadataTensor,
      &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    printf("aclnnKvQuantSparseFlashAttentionMetadataGetWorkspaceSize %d\n",
           ret);
    return -1;
  }

  ret = aclnnKvQuantSparseFlashAttentionMetadata(workspace, workspaceSize,
                                                 executor, stream);
  if (ret != ACL_SUCCESS) {
    printf("aclnnKvQuantSparseFlashAttentionMetadata %d\n", ret);
    return -1;
  }

  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    printf("aclrtSynchronizeStream %d\n", ret);
    return -1;
  }

  std::vector<int32_t> metdataHost(optiling::SFA_META_SIZE);
  ret = aclrtMemcpy(metdataHost.data(),
                    metdataHost.size() * sizeof(metdataHost[0]), metadataDevPtr,
                    optiling::SFA_META_SIZE * sizeof(int32_t),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != ACL_SUCCESS) {
    printf("aclrtMemcpy %d\n", ret);
    return -1;
  }

  DumpMeta(&metdataHost[0]);

  aclDestroyTensor(qSeqLenTensor);
  aclDestroyTensor(kvSeqLenTensor);
  aclDestroyTensor(spSeqLenTensor);
  aclDestroyTensor(metadataTensor);

  aclrtFree(qSeqLenDevPtr);
  aclrtFree(kvSeqLenDevPtr);
  aclrtFree(spSeqLenDevPtr);
  aclrtFree(metadataDevPtr);
  aclrtFree(workspace);

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
