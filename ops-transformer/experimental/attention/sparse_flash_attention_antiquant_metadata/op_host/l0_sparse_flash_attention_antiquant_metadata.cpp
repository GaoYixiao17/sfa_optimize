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
 * \file l0_sparse_flash_attention_antiquant_metadata.cpp
 * \brief
 */

#include "l0_sparse_flash_attention_antiquant_metadata.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(KvQuantSparseFlashAttentionMetadata);

const aclTensor* KvQuantSparseFlashAttentionMetadata(
    const aclTensor* actualSeqLengthsQueryOptional,
    const aclTensor* actualSeqLengthsKvOptional,
    const aclTensor* sparseSeqLengthsKvOptional,
    int64_t batchSize,
    int64_t querySeqSize,
    int64_t queryHeadNum,
    int64_t kvSeqSize,
    int64_t kvHeadNum,
    int64_t headDim,
    int64_t topkSize,
    int64_t sparseBlockSize,
    int64_t aicCoreNum,
    int64_t aivCoreNum,
    char* layoutQueryOptional,
    char* layoutKvOptional,
    int64_t sparseMode,
    int64_t attentionMode,
    int64_t ropeHeadDim,
    int64_t sparseShardSize,
    const char* socVersionOptional,
    const aclTensor* metaData,
    aclOpExecutor* executor) {
  L0_DFX(KvQuantSparseFlashAttentionMetadata, actualSeqLengthsQueryOptional,
         actualSeqLengthsKvOptional, sparseSeqLengthsKvOptional, batchSize,
         querySeqSize, queryHeadNum, kvSeqSize, kvHeadNum, headDim, topkSize,
         sparseBlockSize, aicCoreNum, aivCoreNum, layoutQueryOptional,
         layoutKvOptional, sparseMode, attentionMode, ropeHeadDim,
         sparseShardSize, socVersionOptional, metaData);

  static internal::AicpuTaskSpace space(
      "KvQuantSparseFlashAttentionMetadata");

  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
      KvQuantSparseFlashAttentionMetadata,
      OP_ATTR_NAMES({"batch_size", "query_seq_size", "query_head_num",
                     "kv_seq_size", "kv_head_num", "head_dim", "topk_size",
                     "sparse_block_size", "aic_core_num", "aiv_core_num",
                     "layout_query", "layout_kv", "sparse_mode",
                     "attention_mode", "rope_head_dim", "sparse_shared_size", "soc_version"}),
      OP_INPUT(actualSeqLengthsQueryOptional, actualSeqLengthsKvOptional,
               sparseSeqLengthsKvOptional),
      OP_OUTPUT(metaData),
      OP_ATTR(batchSize, querySeqSize, queryHeadNum, kvSeqSize, kvHeadNum,
              headDim, topkSize, sparseBlockSize, aicCoreNum, aivCoreNum,
              layoutQueryOptional, layoutKvOptional, sparseMode, attentionMode,
              ropeHeadDim, sparseShardSize, socVersionOptional));
  OP_CHECK(ret == ACL_SUCCESS,
           OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                   "KvQuantSparseFlashAttentionMetadata"
                   " ADD_TO_LAUNCHER_LIST_AICPU failed."),
           return nullptr);
  return metaData;
}

}  // namespace l0op
