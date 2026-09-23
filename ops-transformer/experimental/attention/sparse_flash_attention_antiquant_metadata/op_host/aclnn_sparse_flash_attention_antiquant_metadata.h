/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_SPARSE_FLASH_ATTENTION_ANTIQUANT_METADATA_AICPU_H
#define ACLNN_SPARSE_FLASH_ATTENTION_ANTIQUANT_METADATA_AICPU_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnKvQuantSparseFlashAttentionMetadataGetWorkspaceSize
 * parameters :
 * actualSeqLengthsQueryOptional : optional
 * actualSeqLengthsKvOptional : optional
 * sparseSeqLengthsKvOptional : optional
 * batchSize : required
 * querySeqSize : required
 * queryHeadNum : required
 * kvSeqSize : required
 * kvHeadNum : required
 * headDim : required
 * topkSize : required
 * scaleValue : required
 * sparseBlockSize : required
 * layoutQueryOptional : optional
 * layoutKvOptional : optional
 * sparseMode : optional
 * attentionMode : optional
 * ropeHeadDim : optional
 * sparseShardSize : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default"))) aclnnStatus
aclnnKvQuantSparseFlashAttentionMetadataGetWorkspaceSize(
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
    char* layoutQueryOptional,
    char* layoutKvOptional,
    int64_t sparseMode,
    int64_t attentionMode,
    int64_t ropeHeadDim,
    int64_t sparseShardSize,
    const aclTensor* metaData,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

/* funtion: aclnnKvQuantSparseFlashAttentionMetadata
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default"))) aclnnStatus
aclnnKvQuantSparseFlashAttentionMetadata(void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_SPARSE_FLASH_ATTENTION_ANTIQUANT_METADATA_AICPU_H
