/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_QUANT_SALS_INDEXER_METADATA_AICPU_H
#define ACLNN_QUANT_SALS_INDEXER_METADATA_AICPU_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnQuantSalsIndexerMetadataGetWorkspaceSize
 * parameters :
 * actualSeqLengthsKvOptional : optional
 * batchSize : required
 * kvSeqSize : required
 * kvHeadNum : required
 * fixedTailCount : required
 * sparseBlockSize : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default"))) aclnnStatus
aclnnQuantSalsIndexerMetadataGetWorkspaceSize(
    const aclTensor* actualSeqLengthsKvOptional,
    int64_t batchSize,
    int64_t keySeqSize,
    int64_t keyHeadNum,
    int64_t headDim,
    int64_t sparseBlockSize,
    double sparseRatio,
    int64_t fixedTailCount,
    char* layoutKeyOptional,
    const aclTensor* metaData,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

/* funtion: aclnnQuantSalsIndexerMetadata
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default"))) aclnnStatus
aclnnQuantSalsIndexerMetadata(void* workspace,
                              uint64_t workspaceSize,
                              aclOpExecutor* executor,
                              aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNN_QUANT_SALS_INDEXER_METADATA_AICPU_H
