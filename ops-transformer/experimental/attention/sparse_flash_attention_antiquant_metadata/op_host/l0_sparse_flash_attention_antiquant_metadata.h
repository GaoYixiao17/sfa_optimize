/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef L0_SPARSE_FLASH_ATTENTION_ANTIQUANT_METADATA_AICPU_H
#define L0_SPARSE_FLASH_ATTENTION_ANTIQUANT_METADATA_AICPU_H

#include "opdev/op_executor.h"

namespace l0op {
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
    int64_t sparseSharedSize,
    const char* socVersionOptional,
    const aclTensor* metaData,
    aclOpExecutor* executor);
} // namespace l0op

#endif
