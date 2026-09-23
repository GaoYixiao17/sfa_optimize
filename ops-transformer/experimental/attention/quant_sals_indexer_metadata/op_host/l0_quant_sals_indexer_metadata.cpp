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
 * \file l0_quant_sals_indexer_metadata.cpp
 * \brief
 */

#include "l0_quant_sals_indexer_metadata.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(QuantSalsIndexerMetadata);

const aclTensor* QuantSalsIndexerMetadata(
    const aclTensor* actualSeqLengthsKvOptional,
    int64_t batchSize,
    int64_t keySeqSize,
    int64_t keyHeadNum,
    int64_t headDim,
    int64_t aicCoreNum,
    int64_t aivCoreNum,
    int64_t sparseBlockSize,
    double sparseRatio,
    int64_t fixedTailCount,
    char* layoutKeyOptional,
    const char* socVersionOptional,
    const aclTensor* metaData,
    aclOpExecutor* executor) {
  L0_DFX(QuantSalsIndexerMetadata, actualSeqLengthsKvOptional, batchSize, 
            keySeqSize, keyHeadNum, headDim, aicCoreNum, aivCoreNum,
            sparseBlockSize, sparseRatio, fixedTailCount, layoutKeyOptional, socVersionOptional, metaData);

  static internal::AicpuTaskSpace space("QuantSalsIndexerMetadata");

  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
      QuantSalsIndexerMetadata,
      OP_ATTR_NAMES({"batch_size", "key_seq_size", "key_head_num", "head_dim", "aic_core_num",
                     "aiv_core_num", "sparse_block_size", "sparse_ratio", "fixed_tail_count",
                     "layout_key", "soc_version"}),
      OP_INPUT(actualSeqLengthsKvOptional), OP_OUTPUT(metaData),
      OP_ATTR(batchSize, keySeqSize, keyHeadNum, headDim, aicCoreNum, aivCoreNum,
            sparseBlockSize, sparseRatio, fixedTailCount, layoutKeyOptional, socVersionOptional));
  OP_CHECK(ret == ACL_SUCCESS,
           OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                   "QuantSalsIndexerMetadata"
                   " ADD_TO_LAUNCHER_LIST_AICPU failed."),
           return nullptr);
  return metaData;
}
}  // namespace l0op
