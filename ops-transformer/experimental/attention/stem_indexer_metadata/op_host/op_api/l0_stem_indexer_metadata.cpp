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
 * \file l0_stem_indexer_metadata.cpp
 * \brief
 */

#include "l0_stem_indexer_metadata.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(StemIndexerMetadata);

const aclTensor* StemIndexerMetadata(const aclTensor* qSeqLens, const aclTensor* kvSeqLens,
                                     int64_t qHeads, int64_t kvHeads, bool causal, int64_t stemBlockSize,
                                     int64_t dimQkflat, int64_t windowSize,
                                     const char* socVersion, int64_t aicCoreNum, int64_t aivCoreNum,
                                     const aclTensor* metadata, aclOpExecutor* executor)
{
    L0_DFX(StemIndexerMetadata, qSeqLens, kvSeqLens, qHeads, kvHeads, causal, stemBlockSize, dimQkflat, windowSize,
        socVersion, aicCoreNum, aivCoreNum, metadata);

    static internal::AicpuTaskSpace space("StemIndexerMetadata");

    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        StemIndexerMetadata,
        OP_ATTR_NAMES({"q_heads", "kv_heads", "causal", "stem_block_size",
                       "dim_qkflat", "window_size", "soc_version", "aic_core_num", "aiv_core_num"}),
        OP_INPUT(qSeqLens, kvSeqLens),
        OP_OUTPUT(metadata),
        OP_ATTR(qHeads, kvHeads, causal, stemBlockSize, dimQkflat, windowSize, socVersion, aicCoreNum, aivCoreNum));

    OP_CHECK(ret == ACL_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "StemIndexerMetadata ADD_TO_LAUNCHER_LIST_AICPU failed."),
             return nullptr);

    return metadata;
}

} // namespace l0op
