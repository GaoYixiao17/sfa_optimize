/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_blitz_sparse_attention.h"
#include "aclnn_blitz_sparse_attention_inner.h"
#include "opdev/op_dfx.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/pad.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/slice.h"
#include "aclnn_kernels/transpose.h"
#include "opdev/common_types.h"
#include "opdev/fast_vector.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
aclnnStatus aclnnBlitzSparseAttentionGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *pseShift,
    const aclTensor *attenMask,
    const aclTensor *sabi,
    const aclIntArray *actualSeqLengths,
    const aclIntArray *actualSeqLengthsKv,
    const aclTensor *deqScale1,
    const aclTensor *quantScale1,
    const aclTensor *deqScale2,
    const aclTensor *quantScale2,
    const aclTensor *quantOffset2,
    int64_t numHeads,
    double scaleValue,
    int64_t preTokens,
    int64_t nextTokens,
    char *inputLayout,
    int64_t numKeyValueHeads,
    int64_t sparseMode,
    int64_t innerPrecise,
    bool softmaxLseFlag,
    const aclIntArray *blockShape,
    const aclTensor *attentionOut,
    const aclTensor *softmaxLse,
    uint64_t *workspaceSize,
    aclOpExecutor **executor) {
        return InnerBlitzSparseAttentionGetWorkspaceSize(
            query, key, value, pseShift, attenMask, sabi,
            actualSeqLengths, actualSeqLengthsKv,
            deqScale1, quantScale1, deqScale2,
            quantScale2, quantOffset2,
            numHeads, scaleValue, preTokens, nextTokens,
            inputLayout, numKeyValueHeads, sparseMode,
            innerPrecise, softmaxLseFlag, blockShape,
            attentionOut, softmaxLse, workspaceSize, executor);
    }

aclnnStatus aclnnBlitzSparseAttention(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream) {
        return InnerBlitzSparseAttention(workspace, workspaceSize, executor, stream);
    }

}

#ifdef __cplusplus
}
#endif