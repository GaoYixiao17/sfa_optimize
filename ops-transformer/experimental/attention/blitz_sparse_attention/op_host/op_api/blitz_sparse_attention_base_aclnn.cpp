/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include "blitz_sparse_attention.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(BlitzSparseAttention);

const aclTensor *BlitzSparseAttention(
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
    const char *inputLayout,
    int64_t numKeyValueHeads,
    int64_t sparseMode,
    int64_t innerPrecise,
    bool softmaxLseFlag,
    const aclIntArray *blockShape,
    const aclTensor *attentionOut,
    const aclTensor *softmaxLse,
    aclOpExecutor *executor,
    const aclTensor **softmaxLseOutRet)
{
    L0_DFX(BlitzSparseAttention, query, key, value, pseShift, attenMask, sabi, actualSeqLengths, actualSeqLengthsKv,
           deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2,
           numHeads, scaleValue, preTokens, nextTokens, inputLayout, numKeyValueHeads,
           sparseMode, innerPrecise);

    const aclTensor *actualSeqLengthsTensor = nullptr;
    if (executor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "BlitzSparseAttention: executor is nullptr.");
        return nullptr;
    }

    if (actualSeqLengths) {
        actualSeqLengthsTensor = executor->ConvertToTensor(actualSeqLengths, DataType::DT_INT64);
        const_cast<aclTensor *>(actualSeqLengthsTensor)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualSeqLengthsTensor)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualSeqLengthsTensor)->SetOriginalFormat(Format::FORMAT_ND);
    }

    const aclTensor *actualSeqLengthsKvTensor = nullptr;
    if (actualSeqLengthsKv) {
        actualSeqLengthsKvTensor = executor->ConvertToTensor(actualSeqLengthsKv, DataType::DT_INT64);
        const_cast<aclTensor *>(actualSeqLengthsKvTensor)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualSeqLengthsKvTensor)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualSeqLengthsKvTensor)->SetOriginalFormat(Format::FORMAT_ND);
    }

    auto attentionOutOut = executor->AllocTensor(attentionOut->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    auto softmaxLseOut   = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);

    auto ret = INFER_SHAPE(BlitzSparseAttention,
        OP_INPUT(query, key, value, pseShift, attenMask, sabi, actualSeqLengthsTensor, actualSeqLengthsKvTensor,
                 deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2),
        OP_OUTPUT(attentionOutOut, softmaxLseOut),
        OP_ATTR(numHeads, static_cast<float>(scaleValue), preTokens, nextTokens,
                inputLayout, numKeyValueHeads, sparseMode, innerPrecise, softmaxLseFlag, blockShape));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "BlitzSparseAttention InferShape failed.");
        return nullptr;
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(BlitzSparseAttention,
        OP_INPUT(query, key, value, pseShift, attenMask, sabi, actualSeqLengthsTensor, actualSeqLengthsKvTensor,
                 deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2),
        OP_OUTPUT(attentionOutOut, softmaxLseOut),
        OP_ATTR(numHeads, static_cast<float>(scaleValue), preTokens, nextTokens,
                inputLayout, numKeyValueHeads, sparseMode, innerPrecise, softmaxLseFlag, blockShape));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "BlitzSparseAttention LaunchAicore failed.");
        return nullptr;
    }

    if (softmaxLseOutRet != nullptr) {
        *softmaxLseOutRet = softmaxLseOut;
    }
    return attentionOutOut;
}
}