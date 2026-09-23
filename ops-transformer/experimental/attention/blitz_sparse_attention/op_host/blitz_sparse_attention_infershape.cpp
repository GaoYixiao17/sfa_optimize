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
 * \file blitz_sparse_attention_infershape.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"

using namespace ge;
namespace ops {
static constexpr uint32_t PFA_LAYOUT_DIM0 = 0;
static constexpr uint32_t PFA_LAYOUT_DIM1 = 1;
static constexpr uint32_t PFA_LAYOUT_DIM2 = 2;
static constexpr uint32_t PFA_LAYOUT_DIM3 = 3;
static constexpr uint32_t PFA_QUERY_INDEX = 0;
static constexpr uint32_t PFA_VALUE_INDEX = 2;
static constexpr int32_t PFA_UNKNOWN_DIMS = -2;
static constexpr uint32_t PFA_DIM_NUMS_1 = 1;
static constexpr uint32_t PFA_LAYOUT_BNSD_BSND_DIMS = 4;
static constexpr uint32_t PFA_LAYOUT_TND_DIMS = 3;
static constexpr uint32_t PFA_LAYOUT_SH_DIMS = 2;
static constexpr uint32_t PFA_LAYOUT_BSH_DIMS = 3;
static constexpr uint32_t PFA_LAYOUT_NSD_DIMS = 3;
static constexpr uint32_t PFA_LAYOUT_BNSD_DIMS = 4;
static constexpr uint32_t PFA_LAYOUT_BSND_DIMS = 4;
static constexpr uint32_t PFA_ATTR_NUM_HEADS_INDEX = 0;
static constexpr uint32_t PFA_ATTR_NUM_KV_HEADS_INDEX = 5;
static constexpr uint32_t PFA_ATTENTION_OUT_INDEX = 0;
static constexpr uint32_t BSA_SOFTMAX_LSE_OUT_INDEX = 1;
static constexpr uint32_t PFA_ATTR_INPUT_LAYOUT_INDEX = 4;
static constexpr uint32_t BSA_ATTR_SOFTMAX_LSE_FLAG = 8;
static constexpr uint32_t BSA_ATTR_BLOCK_SHAPE = 9;
static constexpr uint32_t PFA_INPUT_ACTUAL_SEQ_LENGTHS_INDEX = 6;
static constexpr uint32_t PFA_INPUT_ACTUAL_SEQ_LENGTHS_KV_INDEX = 7;
static constexpr uint32_t PFA_QUANT_SCALE2_INDEX = 11;
} // namespace ops
namespace ops {

// Infers the softmax_lse output shape as {B, N, q_len} for BNSD/BSND/BSH layouts.
// TND is not supported; returns empty {0} for unsupported layouts.
static ge::graphStatus InferBsaLseShape(const char *layoutPtr,
                                        gert::Shape *lseShape,
                                        const gert::Shape *queryShape,
                                        const int64_t *numHeadsPtr)
{
    int64_t b = 0;
    int64_t n = 0;
    int64_t s = 0;
    if (strcmp(layoutPtr, "BNSD") == 0 || strcmp(layoutPtr, "BNSD_BSND") == 0) {
        b = (*queryShape)[PFA_LAYOUT_DIM0];
        n = (*queryShape)[PFA_LAYOUT_DIM1];
        s = (*queryShape)[PFA_LAYOUT_DIM2];
    } else if (strcmp(layoutPtr, "BSND") == 0) {
        b = (*queryShape)[PFA_LAYOUT_DIM0];
        n = (*queryShape)[PFA_LAYOUT_DIM2];
        s = (*queryShape)[PFA_LAYOUT_DIM1];
    } else if (strcmp(layoutPtr, "BSH") == 0) {
        b = (*queryShape)[PFA_LAYOUT_DIM0];
        n = *numHeadsPtr;
        s = (*queryShape)[PFA_LAYOUT_DIM1];
    } else {
        // Unsupported layout for LSE (e.g. TND, NSD, SH) – return empty tensor
        lseShape->SetDimNum(PFA_DIM_NUMS_1);
        (*lseShape)[PFA_LAYOUT_DIM0] = 0;
        return ge::GRAPH_SUCCESS;
    }
    lseShape->SetDimNum(3U);
    (*lseShape)[PFA_LAYOUT_DIM0] = b;
    (*lseShape)[PFA_LAYOUT_DIM1] = n;
    (*lseShape)[PFA_LAYOUT_DIM2] = s;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeBlitzSparseAttention(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("BlitzSparseAttention", "Context for inferring shape is nullptr!");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Enter BlitzSparseAttention inferShape impl.");
    // query shape : (B, S, H)
    const gert::Shape *queryShape = context->GetInputShape(PFA_QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);

    // value shape
    const gert::Shape *valueShape = context->GetInputShape(PFA_VALUE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, valueShape);

    // attentionOut: (B, S, H)
    gert::Shape *attentionOutShape = context->GetOutputShape(PFA_ATTENTION_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attentionOutShape);

    // softmax_lse output shape
    gert::Shape *softmaxLseShape = context->GetOutputShape(BSA_SOFTMAX_LSE_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxLseShape);

    *attentionOutShape = *queryShape;

    // UNKNOWN DIM
    if (((queryShape->GetDimNum() == PFA_DIM_NUMS_1) && (queryShape->GetDim(PFA_LAYOUT_DIM0) == PFA_UNKNOWN_DIMS)) ||
        ((valueShape->GetDimNum() == PFA_DIM_NUMS_1) && (valueShape->GetDim(PFA_LAYOUT_DIM0) == PFA_UNKNOWN_DIMS))) {
        attentionOutShape->SetDimNum(PFA_DIM_NUMS_1);
        (*attentionOutShape)[PFA_LAYOUT_DIM0] = PFA_UNKNOWN_DIMS;
        softmaxLseShape->SetDimNum(PFA_DIM_NUMS_1);
        (*softmaxLseShape)[PFA_LAYOUT_DIM0] = PFA_UNKNOWN_DIMS;
        return ge::GRAPH_SUCCESS;
    }

    // Get attr
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char *inputLayoutPtr = attrs->GetAttrPointer<char>(PFA_ATTR_INPUT_LAYOUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayoutPtr);
    const int64_t *numHeadsPtr = attrs->GetInt(PFA_ATTR_NUM_HEADS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, numHeadsPtr);
    const int64_t *numKeyValueHeadsPtr = attrs->GetInt(PFA_ATTR_NUM_KV_HEADS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, numKeyValueHeadsPtr);

    // KV_N除零保护, 当KV_N为零时KV_N = Q_N
    if (*numHeadsPtr == 0) {
        OP_LOGE(context->GetNodeName(), "numHeads can not be 0!");
        return ge::GRAPH_FAILED;
    }
    int64_t numKeyValueHeads = (*numKeyValueHeadsPtr == 0) ? *numHeadsPtr : *numKeyValueHeadsPtr;

    // Set output shape
    if (strcmp(inputLayoutPtr, "BNSD_BSND") == 0) {
        if (queryShape->GetDimNum() != PFA_LAYOUT_BNSD_BSND_DIMS) {
            OP_LOGE(context->GetNodeName(), "Layout BNSD_BSND, queryDims(%zu) must be 4!", queryShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
        int64_t outputD = (*valueShape)[PFA_LAYOUT_DIM3];
        outputD = (outputD == 0 || (*queryShape)[PFA_LAYOUT_DIM3] == 0) ? (*queryShape)[PFA_LAYOUT_DIM3] : outputD;
        (*attentionOutShape)[PFA_LAYOUT_DIM0] = (*queryShape)[PFA_LAYOUT_DIM0];
        (*attentionOutShape)[PFA_LAYOUT_DIM1] = (*queryShape)[PFA_LAYOUT_DIM2];
        (*attentionOutShape)[PFA_LAYOUT_DIM2] = (*queryShape)[PFA_LAYOUT_DIM1];
        (*attentionOutShape)[PFA_LAYOUT_DIM3] = outputD;
    } else if (strcmp(inputLayoutPtr, "TND") == 0) {
        if (queryShape->GetDimNum() != PFA_LAYOUT_TND_DIMS || valueShape->GetDimNum() != PFA_LAYOUT_TND_DIMS) {
            OP_LOGE(context->GetNodeName(), "Layout %s, queryDims(%zu) valueDims(%zu) must be 3!",
                inputLayoutPtr, queryShape->GetDimNum(), valueShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
        int64_t outputD = (*valueShape)[PFA_LAYOUT_DIM2];
        outputD = (outputD == 0 || (*queryShape)[PFA_LAYOUT_DIM2] == 0) ? (*queryShape)[PFA_LAYOUT_DIM2] : outputD;
        (*attentionOutShape)[PFA_LAYOUT_DIM0] = (*queryShape)[PFA_LAYOUT_DIM0];
        (*attentionOutShape)[PFA_LAYOUT_DIM1] = (*queryShape)[PFA_LAYOUT_DIM1];
        (*attentionOutShape)[PFA_LAYOUT_DIM2] = outputD;
    } else if (strcmp(inputLayoutPtr, "NTD_TND") == 0) {
        if (queryShape->GetDimNum() != PFA_LAYOUT_TND_DIMS || valueShape->GetDimNum() != PFA_LAYOUT_TND_DIMS) {
            OP_LOGE(context->GetNodeName(), "Layout NTD_TND, queryDims(%zu) valueDims(%zu) must be 3!",
                queryShape->GetDimNum(), valueShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
        (*attentionOutShape)[PFA_LAYOUT_DIM0] = (*queryShape)[PFA_LAYOUT_DIM1];
        (*attentionOutShape)[PFA_LAYOUT_DIM1] = (*queryShape)[PFA_LAYOUT_DIM0];
        (*attentionOutShape)[PFA_LAYOUT_DIM2] = (*valueShape)[PFA_LAYOUT_DIM2];
    } else if (strcmp(inputLayoutPtr, "SH") == 0) {
        if (queryShape->GetDimNum() != PFA_LAYOUT_SH_DIMS) {
            OP_LOGE(context->GetNodeName(), "Layout SH, queryDims(%zu) must be 2!", queryShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
    } else if (strcmp(inputLayoutPtr, "BSH") == 0) {
        if (queryShape->GetDimNum() != PFA_LAYOUT_BSH_DIMS) {
            OP_LOGE(context->GetNodeName(), "Layout BSH, queryDims(%zu) must be 3!", queryShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
        if (valueShape->GetDim(PFA_LAYOUT_DIM2) != -1) {
            int64_t outputH = (*valueShape)[PFA_LAYOUT_DIM2] / numKeyValueHeads * (*numHeadsPtr);
            outputH = (outputH == 0 || (*queryShape)[PFA_LAYOUT_DIM2] == 0) ? (*queryShape)[PFA_LAYOUT_DIM2] : outputH;
            (*attentionOutShape)[PFA_LAYOUT_DIM0] = (*queryShape)[PFA_LAYOUT_DIM0];
            (*attentionOutShape)[PFA_LAYOUT_DIM1] = (*queryShape)[PFA_LAYOUT_DIM1];
            (*attentionOutShape)[PFA_LAYOUT_DIM2] = outputH;
        }
    } else if (strcmp(inputLayoutPtr, "NSD") == 0) {
        if (queryShape->GetDimNum() != PFA_LAYOUT_NSD_DIMS) {
            OP_LOGE(context->GetNodeName(), "Layout NSD, queryDims(%zu) must be 3!", queryShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
    } else if (strcmp(inputLayoutPtr, "BNSD") == 0) {
        if (queryShape->GetDimNum() != PFA_LAYOUT_BNSD_DIMS) {
            OP_LOGE(context->GetNodeName(), "Layout BNSD, queryDims(%zu) must be 4!", queryShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
        int64_t outputD = (*valueShape)[PFA_LAYOUT_DIM3];
        outputD = (outputD == 0 || (*queryShape)[PFA_LAYOUT_DIM3] == 0) ? (*queryShape)[PFA_LAYOUT_DIM3] : outputD;
        (*attentionOutShape)[PFA_LAYOUT_DIM0] = (*queryShape)[PFA_LAYOUT_DIM0];
        (*attentionOutShape)[PFA_LAYOUT_DIM1] = (*queryShape)[PFA_LAYOUT_DIM1];
        (*attentionOutShape)[PFA_LAYOUT_DIM2] = (*queryShape)[PFA_LAYOUT_DIM2];
        (*attentionOutShape)[PFA_LAYOUT_DIM3] = outputD;
    } else if (strcmp(inputLayoutPtr, "BSND") == 0) {
        if (queryShape->GetDimNum() != PFA_LAYOUT_BSND_DIMS) {
            OP_LOGE(context->GetNodeName(), "Layout BSND, queryDims(%zu) must be 4!", queryShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
        int64_t outputD = (*valueShape)[PFA_LAYOUT_DIM3];
        outputD = (outputD == 0 || (*queryShape)[PFA_LAYOUT_DIM3] == 0) ? (*queryShape)[PFA_LAYOUT_DIM3] : outputD;
        (*attentionOutShape)[PFA_LAYOUT_DIM0] = (*queryShape)[PFA_LAYOUT_DIM0];
        (*attentionOutShape)[PFA_LAYOUT_DIM1] = (*queryShape)[PFA_LAYOUT_DIM1];
        (*attentionOutShape)[PFA_LAYOUT_DIM2] = (*queryShape)[PFA_LAYOUT_DIM2];
        (*attentionOutShape)[PFA_LAYOUT_DIM3] = outputD;
    } else {
        OP_LOGE(context->GetNodeName(), "Invalid input layout: %s, not support!", inputLayoutPtr);
        return ge::GRAPH_FAILED;
    }

    // Validate block_shape attr. Both dimensions accept {128, 256, 512, 1024};
    // all 16 combinations are supported.
    auto blockShapeList = attrs->GetListInt(BSA_ATTR_BLOCK_SHAPE);
    if (blockShapeList != nullptr && blockShapeList->GetSize() >= 2) {
        int64_t bsQ = blockShapeList->GetData()[0];
        int64_t bsKV = blockShapeList->GetData()[1];
        if (bsQ != 128 && bsQ != 256 && bsQ != 512 && bsQ != 1024) {
            OP_LOGE(context->GetNodeName(),
                "block_shape[0] (BLOCK_SIZE_Q) must be 128, 256, 512, or 1024 (got %ld).",
                bsQ);
            return ge::GRAPH_FAILED;
        }
        if (bsKV != 128 && bsKV != 256 && bsKV != 512 && bsKV != 1024) {
            OP_LOGE(context->GetNodeName(),
                "block_shape[1] (BLOCK_SIZE_KV) must be 128, 256, 512, or 1024 (got %ld).",
                bsKV);
            return ge::GRAPH_FAILED;
        }
    }

    // Infer softmax_lse output shape
    const bool *softmaxLseFlagPtr = attrs->GetAttrPointer<bool>(BSA_ATTR_SOFTMAX_LSE_FLAG);
    bool softmaxLseFlag = (softmaxLseFlagPtr != nullptr) ? *softmaxLseFlagPtr : false;
    if (softmaxLseFlag) {
        InferBsaLseShape(inputLayoutPtr, softmaxLseShape, queryShape, numHeadsPtr);
    } else {
        softmaxLseShape->SetDimNum(PFA_DIM_NUMS_1);
        (*softmaxLseShape)[PFA_LAYOUT_DIM0] = 0;
    }

    OP_LOGD(context->GetNodeName(), "BlitzSparseAttention inferShape end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeBlitzSparseAttention(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("BlitzSparseAttention", "Context for inferring data type is nullptr!");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Enter BlitzSparseAttention inferDataType impl.");
    // default set q's dtype as BSA's output type
    ge::DataType outputType = context->GetInputDataType(PFA_QUERY_INDEX);
    if (context->GetOptionalInputDataType(PFA_QUANT_SCALE2_INDEX) != ge::DT_UNDEFINED) { // 10 is quant_scale2's index
        outputType = ge::DT_INT8;
    } else if (outputType == ge::DT_INT8) {
        outputType = ge::DT_FLOAT16;
    }
    // attention_out, outidx:0
    context->SetOutputDataType(PFA_ATTENTION_OUT_INDEX, outputType);
    // softmax_lse is always float32 regardless of query dtype
    context->SetOutputDataType(BSA_SOFTMAX_LSE_OUT_INDEX, ge::DT_FLOAT);
    OP_LOGD(context->GetNodeName(), "BlitzSparseAttention inferDataType end.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BlitzSparseAttention)
    .InferShape(InferShapeBlitzSparseAttention)
    .InferDataType(InferDataTypeBlitzSparseAttention)
    .InputsDataDependency({PFA_INPUT_ACTUAL_SEQ_LENGTHS_INDEX, PFA_INPUT_ACTUAL_SEQ_LENGTHS_KV_INDEX});
} // namespace ops