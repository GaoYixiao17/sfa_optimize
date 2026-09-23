/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/ops_log.h"

using namespace ge;

namespace ops {
// INPUT
constexpr uint32_t TOKEN_X_INPUT_INDEX = 0;
constexpr uint32_t WEIGHT_KV_INPUT_INDEX = 1;
constexpr uint32_t WEIGHT_WGATE_INPUT_INDEX = 2;
constexpr uint32_t STATE_CACHE_INPUT_INDEX = 3;
constexpr uint32_t APE_INPUT_INDEX = 4;

// INPUT(OPTION)
constexpr uint32_t X_DESCALE_INPUT_INDEX = 5;
constexpr uint32_t WKV_DESCALE_INPUT_INDEX = 6;
constexpr uint32_t WGATE_DESCALE_INPUT_INDEX = 7;
constexpr uint32_t STATE_BLOCK_TABLE_INPUT_INDEX = 8;
constexpr uint32_t CU_SEQ_LEN_INPUT_INDEX = 9;
constexpr uint32_t SEQ_USED_INPUT_INDEX = 10;
constexpr uint32_t START_POS_INPUT_INDEX = 11;

// ATTR
constexpr uint32_t QUANT_MODE_ATTR_INDEX = 0;
constexpr uint32_t CMP_RATIO_ATTR_INDEX = 1;
constexpr uint32_t COFF_ATTR_INDEX = 2;
constexpr uint32_t CACHE_MODE_ATTR_INDEX = 3;
constexpr uint32_t STATE_CACHE_STRIDE_DIM0_ATTR_INDEX = 4;

// OUTPUT
constexpr uint32_t CMP_KV_OUTPUT_INDEX = 0;

// ATTR DEFAULT VALUE
constexpr uint32_t CMP_RATIO_VALUE = 4;
constexpr uint32_t COFF_VALUE = 1;
constexpr uint32_t QUANT_MODE_VALUE = 1;

struct QuantCompressorProtoShapeParam {
    bool isBsMerge{false};
    int64_t B{0};
    int64_t T{0};
    int64_t S{0};
    int64_t Sr{0};
    int64_t H{0};
    int64_t D{0};
};

// tmp
constexpr uint32_t DIM_NUM_1 = 1;
constexpr uint32_t DIM_NUM_2 = 2;
constexpr uint32_t DIM_NUM_3 = 3;
constexpr uint32_t DIM_NUM_4 = 4;
constexpr uint32_t DIM_INDEX_0 = 0;
constexpr uint32_t DIM_INDEX_1 = 1;
constexpr uint32_t DIM_INDEX_2 = 2;
constexpr uint32_t DIM_INDEX_3 = 3;

ge::graphStatus GetQuantCompressorShapeDim(const gert::InferShapeContext *context,
                                           QuantCompressorProtoShapeParam &shapeParam)
{
    auto xShape = context->GetRequiredInputShape(TOKEN_X_INPUT_INDEX); // (B, S, H) | (T, H)
    OPS_LOG_E_IF_NULL(context, xShape, return ge::GRAPH_FAILED)
    auto wkvShape = context->GetRequiredInputShape(WEIGHT_KV_INPUT_INDEX); // (coff * D, H)
    OPS_LOG_E_IF_NULL(context, wkvShape, return ge::GRAPH_FAILED)

    auto attr = context->GetAttrs();
    const int *cmpRatioPtr = attr->GetAttrPointer<int>(CMP_RATIO_ATTR_INDEX);
    int cmpRatio = (cmpRatioPtr != nullptr) ? *cmpRatioPtr : CMP_RATIO_VALUE;
    OPS_LOG_E_IF(cmpRatio <= 0, context, return ge::GRAPH_FAILED, "cmp_ratio should be greater than 0");

    const int *coffPtr = attr->GetAttrPointer<int>(COFF_ATTR_INDEX);
    int coff = (coffPtr != nullptr) ? *coffPtr : COFF_VALUE;
    OPS_LOG_E_IF(coff <= 0, context, return ge::GRAPH_FAILED, "coff should be greater than 0");

    if (xShape->GetDimNum() == DIM_NUM_3) { // BS
        shapeParam.isBsMerge = false;
        shapeParam.B = xShape->GetDim(DIM_INDEX_0);
        shapeParam.S = xShape->GetDim(DIM_INDEX_1);
        shapeParam.Sr = (xShape->GetDim(DIM_INDEX_1) + cmpRatio - 1) / cmpRatio;
        shapeParam.H = xShape->GetDim(DIM_INDEX_2);
        shapeParam.T = shapeParam.B * shapeParam.S;
    } else { // T
        auto cuSeqlensShape = context->GetRequiredInputShape(CU_SEQ_LEN_INPUT_INDEX); // (B+1,)
        OPS_LOG_E_IF_NULL(context, cuSeqlensShape, return ge::GRAPH_FAILED)
        shapeParam.isBsMerge = true;
        shapeParam.Sr = std::min(xShape->GetDim(DIM_INDEX_0),
                                 xShape->GetDim(DIM_INDEX_0) / cmpRatio + cuSeqlensShape->GetDim(DIM_INDEX_0) - 1);
        shapeParam.T = xShape->GetDim(DIM_INDEX_0);
        shapeParam.H = xShape->GetDim(DIM_INDEX_1);
    }

    shapeParam.D = wkvShape->GetDim(DIM_INDEX_0) / coff;

    return GRAPH_SUCCESS;
}

ge::graphStatus SetQuantCompressorShapeDim(const QuantCompressorProtoShapeParam &shapeParam,
                                           gert::InferShapeContext *context)
{
    auto cmpKvShape = context->GetOutputShape(CMP_KV_OUTPUT_INDEX); // query: (B, S, N, Hckv) | (T, N, Hckv)
    OPS_LOG_E_IF_NULL(context, cmpKvShape, return ge::GRAPH_FAILED)
    // Set output shape
    if (!shapeParam.isBsMerge) {
        cmpKvShape->SetDimNum(DIM_NUM_3); // (B, Sr, H)
        cmpKvShape->SetDim(DIM_INDEX_0, shapeParam.B);
        cmpKvShape->SetDim(DIM_INDEX_1, shapeParam.Sr);
        cmpKvShape->SetDim(DIM_INDEX_2, shapeParam.D);
    } else {
        cmpKvShape->SetDimNum(DIM_NUM_2); // (T, N, Hckv)
        cmpKvShape->SetDim(DIM_INDEX_0, shapeParam.Sr);
        cmpKvShape->SetDim(DIM_INDEX_1, shapeParam.D);
    }

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeQuantCompressor(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("QuantCompressor", "Context is nullptr."),
                return ge::GRAPH_FAILED);
    OPS_LOG_I(context->GetNodeName(), "Enter QuantCompressor inferDataType impl.");

    context->SetOutputDataType(CMP_KV_OUTPUT_INDEX, ge::DT_BF16);

    return GRAPH_SUCCESS;
}

ge::graphStatus InferShapeQuantCompressor(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("QuantCompressor", "Context is nullptr."),
                return ge::GRAPH_FAILED);
    OPS_LOG_I(context->GetNodeName(), "Enter QuantCompressor infershape impl.");

    QuantCompressorProtoShapeParam shapeParam{};
    auto apiRet = GetQuantCompressorShapeDim(context, shapeParam);
    OPS_LOG_E_IF((apiRet != GRAPH_SUCCESS), context, return ge::GRAPH_FAILED, "Context get input shape failed");

    apiRet = SetQuantCompressorShapeDim(shapeParam, context);
    OPS_LOG_E_IF((apiRet != GRAPH_SUCCESS), context, return ge::GRAPH_FAILED, "Context set output shape failed");

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QuantCompressor).InferShape(InferShapeQuantCompressor).InferDataType(InferDataTypeQuantCompressor);
} // namespace ops
