/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_swiglu_infershape.cpp
 * \brief MatmulSwiglu 的 shape / dtype 推导
 *
 * 输出 y 形状为 [M, N], 其中 M 来自 x 的行, N = (weight 的 2N 维) / 2。
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
namespace {
constexpr size_t IN_X = 0;
constexpr size_t IN_WEIGHT = 1;
constexpr size_t OUT_Y = 0;
constexpr size_t ATTR_TRANSPOSE_WEIGHT = 0;
constexpr int64_t SPLIT_NUM = 2;
}  // namespace

static ge::graphStatus InferShapeMatmulSwiglu(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin InferShapeMatmulSwiglu");

    const gert::Shape* xShape = context->GetInputShape(IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* wShape = context->GetInputShape(IN_WEIGHT);
    OP_CHECK_NULL_WITH_CONTEXT(context, wShape);
    gert::Shape* yShape = context->GetOutputShape(OUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    const auto attrs = context->GetAttrs();
    const bool* transW = (attrs == nullptr) ? nullptr : attrs->GetAttrPointer<bool>(ATTR_TRANSPOSE_WEIGHT);
    const bool transposeWeight = (transW == nullptr) ? false : *transW;

    const size_t xDimNum = xShape->GetDimNum();
    const size_t wDimNum = wShape->GetDimNum();
    OP_CHECK_IF(xDimNum < 2 || wDimNum != 2,
        OP_LOGE(context, "MatmulSwiglu: x dim must be >=2 and weight dim must be 2, got x=%zu w=%zu",
                xDimNum, wDimNum),
        return ge::GRAPH_FAILED);

    // weight 为 [K, 2N] (默认) 或 [2N, K] (transpose)
    const int64_t twoN = transposeWeight ? wShape->GetDim(0) : wShape->GetDim(1);
    OP_CHECK_IF(twoN > 0 && twoN % SPLIT_NUM != 0,
        OP_LOGE(context, "MatmulSwiglu: weight 2N dim [%ld] must be divisible by 2", twoN),
        return ge::GRAPH_FAILED);
    const int64_t n = (twoN < 0) ? -1 : twoN / SPLIT_NUM;

    // y 与 x 同 rank, 最后一维替换为 N, 其余沿用 x
    yShape->SetDimNum(xDimNum);
    for (size_t i = 0; i + 1 < xDimNum; i++) {
        yShape->SetDim(i, xShape->GetDim(i));
    }
    yShape->SetDim(xDimNum - 1, n);

    OP_LOGD(context->GetNodeName(), "End InferShapeMatmulSwiglu");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMatmulSwiglu(gert::InferDataTypeContext* context)
{
    const ge::DataType dtype = context->GetInputDataType(IN_X);
    context->SetOutputDataType(OUT_Y, dtype);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MatmulSwiglu)
    .InferShape(InferShapeMatmulSwiglu)
    .InferDataType(InferDataTypeMatmulSwiglu);
}  // namespace ops
