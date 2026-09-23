/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Copyright (c) 2026 联通（广东）产业互联网有限公司.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file swiglu_gated_mlp_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "../op_kernel/swiglu_gated_mlp_stage.h"

using namespace ge;

namespace {
constexpr size_t SWIGLU_GATED_MLP_IN_X = 0;
constexpr size_t SWIGLU_GATED_MLP_IN_GATE_UP_WEIGHT = 1;
constexpr size_t SWIGLU_GATED_MLP_IN_DOWN_WEIGHT = 2;
constexpr size_t SWIGLU_GATED_MLP_OUT_Y = 0;

constexpr int64_t WEIGHT_RANK = 2;
constexpr int64_t MIN_X_RANK = 2;
constexpr int64_t SPLIT_NUM = 2;
constexpr size_t ATTR_CUBE_MATH_TYPE = 0;

bool IsUnknownDimNum(const gert::Shape& shape)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) == ge::UNKNOWN_DIM_NUM;
}

bool IsKnownDim(int64_t dim)
{
    return dim >= 0;
}

int64_t GetStage(gert::InferShapeContext* context)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return 0;
    }
    const int64_t* stage = attrs->GetAttrPointer<int64_t>(ATTR_CUBE_MATH_TYPE);
    if (stage == nullptr) {
        return 0;
    }
    int64_t normalized_stage = *stage;
    if (normalized_stage >= WG_MLP_STAGE_DTYPE_STRIDE) {
        normalized_stage %= WG_MLP_STAGE_DTYPE_STRIDE;
    }
    return normalized_stage;
}

ge::graphStatus ValidateInputRanks(const gert::Shape& x_shape, const gert::Shape& gate_up_weight_shape,
    const gert::Shape& down_weight_shape)
{
    if (x_shape.GetDimNum() < MIN_X_RANK) {
        OP_LOGE("SwigluGatedMlp",
                "The rank of input x must be at least %ld, but got %zu.",
                MIN_X_RANK, x_shape.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    if (!IsUnknownDimNum(gate_up_weight_shape) && gate_up_weight_shape.GetDimNum() != WEIGHT_RANK) {
        OP_LOGE("SwigluGatedMlp",
                "The rank of input gate_up_weight must be %ld, but got %zu.",
                WEIGHT_RANK, gate_up_weight_shape.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    if (!IsUnknownDimNum(down_weight_shape) && down_weight_shape.GetDimNum() != WEIGHT_RANK) {
        OP_LOGE("SwigluGatedMlp",
                "The rank of input down_weight must be %ld, but got %zu.",
                WEIGHT_RANK, down_weight_shape.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferMatmulStageShape(const gert::Shape& x_shape, const gert::Shape& gate_up_weight_shape,
    gert::Shape* y_shape)
{
    if (gate_up_weight_shape.GetDimNum() != WEIGHT_RANK) {
        return ge::GRAPH_FAILED;
    }
    *y_shape = x_shape;
    y_shape->SetDim(x_shape.GetDimNum() - 1, gate_up_weight_shape.GetDim(1));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferSwigluStageShape(const gert::Shape& x_shape, gert::Shape* y_shape)
{
    *y_shape = x_shape;
    const int64_t last_dim = x_shape.GetDim(x_shape.GetDimNum() - 1);
    if (!IsKnownDim(last_dim)) {
        return ge::GRAPH_SUCCESS;
    }
    if (last_dim <= 0 || last_dim % SPLIT_NUM != 0) {
        OP_LOGE("SwigluGatedMlp",
                "The stage swiglu input last dimension [%ld] must be positive and divisible by %ld.",
                last_dim, SPLIT_NUM);
        return ge::GRAPH_FAILED;
    }
    y_shape->SetDim(x_shape.GetDimNum() - 1, last_dim / SPLIT_NUM);
    return ge::GRAPH_SUCCESS;
}

struct WeightDims {
    int64_t gate_up_hidden = ge::UNKNOWN_DIM;
    int64_t gate_up_out = ge::UNKNOWN_DIM;
    int64_t down_in = ge::UNKNOWN_DIM;
    int64_t down_out = ge::UNKNOWN_DIM;
};

WeightDims GetWeightDims(const gert::Shape& gate_up_weight_shape, const gert::Shape& down_weight_shape)
{
    WeightDims dims;
    if (!IsUnknownDimNum(gate_up_weight_shape)) {
        dims.gate_up_hidden = gate_up_weight_shape.GetDim(0);
        dims.gate_up_out = gate_up_weight_shape.GetDim(1);
    }
    if (!IsUnknownDimNum(down_weight_shape)) {
        dims.down_in = down_weight_shape.GetDim(0);
        dims.down_out = down_weight_shape.GetDim(1);
    }
    return dims;
}

ge::graphStatus ValidateWeightDims(int64_t x_hidden, const WeightDims& dims)
{
    if (IsKnownDim(dims.gate_up_hidden) && IsKnownDim(x_hidden) && dims.gate_up_hidden != x_hidden) {
        OP_LOGE("SwigluGatedMlp",
                "The hidden dimension of x [%ld] and gate_up_weight [%ld] must be the same.",
                x_hidden, dims.gate_up_hidden);
        return ge::GRAPH_FAILED;
    }

    if (IsKnownDim(dims.gate_up_out) && (dims.gate_up_out <= 0 || dims.gate_up_out % SPLIT_NUM != 0)) {
        OP_LOGE("SwigluGatedMlp",
                "The second dimension of gate_up_weight [%ld] must be positive and divisible by %ld.",
                dims.gate_up_out, SPLIT_NUM);
        return ge::GRAPH_FAILED;
    }

    if (IsKnownDim(dims.gate_up_out) && IsKnownDim(dims.down_in) &&
        (dims.gate_up_out / SPLIT_NUM) != dims.down_in) {
        OP_LOGE("SwigluGatedMlp",
                "The down_weight input dimension [%ld] must equal gate_up_weight second dimension / %ld [%ld].",
                dims.down_in, SPLIT_NUM, dims.gate_up_out / SPLIT_NUM);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
}  // namespace

namespace ops {

static ge::graphStatus InferShapeForSwigluGatedMlp(gert::InferShapeContext* context)
{
    auto x_shape = context->GetInputShape(SWIGLU_GATED_MLP_IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);

    auto gate_up_weight_shape = context->GetInputShape(SWIGLU_GATED_MLP_IN_GATE_UP_WEIGHT);
    OP_CHECK_NULL_WITH_CONTEXT(context, gate_up_weight_shape);

    auto down_weight_shape = context->GetInputShape(SWIGLU_GATED_MLP_IN_DOWN_WEIGHT);
    OP_CHECK_NULL_WITH_CONTEXT(context, down_weight_shape);

    auto y_shape = context->GetOutputShape(SWIGLU_GATED_MLP_OUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);

    if (IsUnknownDimNum(*x_shape)) {
        y_shape->SetDimNum(1);
        y_shape->SetDim(0, ge::UNKNOWN_DIM_NUM);
        return ge::GRAPH_SUCCESS;
    }

    if (ValidateInputRanks(*x_shape, *gate_up_weight_shape, *down_weight_shape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const int64_t stage = GetStage(context);
    if (stage == WG_MLP_STAGE_MM1 || stage == WG_MLP_STAGE_MM2) {
        return InferMatmulStageShape(*x_shape, *gate_up_weight_shape, y_shape);
    }

    if (stage == WG_MLP_STAGE_SWIGLU) {
        return InferSwigluStageShape(*x_shape, y_shape);
    }

    *y_shape = *x_shape;
    const int64_t x_hidden = x_shape->GetDim(x_shape->GetDimNum() - 1);
    const WeightDims dims = GetWeightDims(*gate_up_weight_shape, *down_weight_shape);
    if (ValidateWeightDims(x_hidden, dims) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    y_shape->SetDim(x_shape->GetDimNum() - 1, dims.down_out);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForSwigluGatedMlp(gert::InferDataTypeContext* context)
{
    const ge::DataType x_dtype = context->GetInputDataType(SWIGLU_GATED_MLP_IN_X);
    const ge::DataType gate_up_weight_dtype = context->GetInputDataType(SWIGLU_GATED_MLP_IN_GATE_UP_WEIGHT);
    const ge::DataType down_weight_dtype = context->GetInputDataType(SWIGLU_GATED_MLP_IN_DOWN_WEIGHT);
    if (x_dtype != gate_up_weight_dtype || x_dtype != down_weight_dtype) {
        OP_LOGE("SwigluGatedMlp",
                "The data types of x, gate_up_weight and down_weight must be the same, but got [%d, %d, %d].",
                static_cast<int32_t>(x_dtype),
                static_cast<int32_t>(gate_up_weight_dtype),
                static_cast<int32_t>(down_weight_dtype));
        return ge::GRAPH_FAILED;
    }

    return context->SetOutputDataType(SWIGLU_GATED_MLP_OUT_Y, x_dtype);
}

IMPL_OP_INFERSHAPE(SwigluGatedMlp)
    .InferShape(InferShapeForSwigluGatedMlp)
    .InferDataType(InferDataTypeForSwigluGatedMlp);

}  // namespace ops
