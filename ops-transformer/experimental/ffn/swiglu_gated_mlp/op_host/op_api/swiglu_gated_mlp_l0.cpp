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

#include "swiglu_gated_mlp_l0.h"

#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "../../op_kernel/swiglu_gated_mlp_stage.h"

using namespace op;

namespace Ops::NN {
const aclTensor* ExecMmOp(
    const aclTensor* self,
    const aclTensor* mat2,
    int8_t cubeMathType,
    aclOpExecutor* executor);
}  // namespace Ops::NN

namespace l0op {

OP_TYPE_REGISTER(SwigluGatedMlp);

namespace {
op::Shape Make2DShape(int64_t dim0, int64_t dim1)
{
    op::Shape shape;
    shape.AppendDim(dim0);
    shape.AppendDim(dim1);
    return shape;
}

int64_t FlattenLeadingDims(const op::Shape& shape)
{
    int64_t rows = 1;
    for (size_t i = 0; i + 1 < shape.GetDimNum(); ++i) {
        rows *= shape.GetDim(i);
    }
    return rows;
}

int64_t MakeStageAttr(int64_t stage, op::DataType dtype)
{
    if (dtype == op::DataType::DT_FLOAT) {
        return stage + WG_MLP_STAGE_DTYPE_STRIDE;
    }
    if (dtype == op::DataType::DT_BF16) {
        return stage + 2 * WG_MLP_STAGE_DTYPE_STRIDE;
    }
    return stage;
}

const aclTensor* AddSelfKernelStage(
    const aclTensor* x,
    const aclTensor* weight,
    const aclTensor* auxWeight,
    aclTensor* out,
    int64_t stage,
    aclOpExecutor* executor)
{
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SwigluGatedMlp stage output is null.");
        return nullptr;
    }

    auto ret = INFER_SHAPE(
        SwigluGatedMlp,
        OP_INPUT(x, weight, auxWeight),
        OP_OUTPUT(out),
        OP_ATTR(stage));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_INFERSHAPE_ERROR, "SwigluGatedMlp stage infer failed, stage=%ld ret=%d.",
                stage, ret);
        return nullptr;
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        SwigluGatedMlp,
        OP_INPUT(x, weight, auxWeight),
        OP_OUTPUT(out),
        OP_ATTR(stage));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
        ret != ACLNN_SUCCESS, return nullptr, "SwigluGatedMlp stage add to launcher list failed.");
    return out;
}

const aclTensor* AddSwigluStage(
    const aclTensor* gateUp,
    const aclTensor* gateUpWeight,
    const aclTensor* downWeight,
    aclTensor* hidden,
    op::DataType dtype,
    aclOpExecutor* executor)
{
    return AddSelfKernelStage(
        gateUp, gateUpWeight, downWeight, hidden, MakeStageAttr(WG_MLP_STAGE_SWIGLU, dtype), executor);
}
}  // namespace

const aclTensor* SwigluGatedMlp(
    const aclTensor* x,
    const aclTensor* gateUpWeight,
    const aclTensor* downWeight,
    int64_t cubeMathType,
    aclOpExecutor* executor)
{
    L0_DFX(SwigluGatedMlp, x, gateUpWeight, downWeight, cubeMathType);
    const auto& xShape = x->GetViewShape();
    const auto& gateUpShape = gateUpWeight->GetViewShape();
    const int64_t rows = FlattenLeadingDims(xShape);
    const int64_t hiddenSize = xShape.GetDim(xShape.GetDimNum() - 1);
    const int64_t gateUpCols = gateUpShape.GetDim(1);
    const int64_t intermediate = gateUpCols / 2;

    auto x2d = executor->CreateView(x, Make2DShape(rows, hiddenSize), x->GetViewOffset());
    auto hidden = executor->AllocTensor(Make2DShape(rows, intermediate), x->GetDataType(), Format::FORMAT_ND);
    if (x2d == nullptr || hidden == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc SwigluGatedMlp internal tensors failed.");
        return nullptr;
    }

    const auto matmulCubeMathType = static_cast<int8_t>(cubeMathType);
    const aclTensor* gateUp = Ops::NN::ExecMmOp(x2d, gateUpWeight, matmulCubeMathType, executor);
    if (gateUp == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SwigluGatedMlp ExecMmOp MM1 failed.");
        return nullptr;
    }

    if (AddSwigluStage(gateUp, gateUpWeight, downWeight, hidden, x->GetDataType(), executor) == nullptr) {
        return nullptr;
    }

    const aclTensor* nativeOut = Ops::NN::ExecMmOp(hidden, downWeight, matmulCubeMathType, executor);
    if (nativeOut == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SwigluGatedMlp ExecMmOp MM2 failed.");
        return nullptr;
    }

    return nativeOut;
}

}  // namespace l0op
