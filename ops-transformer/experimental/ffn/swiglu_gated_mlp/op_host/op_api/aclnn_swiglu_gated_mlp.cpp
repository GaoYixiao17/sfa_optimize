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

#include "aclnn_swiglu_gated_mlp.h"
#include "swiglu_gated_mlp_l0.h"

#include "aclnn_kernels/contiguous.h"
#include "op_api/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static const std::initializer_list<DataType> ASCEND_910_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16,
    DataType::DT_FLOAT
};

static const std::initializer_list<DataType> ASCEND_910B_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16,
    DataType::DT_FLOAT,
    DataType::DT_BF16
};

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
        return ASCEND_910B_DTYPE_SUPPORT_LIST;
    }
    return ASCEND_910_DTYPE_SUPPORT_LIST;
}

static bool CheckNotNull(
    const aclTensor* x,
    const aclTensor* gateUpWeight,
    const aclTensor* downWeight,
    const aclTensor* y)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(gateUpWeight, return false);
    OP_CHECK_NULL(downWeight, return false);
    OP_CHECK_NULL(y, return false);
    return true;
}

static bool CheckDtypeValid(
    const aclTensor* x,
    const aclTensor* gateUpWeight,
    const aclTensor* downWeight,
    const aclTensor* y)
{
    auto supportList = GetDtypeSupportList();

    OP_CHECK_DTYPE_NOT_SUPPORT(x, supportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(gateUpWeight, supportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(downWeight, supportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(y, supportList, return false);

    OP_CHECK_DTYPE_NOT_MATCH(gateUpWeight, x->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(downWeight, x->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(y, x->GetDataType(), return false);

    return true;
}

static bool CheckShapeRank(const aclTensor* x, const aclTensor* gateUpWeight, const aclTensor* downWeight,
    const aclTensor* y)
{
    OP_CHECK_MAX_DIM(x, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(gateUpWeight, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(downWeight, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(y, MAX_SUPPORT_DIMS_NUMS, return false);

    const auto& xShape = x->GetViewShape();
    const auto& gateUpShape = gateUpWeight->GetViewShape();
    const auto& downShape = downWeight->GetViewShape();
    const auto& yShape = y->GetViewShape();
    const int64_t xDimNum = static_cast<int64_t>(xShape.GetDimNum());
    const int64_t yDimNum = static_cast<int64_t>(yShape.GetDimNum());

    if (xDimNum < 2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x dim num must be >= 2, but got %ld.", xDimNum);
        return false;
    }

    if (gateUpShape.GetDimNum() != 2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gate_up_weight must be 2D, but got dim num %zu.",
                gateUpShape.GetDimNum());
        return false;
    }

    if (downShape.GetDimNum() != 2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "down_weight must be 2D, but got dim num %zu.",
                downShape.GetDimNum());
        return false;
    }

    if (yDimNum != xDimNum) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y dim num[%ld] must equal x dim num[%ld].", yDimNum, xDimNum);
        return false;
    }
    return true;
}

static bool CheckPositiveDims(const aclTensor* x, const aclTensor* gateUpWeight, const aclTensor* downWeight)
{
    const auto& xShape = x->GetViewShape();
    const auto& gateUpShape = gateUpWeight->GetViewShape();
    const auto& downShape = downWeight->GetViewShape();
    const int64_t xDimNum = static_cast<int64_t>(xShape.GetDimNum());

    const int64_t hiddenSize = xShape.GetDim(xDimNum - 1);
    const int64_t gateUpIn = gateUpShape.GetDim(0);
    const int64_t gateUpSize = gateUpShape.GetDim(1);
    const int64_t downIn = downShape.GetDim(0);
    const int64_t outSize = downShape.GetDim(1);
    if (hiddenSize <= 0 || gateUpIn <= 0 || gateUpSize <= 0 || downIn <= 0 || outSize <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "All reduced shape dims must be positive, but got hidden:%ld gateIn:%ld gateSize:%ld "
                "downIn:%ld out:%ld.",
                hiddenSize, gateUpIn, gateUpSize, downIn, outSize);
        return false;
    }
    return true;
}

static bool CheckWeightShapeRelation(const aclTensor* x, const aclTensor* gateUpWeight, const aclTensor* downWeight)
{
    const auto& xShape = x->GetViewShape();
    const auto& gateUpShape = gateUpWeight->GetViewShape();
    const auto& downShape = downWeight->GetViewShape();
    const int64_t xDimNum = static_cast<int64_t>(xShape.GetDimNum());
    const int64_t hiddenSize = xShape.GetDim(xDimNum - 1);
    const int64_t gateUpIn = gateUpShape.GetDim(0);
    const int64_t gateUpSize = gateUpShape.GetDim(1);
    const int64_t downIn = downShape.GetDim(0);

    if (gateUpIn != hiddenSize) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "hidden mismatch, x hidden:%ld gate_up_weight dim0:%ld.", hiddenSize, gateUpIn);
        return false;
    }

    if (gateUpSize % 2 != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "gate_up_weight dim1 must be divisible by 2, but got %ld.", gateUpSize);
        return false;
    }

    const int64_t intermediateSize = gateUpSize / 2;
    if (downIn != intermediateSize) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "down_weight dim0:%ld must equal intermediate size:%ld.", downIn, intermediateSize);
        return false;
    }
    return true;
}

static bool CheckOutputShape(const aclTensor* x, const aclTensor* downWeight, const aclTensor* y)
{
    const auto& xShape = x->GetViewShape();
    const auto& downShape = downWeight->GetViewShape();
    const auto& yShape = y->GetViewShape();
    const int64_t xDimNum = static_cast<int64_t>(xShape.GetDimNum());
    const int64_t yDimNum = static_cast<int64_t>(yShape.GetDimNum());
    const int64_t outSize = downShape.GetDim(1);

    for (int64_t i = 0; i + 1 < xDimNum; ++i) {
        if (yShape.GetDim(i) != xShape.GetDim(i)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "y dim[%ld]=%ld must equal x dim[%ld]=%ld.",
                    i, yShape.GetDim(i), i, xShape.GetDim(i));
            return false;
        }
    }

    if (yShape.GetDim(yDimNum - 1) != outSize) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "y last dim:%ld must equal down_weight dim1:%ld.",
                yShape.GetDim(yDimNum - 1), outSize);
        return false;
    }

    return true;
}

static bool CheckShape(const aclTensor* x, const aclTensor* gateUpWeight, const aclTensor* downWeight,
    const aclTensor* y)
{
    return CheckShapeRank(x, gateUpWeight, downWeight, y) &&
        CheckPositiveDims(x, gateUpWeight, downWeight) &&
        CheckWeightShapeRelation(x, gateUpWeight, downWeight) &&
        CheckOutputShape(x, downWeight, y);
}

static bool CheckCubeMathType(int64_t cubeMathType)
{
    if (cubeMathType < 0 || cubeMathType > 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cubeMathType must be 0 or 1, but got %ld.", cubeMathType);
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* x,
    const aclTensor* gateUpWeight,
    const aclTensor* downWeight,
    int64_t cubeMathType,
    const aclTensor* y)
{
    CHECK_RET(CheckNotNull(x, gateUpWeight, downWeight, y), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(x, gateUpWeight, downWeight, y), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(x, gateUpWeight, downWeight, y), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckCubeMathType(cubeMathType), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static const aclTensor* MaybeContiguous(const aclTensor* input, aclOpExecutor* executor)
{
    auto contiguousInput = l0op::Contiguous(input, executor);
    CHECK_RET(contiguousInput != nullptr, nullptr);
    return contiguousInput;
}

}  // namespace

aclnnStatus aclnnSwigluGatedMlpGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* gateUpWeight,
    const aclTensor* downWeight,
    int64_t cubeMathType,
    aclTensor* y,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnSwigluGatedMlp, DFX_IN(x, gateUpWeight, downWeight, cubeMathType), DFX_OUT(y));

    auto ret = CheckParams(x, gateUpWeight, downWeight, cubeMathType, y);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (x->IsEmpty() || y->IsEmpty()) {
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    const aclTensor* xExec = MaybeContiguous(x, uniqueExecutor.get());
    CHECK_RET(xExec != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* gateUpExec = MaybeContiguous(gateUpWeight, uniqueExecutor.get());
    CHECK_RET(gateUpExec != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* downExec = MaybeContiguous(downWeight, uniqueExecutor.get());
    CHECK_RET(downExec != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* nativeOut =
        l0op::SwigluGatedMlp(xExec, gateUpExec, downExec, cubeMathType, uniqueExecutor.get());
    CHECK_RET(nativeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto shapedOut = uniqueExecutor->CreateView(nativeOut, y->GetViewShape(), nativeOut->GetViewOffset());
    CHECK_RET(shapedOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(shapedOut, y, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSwigluGatedMlp(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSwigluGatedMlp);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
