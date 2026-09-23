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
 * \file swiglu_gated_mlp.cpp
 * \brief
 */

#include "glu_tiling.hpp"
#include "swiglu_gated_mlp_impl.hpp"

using namespace AscendC;

namespace {

template <typename T>
__aicore__ inline void RunCubeMatmulStage(
    GM_ADDR xGm,
    GM_ADDR weightGm,
    GM_ADDR yGm,
    const SwigluGatedMlpTilingData &tilingData)
{
    swiglu_gated_mlp_kernel::SwigluGatedMlpCubeMatmulStage<T> op;
    op.Init(xGm, weightGm, yGm, tilingData);
    op.Process();
}

template <typename T, int BUFFER_NUM>
__aicore__ inline void RunSwigluStage(GM_ADDR gateUpGm, GM_ADDR hiddenGm, const SwigluGatedMlpTilingData &tilingData)
{
    swiglu_gated_mlp_kernel::SwigluGatedMlpSwigluStage<T, BUFFER_NUM> op;
    op.Init(gateUpGm, hiddenGm, tilingData);
    op.Process();
}

template <typename T>
__aicore__ inline void RunScalarSwigluStage(
    GM_ADDR gateUpGm,
    GM_ADDR hiddenGm,
    const SwigluGatedMlpTilingData &tilingData)
{
    swiglu_gated_mlp_kernel::SwigluGatedMlpScalarSwigluStage<T> op;
    op.Init(gateUpGm, hiddenGm, tilingData);
    op.Process();
}

template <typename T>
__aicore__ inline void RunStage(
    GM_ADDR xGm,
    GM_ADDR gateUpWeightGm,
    GM_ADDR downWeightGm,
    GM_ADDR yGm,
    GM_ADDR workspaceGm,
    const SwigluGatedMlpTilingData &tilingData)
{
    if (tilingData.stage == WG_MLP_STAGE_FUSED) {
        if ASCEND_IS_AIC {
            swiglu_gated_mlp_kernel::SwigluGatedMlpFusedFloatStage op;
            op.Init(xGm, gateUpWeightGm, downWeightGm, yGm, workspaceGm, tilingData);
            op.Process();
        }
        return;
    }

    if (tilingData.stage == WG_MLP_STAGE_MM1 || tilingData.stage == WG_MLP_STAGE_MM2) {
        if ASCEND_IS_AIC {
            RunCubeMatmulStage<T>(xGm, gateUpWeightGm, yGm, tilingData);
        }
        return;
    }

    if ASCEND_IS_AIV {
        if (tilingData.isDoubleBuffer != 0U) {
            RunSwigluStage<T, 2>(xGm, yGm, tilingData);
        } else {
            RunSwigluStage<T, 1>(xGm, yGm, tilingData);
        }
    }
}

}  // namespace

extern "C" __global__ __aicore__ void swiglu_gated_mlp(
    GM_ADDR x_gm,
    GM_ADDR gate_up_weight_gm,
    GM_ADDR down_weight_gm,
    GM_ADDR y_gm,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SwigluGatedMlpTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    KERNEL_TASK_TYPE(100, KERNEL_TYPE_AIC_ONLY);
    KERNEL_TASK_TYPE(1101, KERNEL_TYPE_AIC_ONLY);
    KERNEL_TASK_TYPE(1102, KERNEL_TYPE_AIV_ONLY);
    KERNEL_TASK_TYPE(1103, KERNEL_TYPE_AIC_ONLY);
    KERNEL_TASK_TYPE(2101, KERNEL_TYPE_AIC_ONLY);
    KERNEL_TASK_TYPE(2102, KERNEL_TYPE_AIV_ONLY);
    KERNEL_TASK_TYPE(2103, KERNEL_TYPE_AIC_ONLY);
    KERNEL_TASK_TYPE(2104, KERNEL_TYPE_AIC_ONLY);
    KERNEL_TASK_TYPE(3101, KERNEL_TYPE_AIC_ONLY);
    KERNEL_TASK_TYPE(3102, KERNEL_TYPE_AIV_ONLY);
    KERNEL_TASK_TYPE(3103, KERNEL_TYPE_AIC_ONLY);

    if (TILING_KEY_IS(WG_MLP_EMPTY_TILING_KEY)) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(WG_MLP_STATIC_FLOAT16) || TILING_KEY_IS(WG_MLP_DYNAMIC_FLOAT16) ||
        TILING_KEY_IS(WG_MLP_KEY_FP16_MM1) || TILING_KEY_IS(WG_MLP_KEY_FP16_SWIGLU) ||
        TILING_KEY_IS(WG_MLP_KEY_FP16_MM2)) {
        RunStage<half>(x_gm, gate_up_weight_gm, down_weight_gm, y_gm, workspace, tilingData);
    } else if (TILING_KEY_IS(WG_MLP_STATIC_FLOAT32) || TILING_KEY_IS(WG_MLP_DYNAMIC_FLOAT32) ||
        TILING_KEY_IS(WG_MLP_KEY_FP32_MM1) || TILING_KEY_IS(WG_MLP_KEY_FP32_SWIGLU) ||
        TILING_KEY_IS(WG_MLP_KEY_FP32_MM2) || TILING_KEY_IS(WG_MLP_KEY_FP32_FUSED)) {
        RunStage<float>(x_gm, gate_up_weight_gm, down_weight_gm, y_gm, workspace, tilingData);
    } else if (TILING_KEY_IS(WG_MLP_STATIC_BF16) || TILING_KEY_IS(WG_MLP_DYNAMIC_BF16) ||
        TILING_KEY_IS(WG_MLP_KEY_BF16_MM1) || TILING_KEY_IS(WG_MLP_KEY_BF16_SWIGLU) ||
        TILING_KEY_IS(WG_MLP_KEY_BF16_MM2)) {
        RunStage<bfloat16_t>(x_gm, gate_up_weight_gm, down_weight_gm, y_gm, workspace, tilingData);
    }
}
