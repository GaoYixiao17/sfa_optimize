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
 * \file swiglu_gated_mlp_single.hpp
 * \brief
 */

#ifndef OP_KERNEL_SWIGLU_GATED_MLP_SINGLE_HPP_
#define OP_KERNEL_SWIGLU_GATED_MLP_SINGLE_HPP_

#include "kernel_operator.h"
#include "glu_tiling.hpp"
#include "swiglu_gated_mlp_impl.hpp"

using namespace AscendC;

namespace swiglu_gated_mlp_kernel {

template <class KernelImpl, typename InT, typename OutT>
class SwigluGatedMlpSingle {
public:
    __aicore__ SwigluGatedMlpSingle() = default;
    __aicore__ ~SwigluGatedMlpSingle() = default;

    __aicore__ void Init(
        GM_ADDR xGm,
        GM_ADDR gateUpWeightGm,
        GM_ADDR downWeightGm,
        GM_ADDR yGm,
        GM_ADDR workspaceGm,
        GM_ADDR tilingGm)
    {
        GET_TILING_DATA(localTiling, tilingGm);
        Init(xGm, gateUpWeightGm, downWeightGm, yGm, workspaceGm, localTiling);
    }

    __aicore__ void Init(
        GM_ADDR xGm,
        GM_ADDR gateUpWeightGm,
        GM_ADDR downWeightGm,
        GM_ADDR yGm,
        GM_ADDR workspaceGm,
        const SwigluGatedMlpTilingData &tilingData)
    {
        tilingData_ = tilingData;
        kernel_.Init(xGm, gateUpWeightGm, downWeightGm, yGm, workspaceGm, tilingData_);
    }

    __aicore__ void Process()
    {
        if (tilingData_.dtypeTag == WG_MLP_EMPTY_TILING_KEY) {
            return;
        }
        kernel_.Process();
    }

private:
    SwigluGatedMlpTilingData tilingData_ {};
    KernelImpl kernel_ {};
};

}  // namespace swiglu_gated_mlp_kernel

#endif  // OP_KERNEL_SWIGLU_GATED_MLP_SINGLE_HPP_
