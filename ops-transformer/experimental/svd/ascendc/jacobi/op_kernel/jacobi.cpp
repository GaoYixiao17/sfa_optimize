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
 * \file jacobi.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "jacobi_base.h"
#include "lib/matmul_intf.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void jacobi(__gm__ uint8_t * a,
                                             __gm__ uint8_t * u, __gm__ uint8_t * s, __gm__ uint8_t * v,
                                             __gm__ uint8_t * workspace, __gm__ uint8_t * tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (tilingData.vecInstructionSize == 32) {
        SVD::JacobiBase<float, float, float, float, 32> op(&pipe);
        op.Init(a, s, u, v, user, &tilingData);
        op.Process();
    } else if (tilingData.vecInstructionSize == 64) {
        SVD::JacobiBase<float, float, float, float, 64> op(&pipe);
        op.Init(a, s, u, v, user, &tilingData);
        op.Process();
    }
}
