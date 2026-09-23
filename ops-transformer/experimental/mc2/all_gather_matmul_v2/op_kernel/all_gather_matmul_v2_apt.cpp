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
 * \file all_gather_matmul_apt.cpp
 * \brief
 */

#include "lib/matmul_intf.h"
#include "common.h"
#include "all_gather_matmul_v2_apt_tiling_key.h"
#include "all_gather_matmul_fp16_bf16.h"

using namespace Mc2Tiling;
using namespace AllGatherMatmulImpl;

#define INVOKE_ALLGATHERMM_FP16_BF16_OP_IMPL(templateClass, isTransB, ...)                                             \
    do {                                                                                                               \
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, A_DTYPE, false>;                              \
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, B_DTYPE, isTransB>;                           \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, typename BiasType<BIAS_DTYPE>::type>;      \
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;                                     \
        REGISTER_TILING_DEFAULT(Mc2Tiling::AllGatherMatmulTilingData);                                                 \
        auto tiling = (__gm__ Mc2Tiling::AllGatherMatmulTilingData *)tilingGM;                                         \
        __gm__ void *mc2InitTiling = (__gm__ void *)(&(tiling->mc2InitTiling));                                        \
        __gm__ void *mc2CcTiling = (__gm__ void *)(&(tiling->mc2CcTiling));                                            \
        GET_TILING_DATA(tilingData, tilingGM);                                                                         \
        templateClass<aType, bType, biasType, cType> op;                                                               \
        op.Init(aGM, bGM, biasGM, cGM, (__gm__ uint8_t *)context, workspaceGM, gatherOut, &tilingData, mc2InitTiling,  \
                mc2CcTiling, &pipe);                                                                                   \
        op.Process();                                                                                                  \
    } while (0)

template <TPL_PARAMS_COMM, TPL_QUANT_BMM_PARAMS_COMM>
__global__ __aicore__ void all_gather_matmul_v2(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR scaleInv1,
                                                GM_ADDR scaleInv2, GM_ADDR scale, GM_ADDR cGM, GM_ADDR gatherOut,
                                                GM_ADDR amax, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    __gm__ HcclCombinOpParam *context = (__gm__ HcclCombinOpParam *)(GetHcclContext<0>());
    INVOKE_ALLGATHERMM_FP16_BF16_OP_IMPL(AllGatherMatmulFP16BF16, TRANS_B);
}
