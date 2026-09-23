/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "matmul_swiglu_tiling_def.h"
#include "matmul_swiglu_impl.hpp"
using namespace AscendC;

// DTYPE_X 由编译框架按 x 的数据类型注入(fp16->half, bf16->bfloat16_t, fp32->float),
// 各 dtype 编译为独立 kernel 二进制, 无需运行时按 TilingKey 分发。
// 未注入时给出默认值以保证可编译(如 op_kernel UT 编译, 不实际执行)。
#ifndef DTYPE_X
#define DTYPE_X half
#endif

extern "C" __global__ __aicore__ void matmul_swiglu(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
                                                       GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR userWs = GetUserWorkspace(workspace);  // 用户 workspace: 存 [M,2N] fp32 中间结果
    if (userWs == nullptr) {
        return;
    }
    GET_TILING_DATA_WITH_STRUCT(optiling::MatmulSwigluTilingData, tilingDataIn, tiling);
    const optiling::MatmulSwigluTilingData* __restrict__ tilingData = &tilingDataIn;
    // 数据类型由编译期 DTYPE_X 决定; cube 累加固定 fp32。方案一: 单 matmul + 向量 SwiGLU。
    // transpose_weight 经 TilingKey 选择编译期模板分支(0=非转置, 1=转置), 避免运行时判断。
    // 注: 每个 key 需显式 TILING_KEY_IS(编译期据此为各 key 登记 kernel 函数入口), 不能用 else 兜底。
    if (TILING_KEY_IS(0)) {
        NsMatmulSwiglu::MatmulSwigluImpl<DTYPE_X, float, false> op;
        REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.mm, &tilingData->mmTiling);
        op.Init(x, weight, bias, y, userWs, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        NsMatmulSwiglu::MatmulSwigluImpl<DTYPE_X, float, true> op;
        REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.mm, &tilingData->mmTiling);
        op.Init(x, weight, bias, y, userWs, tilingData);
        op.Process();
    }
}
