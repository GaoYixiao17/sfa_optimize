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
 * \file matmul_swiglu_tiling_def.h
 * \brief MatmulSwiglu kernel 侧 TilingData 结构定义
 */

#ifndef OPS_OP_KERNEL_MATMUL_SWIGLU_TILING_DEF_H_
#define OPS_OP_KERNEL_MATMUL_SWIGLU_TILING_DEF_H_
#include "adv_api/kernel_tiling.h"

namespace optiling {
struct MatmulSwigluTilingData {
    uint32_t hasBias;
    uint32_t tileN;  // 向量 SwiGLU 行内分块列数 (满足 UB 预算)
    TCubeTiling mmTiling;  // m/k/2N 从此派生: m=mmTiling.M, k=mmTiling.Ka, 2N=mmTiling.N
};
}  // namespace optiling
#endif
