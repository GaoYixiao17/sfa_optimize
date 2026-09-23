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
 * \file matmul_allto_all_tiling_key.h
 * \brief
 */

#ifndef __OP_KERNEL_MATMUL_ALLTO_ALL_TILING_KEY_H__
#define __OP_KERNEL_MATMUL_ALLTO_ALL_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"
#include "./matmul_allto_all_tiling.h"
#include "./matmul_allto_all_tiling_data_910_93.h"

#define TILINGKEY_TPL_FP16 0
#define TILINGKEY_TPL_FP32 1
#define TILINGKEY_TPL_BF16 2
#define TILINGKEY_TPL_NOQUANT 0

// 对A2A3版本用模板参数区分
#define SOC_ASCEND910B 0
#define SOC_ASCEND910_93 1

namespace matmul_allto_all_910b_tiling_key {
// 模板参数
ASCENDC_TPL_ARGS_DECL(MatmulAlltoAll, // 算子OpType
                      ASCENDC_TPL_BOOL_DECL(MM_ALLTO_ALL_TRANS_X2, 0, 1),
                      ASCENDC_TPL_BOOL_DECL(MM_ALLTO_ALL_HAS_BIAS, 0, 1),
                      ASCENDC_TPL_UINT_DECL(MM_ALLTO_ALL_BIAS_DTYPE, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST,
                                            TILINGKEY_TPL_FP16, TILINGKEY_TPL_FP32, TILINGKEY_TPL_BF16),
                      ASCENDC_TPL_UINT_DECL(MM_ALLTO_ALL_SOC_VERSION, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST,
                                            SOC_ASCEND910B, SOC_ASCEND910_93), );

// 模板参数组合
// 用于调用GET_TPL_TILING_KEY获取TilingKey时，接口内部校验TilingKey是否合法
ASCENDC_TPL_SEL(
    // A2的模板参数校验
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(MM_ALLTO_ALL_TRANS_X2, 0, 1),
                         ASCENDC_TPL_BOOL_SEL(MM_ALLTO_ALL_HAS_BIAS, 0, 1),
                         ASCENDC_TPL_UINT_SEL(MM_ALLTO_ALL_BIAS_DTYPE, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_FP16,
                                              TILINGKEY_TPL_FP32),
                         ASCENDC_TPL_UINT_SEL(MM_ALLTO_ALL_SOC_VERSION, ASCENDC_TPL_UI_LIST, SOC_ASCEND910B),
                         ASCENDC_TPL_TILING_STRUCT_SEL(MatmulAlltoAllTilingData)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(MM_ALLTO_ALL_TRANS_X2, 0, 1),
                         ASCENDC_TPL_BOOL_SEL(MM_ALLTO_ALL_HAS_BIAS, 1),
                         ASCENDC_TPL_UINT_SEL(MM_ALLTO_ALL_BIAS_DTYPE, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_BF16),
                         ASCENDC_TPL_UINT_SEL(MM_ALLTO_ALL_SOC_VERSION, ASCENDC_TPL_UI_LIST, SOC_ASCEND910B),
                         ASCENDC_TPL_TILING_STRUCT_SEL(MatmulAlltoAllTilingData)),
    // A3的模板参数校验
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(MM_ALLTO_ALL_TRANS_X2, 0, 1),
                         ASCENDC_TPL_BOOL_SEL(MM_ALLTO_ALL_HAS_BIAS, 0, 1),
                         ASCENDC_TPL_UINT_SEL(MM_ALLTO_ALL_BIAS_DTYPE, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_FP16,
                                              TILINGKEY_TPL_FP32),
                         ASCENDC_TPL_UINT_SEL(MM_ALLTO_ALL_SOC_VERSION, ASCENDC_TPL_UI_LIST, SOC_ASCEND910_93),
                         ASCENDC_TPL_TILING_STRUCT_SEL(MatmulAlltoAllTilingDataA3)), );
} // namespace matmul_allto_all_910b_tiling_key
#endif // __OP_KERNEL_MATMUL_ALLTO_ALL_TILING_KEY_H__
