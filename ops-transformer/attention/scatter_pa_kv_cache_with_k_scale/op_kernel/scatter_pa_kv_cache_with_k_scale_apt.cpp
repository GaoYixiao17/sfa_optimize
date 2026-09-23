/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "arch35/scatter_pa_kv_cache_with_k_scale_simt.h"
#include "arch35/scatter_pa_kv_cache_with_k_scale_tiling_key.h"

/**
 * @brief scatter_pa_kv_cache_with_k_scale 算子 kernel 入口
 *
 * schMode 由 tiling key 决定，DTYPE_KEY / DTYPE_SLOT_MAPPING 由框架根据 def.cpp 自动生成。
 * 本算子为 inplace 算子（key_cache/value_cache/key_scale_cache 输入输出同名），
 * 输出参数添加 _out 后缀。
 */
template <uint32_t schMode>
__global__ __aicore__ void scatter_pa_kv_cache_with_k_scale(
    GM_ADDR key, GM_ADDR value,
    GM_ADDR key_cache, GM_ADDR value_cache,
    GM_ADDR slot_mapping, GM_ADDR key_scale, GM_ADDR key_scale_cache,
    GM_ADDR key_cache_out, GM_ADDR value_cache_out, GM_ADDR key_scale_cache_out,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ScatterPaKvCacheWithKScaleTilingData);
    GET_TILING_DATA_WITH_STRUCT(ScatterPaKvCacheWithKScaleTilingData, tilingData, tiling);

    // 两种场景模式（特化/泛化）使用统一的 Process 函数
    // dtype 由 DTYPE_KEY / DTYPE_SLOT_MAPPING 宏自动实例化
    NsScatterPaKvCacheWithKScale::Process<DTYPE_KEY, DTYPE_SLOT_MAPPING, schMode>(
        key, value, key_cache_out, value_cache_out,
        slot_mapping, key_scale, key_scale_cache_out,
        &tilingData);
}
