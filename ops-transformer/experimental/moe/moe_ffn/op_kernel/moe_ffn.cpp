/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * @file moe_ffn.cpp
 */

#include "moe_w4a16.cpp"
#include "moe_w4a16_single.cpp"

extern "C" __global__ __aicore__ void moe_ffn(GM_ADDR x, GM_ADDR quantized_weight_13, GM_ADDR weight_scale_13,
                                              GM_ADDR quantized_weight_2, GM_ADDR weight_scale_2,
                                              GM_ADDR expanded_expert_idx, GM_ADDR sorted_row_idx,
                                              GM_ADDR expanded_row_idx, GM_ADDR topk_weights, GM_ADDR weight_offset_13,
                                              GM_ADDR weight_offset_2, GM_ADDR y, GM_ADDR usrworkspace,
                                              GM_ADDR moeTiling)
{
    AscendC::SetAtomicNone();
    GET_TILING_DATA(tiling_data, moeTiling);
    const MoeFFNTilingData *__restrict tilingData = &tiling_data;
    if (TILING_KEY_IS(10000001)) {
        MOEW4A16<half> moe_handle;
        moe_handle.Init(x, quantized_weight_13, weight_scale_13, weight_offset_13, quantized_weight_2, weight_scale_2,
                        weight_offset_2, expanded_expert_idx, sorted_row_idx, expanded_row_idx, topk_weights, y,
                        usrworkspace, tilingData);
        moe_handle.Process();
    } else if (TILING_KEY_IS(10000027)) {
        MOEW4A16<bfloat16_t> moe_handle;
        moe_handle.Init(x, quantized_weight_13, weight_scale_13, weight_offset_13, quantized_weight_2, weight_scale_2,
                        weight_offset_2, expanded_expert_idx, sorted_row_idx, expanded_row_idx, topk_weights, y,
                        usrworkspace, tilingData);
        moe_handle.Process();
    } else if (TILING_KEY_IS(11000001)) {
        MOEW4A16Single<half> moe_handle;
        moe_handle.Init(x, quantized_weight_13, weight_scale_13, weight_offset_13, quantized_weight_2, weight_scale_2,
                        weight_offset_2, expanded_expert_idx, sorted_row_idx, expanded_row_idx, topk_weights, y,
                        usrworkspace, tilingData);
        moe_handle.Process();
    } else if (TILING_KEY_IS(11000027)) {
        MOEW4A16Single<bfloat16_t> moe_handle;
        moe_handle.Init(x, quantized_weight_13, weight_scale_13, weight_offset_13, quantized_weight_2, weight_scale_2,
                        weight_offset_2, expanded_expert_idx, sorted_row_idx, expanded_row_idx, topk_weights, y,
                        usrworkspace, tilingData);
        moe_handle.Process();
    }
}
