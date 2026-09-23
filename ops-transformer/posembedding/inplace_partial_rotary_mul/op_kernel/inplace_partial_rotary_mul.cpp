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
 * \file inplace_partial_rotary_mul_aba.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "inplace_partial_rotary_mul.h"
#include "inplace_rotate_interleaved_split_s.h"
#include "inplace_rotate_interleaved_split_bs.h"
#include "inplace_rotate_interleaved_split_bsn.h"
#include "inplace_rotate_interleaved_split_s_pad.h"
#include "inplace_rotate_interleaved_split_bs_pad.h"
#include "inplace_rotate_interleaved_split_bsn_pad.h"
#include "inplace_rotate_interleaved_split_s_mixed.h"
#include "inplace_rotate_interleaved_split_bs_mixed.h"
#include "inplace_rotate_interleaved_split_bsn_mixed.h"
#include "inplace_rotate_interleaved_split_s_pad_mixed.h"
#include "inplace_rotate_interleaved_split_bs_pad_mixed.h"
#include "inplace_rotate_interleaved_split_bsn_pad_mixed.h"

#define TILING_KEY1 1
#define TILING_KEY2 2

using namespace AscendC;
using namespace RotateInterleavedN;

extern "C" __global__ __aicore__ void inplace_partial_rotary_mul(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y,
                                                                 GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(InplacePartialRopeRegbaseTilingData, tilingData, tiling);
    const InplacePartialRopeRegbaseTilingData* __restrict__ tilingData1 = &tilingData;
    // No-op: empty slice (sliceStart==sliceEnd), return directly
    if (tilingData1->headDim == 0) {
        return;
    }
    if (TILING_KEY_IS(TILING_KEY1)) {
        InplacePartialRotaryMul::InplacePartialRotaryMulABA<DTYPE_X, true> op;
        op.Init(x, cos, sin, y, workspace, tilingData1, &pipe);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(TILING_KEY2)) {
        InplacePartialRotaryMul::InplacePartialRotaryMulABA<DTYPE_X, false> op;
        op.Init(x, cos, sin, y, workspace, tilingData1, &pipe);
        op.Process();
        return;
    }
    // mode: rotate_interleaved
    if (TILING_KEY_IS(2000)) {
        InterleavedSplitS<half> interleavedSplitS;
        interleavedSplitS.Init(x, cos, sin, y, tilingData1, &pipe);
        interleavedSplitS.Process();
    } else if (TILING_KEY_IS(2010)) {
        InterleavedSplitS<bfloat16_t> interleavedSplitS;
        interleavedSplitS.Init(x, cos, sin, y, tilingData1, &pipe);
        interleavedSplitS.Process();
    } else if (TILING_KEY_IS(2020)) {
        InterleavedSplitS<float> interleavedSplitS;
        interleavedSplitS.Init(x, cos, sin, y, tilingData1, &pipe);
        interleavedSplitS.Process();
    } else if (TILING_KEY_IS(2030)) {
        InterleavedSplitSMixed<bfloat16_t> interleavedSplitSMixed;
        interleavedSplitSMixed.Init(x, cos, sin, y, tilingData1, &pipe);
        interleavedSplitSMixed.Process();
    } else if (TILING_KEY_IS(2040)) {
        InterleavedSplitSMixed<half> interleavedSplitSMixed;
        interleavedSplitSMixed.Init(x, cos, sin, y, tilingData1, &pipe);
        interleavedSplitSMixed.Process();
    } else if (TILING_KEY_IS(2100)) {
        InterleavedSplitBS<half> interleavedSplitBS;
        interleavedSplitBS.Init(x, cos, sin, y, tilingData1, &pipe);
        interleavedSplitBS.Process();
    } else if (TILING_KEY_IS(2110)) {
        InterleavedSplitBS<bfloat16_t> interleavedSplitBS;
        interleavedSplitBS.Init(x, cos, sin, y, tilingData1, &pipe);
        interleavedSplitBS.Process();
    } else if (TILING_KEY_IS(2120)) {
        InterleavedSplitBS<float> interleavedSplitBS;
        interleavedSplitBS.Init(x, cos, sin, y, tilingData1, &pipe);
        interleavedSplitBS.Process();
    } else if (TILING_KEY_IS(2130)) {
        InterleavedSplitBSMixed<bfloat16_t> interleavedSplitBSMixed;
        interleavedSplitBSMixed.Init(x, cos, sin, y, tilingData1, &pipe);
        interleavedSplitBSMixed.Process();
    } else if (TILING_KEY_IS(2140)) {
        InterleavedSplitBSMixed<half> interleavedSplitBSMixed;
        interleavedSplitBSMixed.Init(x, cos, sin, y, tilingData1, &pipe);
        interleavedSplitBSMixed.Process();
    } else if (TILING_KEY_IS(2200)) {
        InterleavedSplitBSN<half> interleavedSplitBSN;
        interleavedSplitBSN.Init(x, cos, sin, y, tilingData1, &pipe);
        interleavedSplitBSN.Process();
    } else if (TILING_KEY_IS(2210)) {
        InterleavedSplitBSN<bfloat16_t> interleavedSplitBSN;
        interleavedSplitBSN.Init(x, cos, sin, y, tilingData1, &pipe);
        interleavedSplitBSN.Process();
    } else if (TILING_KEY_IS(2220)) {
        InterleavedSplitBSN<float> interleavedSplitBSN;
        interleavedSplitBSN.Init(x, cos, sin, y, tilingData1, &pipe);
        interleavedSplitBSN.Process();
    } else if (TILING_KEY_IS(2230)) {
        InterleavedSplitBSNMixed<bfloat16_t> interleavedSplitBSNMixed;
        interleavedSplitBSNMixed.Init(x, cos, sin, y, tilingData1, &pipe);
        interleavedSplitBSNMixed.Process();
    } else if (TILING_KEY_IS(2240)) {
        InterleavedSplitBSNMixed<half> interleavedSplitBSNMixed;
        interleavedSplitBSNMixed.Init(x, cos, sin, y, tilingData1, &pipe);
        interleavedSplitBSNMixed.Process();
    } else if (TILING_KEY_IS(2001)) {
        InterleavedSplitSPad<half> interleavedSplitSPad;
        interleavedSplitSPad.Init(x, cos, sin, y, tilingData1, &pipe);
        interleavedSplitSPad.Process();
    } else if (TILING_KEY_IS(2011)) {
        InterleavedSplitSPad<bfloat16_t> interleavedSplitSPad;
        interleavedSplitSPad.Init(x, cos, sin, y, tilingData1, &pipe);
        interleavedSplitSPad.Process();
    } else if (TILING_KEY_IS(2021)) {
        InterleavedSplitSPad<float> interleavedSplitSPad;
        interleavedSplitSPad.Init(x, cos, sin, y, tilingData1, &pipe);
        interleavedSplitSPad.Process();
    } else if (TILING_KEY_IS(2031)) {
        InterleavedSplitSPadMixed<bfloat16_t> interleavedSplitSPadMixed;
        interleavedSplitSPadMixed.Init(x, cos, sin, y, tilingData1, &pipe);
        interleavedSplitSPadMixed.Process();
    } else if (TILING_KEY_IS(2041)) {
        InterleavedSplitSPadMixed<half> interleavedSplitSPadMixed;
        interleavedSplitSPadMixed.Init(x, cos, sin, y, tilingData1, &pipe);
        interleavedSplitSPadMixed.Process();
    } else if (TILING_KEY_IS(2101)) {
        InterleavedSplitBSPad<half> interleavedSplitBSPad;
        interleavedSplitBSPad.Init(x, cos, sin, y, tilingData1, &pipe);
        interleavedSplitBSPad.Process();
    } else if (TILING_KEY_IS(2111)) {
        InterleavedSplitBSPad<bfloat16_t> interleavedSplitBSPad;
        interleavedSplitBSPad.Init(x, cos, sin, y, tilingData1, &pipe);
        interleavedSplitBSPad.Process();
    } else if (TILING_KEY_IS(2121)) {
        InterleavedSplitBSPad<float> interleavedSplitBSPad;
        interleavedSplitBSPad.Init(x, cos, sin, y, tilingData1, &pipe);
        interleavedSplitBSPad.Process();
    } else if (TILING_KEY_IS(2131)) {
        InterleavedSplitBSPadMixed<bfloat16_t> interleavedSplitBSPadMixed;
        interleavedSplitBSPadMixed.Init(x, cos, sin, y, tilingData1, &pipe);
        interleavedSplitBSPadMixed.Process();
    } else if (TILING_KEY_IS(2141)) {
        InterleavedSplitBSPadMixed<half> interleavedSplitBSPadMixed;
        interleavedSplitBSPadMixed.Init(x, cos, sin, y, tilingData1, &pipe);
        interleavedSplitBSPadMixed.Process();
    } else if (TILING_KEY_IS(2201)) {
        InterleavedSplitBSNPad<half> interleavedSplitBSNPad;
        interleavedSplitBSNPad.Init(x, cos, sin, y, tilingData1, &pipe);
        interleavedSplitBSNPad.Process();
    } else if (TILING_KEY_IS(2211)) {
        InterleavedSplitBSNPad<bfloat16_t> interleavedSplitBSNPad;
        interleavedSplitBSNPad.Init(x, cos, sin, y, tilingData1, &pipe);
        interleavedSplitBSNPad.Process();
    } else if (TILING_KEY_IS(2221)) {
        InterleavedSplitBSNPad<float> interleavedSplitBSNPad;
        interleavedSplitBSNPad.Init(x, cos, sin, y, tilingData1, &pipe);
        interleavedSplitBSNPad.Process();
    } else if (TILING_KEY_IS(2231)) {
        InterleavedSplitBSNPadMixed<bfloat16_t> interleavedSplitBSNPadMixed;
        interleavedSplitBSNPadMixed.Init(x, cos, sin, y, tilingData1, &pipe);
        interleavedSplitBSNPadMixed.Process();
    } else if (TILING_KEY_IS(2241)) {
        InterleavedSplitBSNPadMixed<half> interleavedSplitBSNPadMixed;
        interleavedSplitBSNPadMixed.Init(x, cos, sin, y, tilingData1, &pipe);
        interleavedSplitBSNPadMixed.Process();
    }
}
