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
 * \file masked_causal_conv1d_struct.h
 * \brief TilingData structure for MaskedCausalConv1d
 */

#ifndef MASKED_CAUSAL_CONV1D_STRUCT_H
#define MASKED_CAUSAL_CONV1D_STRUCT_H

struct MaskedCausalConv1dTilingData {
    // ===== Shape =====
    int64_t S; // token sequence length
    int64_t B; // batch size
    int64_t H; // hidden dimension

    // ===== H-dimension inter-core split =====
    int64_t hCoreCnt;         // number of cores in H direction
    int64_t hMainCnt;         // first hMainCnt cores are "main" (larger)
    int64_t hBlockFactor;     // H elements per main core
    int64_t hBlockTailFactor; // H elements per tail core

    // ===== B-dimension inter-core split =====
    int64_t bCoreCnt;         // number of cores in B direction
    int64_t bMainCnt;         // first bMainCnt cores are "main"
    int64_t bBlockFactor;     // B elements per main core
    int64_t bBlockTailFactor; // B elements per tail core

    // ===== S-dimension inter-core split =====
    int64_t sCoreCnt;         // number of cores in S direction
    int64_t sMainCnt;         // first sMainCnt cores are "main"
    int64_t sBlockFactor;     // S elements per main core
    int64_t sBlockTailFactor; // S elements per tail core

    // ===== UB tile sizes =====
    int64_t ubFactorH; // k×H_REG (variable, §6.2: starts at 64, expands when B+S fully loaded and UB<50%)
    int64_t ubFactorB; // bUb: batch tile size
    int64_t ubFactorS; // sUb: sequence tile size

    // ===== Main-core loop params =====
    int64_t loopNumH;      // H loop count for main H-core
    int64_t ubTailFactorH; // H tail tile size for main H-core
    int64_t loopNumB;      // B loop count for main B-core
    int64_t ubTailFactorB; // B tail tile for main B-core
    int64_t loopNumS;      // S loop count for main S-core
    int64_t ubTailFactorS; // S tail tile for main S-core

    // ===== Tail-core loop params =====
    int64_t tailBlockLoopNumH;
    int64_t tailBlockUbTailFactorH;
    int64_t tailBlockLoopNumB;
    int64_t tailBlockUbTailFactorB;
    int64_t tailBlockLoopNumS;
    int64_t tailBlockUbTailFactorS;

    // ===== Meta =====
    int64_t realCoreNum; // = hCoreCnt * bCoreCnt * sCoreCnt
    int64_t isMaskNone;  // 1 = mask input is None (skip mask multiply); 0 = valid mask
};

#endif // MASKED_CAUSAL_CONV1D_STRUCT_H
