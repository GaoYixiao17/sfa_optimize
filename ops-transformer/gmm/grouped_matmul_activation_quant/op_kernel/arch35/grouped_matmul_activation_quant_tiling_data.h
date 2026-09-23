/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GROUPED_MATMUL_ACTIVATION_QUANT_TILING_DATA_H
#define GROUPED_MATMUL_ACTIVATION_QUANT_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

namespace GroupedMatmulActivationQuant {
#pragma pack(push, 8)
struct GMMActivationQuantParams {
    uint32_t groupNum = 0;
    uint8_t groupListType = 0;
    uint8_t activationType = 0;
    uint8_t quantDtype = 0;
    uint8_t roundMode = 0;
    uint8_t reserved0 = 0;
    uint8_t scaleAlg = 0;
    uint16_t reserved = 0;
    uint32_t rowLen = 0;
    uint32_t ubAvail = 0;
    float dstTypeMax = 0.0f;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct GMMActivationQuantMMTiling {
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t baseK = 0;
    uint32_t kAL1 = 0;
    uint32_t kBL1 = 0;
    uint32_t scaleKAL1 = 0;
    uint32_t scaleKBL1 = 0;
    uint8_t isBias = 0;
    uint8_t dbL0C = 0;
    uint16_t reserved1 = 0;
    uint32_t reserved2 = 0;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct GMMActivationQuantTilingDataParams {
    GMMActivationQuantParams gmmActivationQuantParams;
    GMMActivationQuantMMTiling mmTilingData;
};
#pragma pack(pop)
} // namespace GroupedMatmulActivationQuant

#endif
