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
 * \file grouped_matmul_finalize_routing_tiling_data.h
 * \brief
 */
#ifndef GROUPED_MATMUL_FINALIZE_ROUTING_WEIGHT_QUANT_TILING_DATA_H
#define GROUPED_MATMUL_FINALIZE_ROUTING_WEIGHT_QUANT_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

namespace GMMFinalizeRoutingArch35Tiling {
#pragma pack(push, 8)
struct GMMFinalizeRoutingWeightQuantTilingData {
    uint8_t groupListType = 0;
    uint8_t hasBias = 0;
    uint16_t reserve1 = 0;

    uint32_t groupNum = 0;
    uint32_t coreNum = 0;
    uint32_t outputBs = 0;
    uint32_t sharedInputOffset = 0;
    uint32_t sharedInputLen = 0;
    float sharedInputWeight = 0;
    uint32_t groupSize = 0;

    uint64_t kSize = 0;
    uint64_t nSize = 0;
    uint64_t initSize = 0;
};
#pragma pack(pop)
} // namespace GMMFinalizeRoutingArch35Tiling
#endif // GROUPED_MATMUL_FINALIZE_ROUTING_WEIGHT_QUANT_TILING_DATA_H
