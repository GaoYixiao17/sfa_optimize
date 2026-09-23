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
 * \file quant_grouped_matmul_dequant_staged_tiling.h
 * \brief Plain tiling struct for the staged (xZN split-distribution) path.
 *
 * This is the in-kernel view of the tiling data consumed by the vendored
 * master grouped/normal/gemv/base kernel (namespace AscendC::qgmmd_staged).
 * Field names + layout mirror master's QuantGroupedMatmulDequantTilingData
 * (op_host tiling.h TILING_DATA_FIELD_DEF block) byte-for-byte so the
 * vendored kernel's `tilingData->field` accesses compile verbatim. The host
 * tiling fills this layout for the staged tiling key.
 */
#ifndef QUANT_GROUPED_MATMUL_DEQUANT_STAGED_TILING_H
#define QUANT_GROUPED_MATMUL_DEQUANT_STAGED_TILING_H

#include <cstdint>

namespace AscendC {
namespace qgmmd_staged {

// Mirrors master QuantGroupedMatmulDequantTilingData (all uint32_t except
// ubKMask). Order matches the host TILING_DATA_FIELD_DEF block exactly.
struct QuantGroupedMatmulDequantTilingData {
    uint32_t CoreNum;
    uint32_t perToken;
    uint32_t dynamicQuant;
    uint32_t smoothScale;

    uint32_t originE;
    uint32_t originM;
    uint32_t originN;
    uint32_t originK;
    uint32_t originKAligned32;
    uint32_t originKAligned512;
    uint32_t fracN;
    uint32_t fracK;
    // dynamic
    uint32_t dynamicBaseK;
    uint32_t dynamicIterK;
    uint32_t dynamicBaseKTail;
    // gemv
    uint32_t singleCoreFracN;
    uint32_t singleCoreFracNTail;
    uint32_t baseFracN;
    uint32_t baseFracK;
    uint32_t baseFracNL0C;
    uint32_t ubBaseK;
    uint32_t ubIterK;
    uint32_t ubBaseKTail;
    // normal
    uint32_t fracM;
    uint32_t tailM;
    uint32_t processXKBaseNMax;
    uint32_t processXKBaseN;
    uint32_t processXKloop;
    uint32_t processXKloopPerfracM;
    uint32_t processXKTailN;
    uint32_t MMmod;
    uint32_t MCoreNum;
    uint32_t NCoreNum;
    uint32_t singleCoreM;
    uint32_t singleCoreN;
    uint32_t singleCoreMTail;
    uint32_t singleCoreNTail;
    uint32_t baseMNum;
    uint32_t baseNNum;
    uint32_t baseKNum;
    // swift
    uint32_t swiftGEMVThreshold;
    uint32_t baseNNum_2;
    uint32_t baseKNum_2;
    uint32_t baseK;
    uint32_t baseKTail;
    uint32_t baseK_2;
    uint32_t baseKTail_2;

    uint64_t ubKMask;
    uint32_t isXScaleHalf;
};

} // namespace qgmmd_staged
} // namespace AscendC

#endif // QUANT_GROUPED_MATMUL_DEQUANT_STAGED_TILING_H
