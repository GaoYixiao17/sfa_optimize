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
 * \file quant_grouped_matmul_dequant_tiling_key.h
 * \brief Template-parameter packing for the unified qgmmd kernel.
 *
 * Host tiling sets one bit per template slot in TilingData::flags; the
 * kernel entry unpacks them and dispatches to the pruned instantiation.
 * Flag layout kept in sync with `QgmmdFlagBits` on the host side.
 */
#ifndef QUANT_GROUPED_MATMUL_DEQUANT_TILING_KEY_H
#define QUANT_GROUPED_MATMUL_DEQUANT_TILING_KEY_H

#include <cstdint>

namespace AscendC {

// Bit positions inside TilingData::flags.
// Any change here must be mirrored in host tiling EmitTilingData.
enum QgmmdFlagBit : uint32_t {
    FLAG_BIT_DYNAMIC_QUANT = 0,      // 0 = static-quant (x_scale path), 1 = dynamic
    FLAG_BIT_HAS_SMOOTH = 1,         // smooth_scale input present
    FLAG_BIT_PER_TOKEN = 2,          // per-token quant (vs pertensor)
    FLAG_BIT_SCALE_IS_INT64 = 3,     // weight_scale pre-encoded uint64 vs fp32
    FLAG_BIT_BLOCKED_ZN = 4,         // weight storage = 6-D blocked ZN
    FLAG_BIT_SMALL_M = 5,            // Mb <= SMALL_M_THRESHOLD
    FLAG_BIT_L2_WEIGHT_FITS = 6,     // K*N <= L2_HEADROOM_BYTES
    FLAG_BIT_HAS_X_RAW_L1 = 7,       // xRawL1 cross-iter pre-load fits L1
    FLAG_BIT_USE_L2_HINT = 8,        // SetL2CacheHint runtime call (S16-pre gated)
    FLAG_BIT_DBL0C = 9,              // Y.2: L0C ping-pong across cbs
    FLAG_BIT_USE_XZN_WORKSPACE = 10, // xZN GM workspace stage: PhaseX writes quantized x to GM,
                                     //   PhaseMM reads from GM (replaces cross-cb xL1 reuse).
};

__aicore__ inline bool GetFlag(uint32_t flags, QgmmdFlagBit bit)
{
    return ((flags >> static_cast<uint32_t>(bit)) & 0x1u) != 0u;
}

// Packs all template-dispatch bits — used by the host to build flags.
struct QgmmdTilingKey {
    bool hasPertoken;
    bool hasSmoothScale;
    bool weightScaleIsInt64;
    bool blockedZN;
    bool smallM;

    constexpr uint32_t Pack() const
    {
        return (static_cast<uint32_t>(hasPertoken) << FLAG_BIT_PER_TOKEN) |
               (static_cast<uint32_t>(hasSmoothScale) << FLAG_BIT_HAS_SMOOTH) |
               (static_cast<uint32_t>(weightScaleIsInt64) << FLAG_BIT_SCALE_IS_INT64) |
               (static_cast<uint32_t>(blockedZN) << FLAG_BIT_BLOCKED_ZN) |
               (static_cast<uint32_t>(smallM) << FLAG_BIT_SMALL_M);
    }
};

} // namespace AscendC

#endif // QUANT_GROUPED_MATMUL_DEQUANT_TILING_KEY_H
