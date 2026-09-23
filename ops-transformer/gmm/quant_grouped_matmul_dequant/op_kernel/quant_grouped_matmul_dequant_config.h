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
 * \file quant_grouped_matmul_dequant_config.h
 * \brief Shared host+kernel constants and tiling candidates for qgmmd.
 *        No __aicore__ content here - safe to include from both host tiling
 *        and kernel code.
 */
#ifndef QUANT_GROUPED_MATMUL_DEQUANT_CONFIG_H
#define QUANT_GROUPED_MATMUL_DEQUANT_CONFIG_H

#include <cstdint>

namespace AscendC {

// ============================================================
// UB bank padding between adjacent persistent sub-buffers.
// 0 = no padding; widen if profiling shows UB bank conflicts.
// ============================================================
constexpr uint32_t UB_BANK_PAD = 0;

/**
 * Tiling candidate arrays (driven by L0A/L0B = 64 KB pingpong halves).
 *
 * W8A8: int8 weight on L0B, int8 X on L0A. Per pingpong half = 32 KB.
 *   Nb_max when Kb = 128 B:  32768 / 128 = 256
 *   Nb_max when Kb = 256 B:  32768 / 256 = 128
 *   Mb_max same dimensions.
 * SmallM template picks from the tail (<=8) of MB_CANDIDATES.
 */
constexpr uint32_t NB_CANDIDATES[] = {256, 192, 128, 96, 64, 32};
constexpr uint32_t MB_CANDIDATES[] = {256, 192, 128, 96, 64, 32, 16, 8, 4};
constexpr uint32_t KB_DIVISORS[] = {1, 2, 4, 8, 16};

constexpr uint32_t NB_CANDIDATE_COUNT = sizeof(NB_CANDIDATES) / sizeof(uint32_t);
constexpr uint32_t MB_CANDIDATE_COUNT = sizeof(MB_CANDIDATES) / sizeof(uint32_t);
constexpr uint32_t KB_DIVISOR_COUNT = sizeof(KB_DIVISORS) / sizeof(uint32_t);

// Threshold separating SmallM (<=8) from full MxN tiling.
constexpr uint32_t SMALL_M_THRESHOLD = 8;

// All candidates must align with M/N fractal = 16 at the coarse end.
// The tail {8, 4} is legal only on SmallM specialization (row count < 16).
static_assert(NB_CANDIDATES[0] % 16 == 0, "Nb candidates must be multiples of 16");
static_assert(NB_CANDIDATES[NB_CANDIDATE_COUNT - 1] % 16 == 0, "Nb candidates must be multiples of 16");
static_assert(MB_CANDIDATES[0] % 16 == 0, "Mb candidates must be multiples of 16");

// ============================================================
// 310P memory hierarchy (from Ascend310P3.ini)
// ============================================================
constexpr uint32_t L0A_TOTAL_BYTES = 64 * 1024;
constexpr uint32_t L0B_TOTAL_BYTES = 64 * 1024;
constexpr uint32_t L0C_TOTAL_BYTES = 256 * 1024;
constexpr uint32_t L1_TOTAL_BYTES = 1024 * 1024;
constexpr uint32_t UB_TOTAL_BYTES = 256 * 1024;
constexpr uint32_t L2_TOTAL_BYTES = 16 * 1024 * 1024;

constexpr uint32_t L0A_HALF_BYTES = L0A_TOTAL_BYTES / 2; // pingpong half
constexpr uint32_t L0B_HALF_BYTES = L0B_TOTAL_BYTES / 2;

// L2 headroom: K*N <= L2_HEADROOM_BYTES activates L2-aware tiling (weight
// stays L2-resident per expert, cb-outer swizzle).
constexpr uint32_t L2_HEADROOM_BYTES = 14 * 1024 * 1024;

// L2 weight-share percent: nBlockSize = L2_HEADROOM_BYTES * pct / 100 /
// (Nb*K bytes per cb) = cbs that fit L2 together across the 8 cores' cbs.
constexpr uint32_t L2_WEIGHT_SHARE_PCT = 70;

// ============================================================
// Bandwidth constants (bytes/cycle) - used by host-side cost model
// ============================================================
constexpr uint32_t MTE2_BW_BPC = 17;      // GM DDR
constexpr uint32_t L2_BW_BPC = 114;       // L2 cache hit
constexpr uint32_t MTE3_BW_BPC = 17;      // UB -> GM
constexpr uint32_t MTE1_L0A_BW_BPC = 512; // L1 -> L0A (feature map)
constexpr uint32_t MTE1_L0B_BW_BPC = 256; // L1 -> L0B (weight)
constexpr uint32_t L0C_UB_BW_BPC = 512;   // L0C -> UB (VDEQ16 path)

constexpr uint32_t MMAD_INT8_MACS_PCYC = 4096; // peak INT8 MACs/cycle (310P3, per Ascend Agent mmad-cycles.md)
constexpr uint32_t V_FP16_OPS_PCYC = 128;      // half elements / cycle

constexpr uint32_t DMA_PROLOG_CYC = 422; // per-DMA setup overhead

// ============================================================
// Fractal sizes (target 310P Mmad)
// ============================================================
constexpr uint32_t M_FRACTAL = 16;
constexpr uint32_t N_FRACTAL = 16;
constexpr uint32_t K_FRACTAL_INT8_BYTES = 32;
constexpr uint32_t L0C_FRACTAL_SIZE = M_FRACTAL * N_FRACTAL; // 256 elements

// ============================================================
// Mmad + blocked ZN hard limits
// ============================================================
// Largest 32-aligned inner-K Mmad can accept in a single issue.
constexpr uint32_t MMAD_K_MAX = 4064;

// Blocked ZN outer-N block size (fractals of 16). Blocks of 4080 fractals
// (= 65280 elements) keep srcStride representable in 16 bits for MTE2.
constexpr uint32_t BLOCKED_ZN_NG = 4080;
constexpr uint32_t BLOCKED_ZN_N_ELEMS = BLOCKED_ZN_NG * N_FRACTAL; // 65280

// ============================================================
// DynamicQuant math constants
// ============================================================
// x_q = round(x * 127 / max_f); pertoken = max_f / 127 for dequant.
constexpr float FLOAT_RECIP_127 = 1.0f / 127.0f;
// Lower bound on max_f to avoid division-by-zero in 127/max_f.
constexpr float DQ_MAX_EPS = 1e-7f;

// ============================================================
// Forward-compatibility tiling keys (emitted from host but all
// route to the unified kernel class today).
// ============================================================
constexpr uint32_t TILING_KEY_GEMV = 10000001;
constexpr uint32_t TILING_KEY_NORMAL = 10000002;
constexpr uint32_t TILING_KEY_GROUPED = 10000003;
constexpr uint32_t TILING_KEY_SWIFT = 10000004;

// Core count on 310P3.
constexpr uint32_t NUM_CORES_310P = 8;

} // namespace AscendC

#endif // QUANT_GROUPED_MATMUL_DEQUANT_CONFIG_H
