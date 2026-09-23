/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Copyright (c) 2026 联通（广东）产业互联网有限公司.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file swiglu_gated_mlp_tiling.h
 * \brief
 */

#ifndef OP_HOST_SWIGLU_GATED_MLP_TILING_H_
#define OP_HOST_SWIGLU_GATED_MLP_TILING_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "../op_kernel/swiglu_gated_mlp_stage.h"

namespace optiling {

struct SwigluGatedMlpCompileInfo {
};

constexpr uint64_t WG_MLP_STATIC_FLOAT16 = 1;
constexpr uint64_t WG_MLP_STATIC_FLOAT32 = 2;
constexpr uint64_t WG_MLP_STATIC_BF16 = 3;

constexpr uint64_t WG_MLP_DYNAMIC_FLOAT16 = 1;
constexpr uint64_t WG_MLP_DYNAMIC_FLOAT32 = 2;
constexpr uint64_t WG_MLP_DYNAMIC_BF16 = 3;

using ::WG_MLP_STAGE_MM1;
using ::WG_MLP_STAGE_SWIGLU;
using ::WG_MLP_STAGE_MM2;
using ::WG_MLP_STAGE_FUSED;
using ::WG_MLP_STAGE_DTYPE_STRIDE;

constexpr uint64_t WG_MLP_KEY_FP16_MM1 = 1101;
constexpr uint64_t WG_MLP_KEY_FP16_SWIGLU = 1102;
constexpr uint64_t WG_MLP_KEY_FP16_MM2 = 1103;
constexpr uint64_t WG_MLP_KEY_FP32_MM1 = 2101;
constexpr uint64_t WG_MLP_KEY_FP32_SWIGLU = 2102;
constexpr uint64_t WG_MLP_KEY_FP32_MM2 = 2103;
constexpr uint64_t WG_MLP_KEY_FP32_FUSED = 2104;
constexpr uint64_t WG_MLP_KEY_BF16_MM1 = 3101;
constexpr uint64_t WG_MLP_KEY_BF16_SWIGLU = 3102;
constexpr uint64_t WG_MLP_KEY_BF16_MM2 = 3103;

BEGIN_TILING_DATA_DEF(SwigluGatedMlpTilingData)
    // basic flags
    TILING_DATA_FIELD_DEF(uint32_t, dtypeTag);
    TILING_DATA_FIELD_DEF(uint32_t, stage);
    TILING_DATA_FIELD_DEF(uint32_t, isDynamicShape);
    TILING_DATA_FIELD_DEF(uint32_t, is32BAligned);
    TILING_DATA_FIELD_DEF(uint32_t, isDoubleBuffer);
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, dtypeSize);

    // global problem size
    TILING_DATA_FIELD_DEF(uint64_t, totalRows);         // flattened rows, M
    TILING_DATA_FIELD_DEF(uint64_t, hiddenSize);        // H
    TILING_DATA_FIELD_DEF(uint64_t, gateUpSize);        // 2I
    TILING_DATA_FIELD_DEF(uint64_t, intermediateSize);  // I
    TILING_DATA_FIELD_DEF(uint64_t, outSize);           // O

    // row split
    TILING_DATA_FIELD_DEF(uint32_t, baseRowsPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, tailRows);
    TILING_DATA_FIELD_DEF(uint32_t, tileRows);

    // first matmul tile
    TILING_DATA_FIELD_DEF(uint32_t, mm1BaseM);
    TILING_DATA_FIELD_DEF(uint32_t, mm1BaseN);
    TILING_DATA_FIELD_DEF(uint32_t, mm1BaseK);

    // swiglu tile
    TILING_DATA_FIELD_DEF(uint32_t, swiBaseRows);
    TILING_DATA_FIELD_DEF(uint32_t, swiBaseCols);

    // second matmul tile
    TILING_DATA_FIELD_DEF(uint32_t, mm2BaseM);
    TILING_DATA_FIELD_DEF(uint32_t, mm2BaseN);
    TILING_DATA_FIELD_DEF(uint32_t, mm2BaseK);

    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SwigluGatedMlp, SwigluGatedMlpTilingData)

}  // namespace optiling

#endif  // OP_HOST_SWIGLU_GATED_MLP_TILING_H_
