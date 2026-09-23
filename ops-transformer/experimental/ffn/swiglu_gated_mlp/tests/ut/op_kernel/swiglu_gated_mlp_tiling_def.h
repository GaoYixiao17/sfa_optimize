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

#ifndef TEST_SWIGLU_GATED_MLP_TILING_DEF_H
#define TEST_SWIGLU_GATED_MLP_TILING_DEF_H

#include <cstdint>
#include <cstring>

#include "tiling/platform/platform_ascendc.h"
#include "glu_tiling.hpp"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

inline void InitSwigluGatedMlpTilingData(uint8_t *tiling, SwigluGatedMlpTilingData *tilingData)
{
    std::memcpy(tilingData, tiling, sizeof(SwigluGatedMlpTilingData));
}

#ifdef GET_TILING_DATA
#undef GET_TILING_DATA
#endif

#define GET_TILING_DATA(tilingData, tilingArg)                                      \
    SwigluGatedMlpTilingData tilingData;                                            \
    InitSwigluGatedMlpTilingData(reinterpret_cast<uint8_t *>(tilingArg), &tilingData)

#ifdef GET_TILING_DATA_WITH_STRUCT
#undef GET_TILING_DATA_WITH_STRUCT
#endif

#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg)            \
    tilingStruct tilingData;                                                        \
    InitSwigluGatedMlpTilingData(reinterpret_cast<uint8_t *>(tilingArg), &tilingData)

#endif  // TEST_SWIGLU_GATED_MLP_TILING_DEF_H
