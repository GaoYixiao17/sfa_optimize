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
 * \file glu_tiling.hpp
 * \brief
 */

#ifndef OP_KERNEL_GLU_TILING_HPP_
#define OP_KERNEL_GLU_TILING_HPP_

#include <cstdint>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "swiglu_gated_mlp_stage.h"

using namespace AscendC;

#define WG_MLP_STATIC_FLOAT16    1
#define WG_MLP_STATIC_FLOAT32    2
#define WG_MLP_STATIC_BF16       3

#define WG_MLP_DYNAMIC_FLOAT16   1
#define WG_MLP_DYNAMIC_FLOAT32   2
#define WG_MLP_DYNAMIC_BF16      3

#define WG_MLP_EMPTY_TILING_KEY  100
#define WG_MLP_ALIGN_UNIT        16U

#define WG_MLP_KEY_FP16_MM1      1101
#define WG_MLP_KEY_FP16_SWIGLU   1102
#define WG_MLP_KEY_FP16_MM2      1103
#define WG_MLP_KEY_FP32_MM1      2101
#define WG_MLP_KEY_FP32_SWIGLU   2102
#define WG_MLP_KEY_FP32_MM2      2103
#define WG_MLP_KEY_FP32_FUSED    2104
#define WG_MLP_KEY_BF16_MM1      3101
#define WG_MLP_KEY_BF16_SWIGLU   3102
#define WG_MLP_KEY_BF16_MM2      3103

namespace swiglu_gated_mlp_kernel {

#pragma pack(push, 8)
struct alignas(8) SwigluGatedMlpTilingData {
    uint32_t dtypeTag;
    uint32_t stage;
    uint32_t isDynamicShape;
    uint32_t is32BAligned;
    uint32_t isDoubleBuffer;
    uint32_t usedCoreNum;
    uint32_t dtypeSize;

    uint64_t totalRows;
    uint64_t hiddenSize;
    uint64_t gateUpSize;
    uint64_t intermediateSize;
    uint64_t outSize;

    uint32_t baseRowsPerCore;
    uint32_t tailRows;
    uint32_t tileRows;

    uint32_t mm1BaseM;
    uint32_t mm1BaseN;
    uint32_t mm1BaseK;

    uint32_t swiBaseRows;
    uint32_t swiBaseCols;

    uint32_t mm2BaseM;
    uint32_t mm2BaseN;
    uint32_t mm2BaseK;

    TCubeTiling matmulTiling;
};
#pragma pack(pop)

template <typename T>
__aicore__ inline T DivCeil(T a, T b)
{
    return (b == 0) ? 0 : ((a + b - 1) / b);
}

template <typename T>
__aicore__ inline T AlignUp(T a, T b)
{
    return DivCeil(a, b) * b;
}

template <typename T>
__aicore__ inline T AlignDown(T a, T b)
{
    return (b == 0 || a < b) ? 0 : (a / b) * b;
}

struct GluTilingInfo {
    __aicore__ GluTilingInfo() = default;

    __aicore__ explicit GluTilingInfo(const SwigluGatedMlpTilingData &tilingData)
    {
        Parse(tilingData);
    }

    __aicore__ void Parse(const SwigluGatedMlpTilingData &tilingData)
    {
        dtypeTag = tilingData.dtypeTag;
        stage = tilingData.stage;
        isDynamicShape = tilingData.isDynamicShape;
        is32BAligned = tilingData.is32BAligned;
        isDoubleBuffer = tilingData.isDoubleBuffer;
        usedCoreNum = tilingData.usedCoreNum;
        dtypeSize = tilingData.dtypeSize;

        totalRows = tilingData.totalRows;
        hiddenSize = tilingData.hiddenSize;
        gateUpSize = tilingData.gateUpSize;
        intermediateSize = tilingData.intermediateSize;
        outSize = tilingData.outSize;

        baseRowsPerCore = tilingData.baseRowsPerCore;
        tailRows = tilingData.tailRows;
        tileRows = tilingData.tileRows;

        mm1BaseM = tilingData.mm1BaseM;
        mm1BaseN = tilingData.mm1BaseN;
        mm1BaseK = tilingData.mm1BaseK;

        swiBaseRows = tilingData.swiBaseRows;
        swiBaseCols = tilingData.swiBaseCols;

        mm2BaseM = tilingData.mm2BaseM;
        mm2BaseN = tilingData.mm2BaseN;
        mm2BaseK = tilingData.mm2BaseK;
    }

    __aicore__ bool IsEmpty() const
    {
        return dtypeTag == WG_MLP_EMPTY_TILING_KEY;
    }

    __aicore__ bool IsFp16() const
    {
        return dtypeTag == WG_MLP_STATIC_FLOAT16 || dtypeTag == WG_MLP_DYNAMIC_FLOAT16 ||
               dtypeTag == WG_MLP_KEY_FP16_MM1 || dtypeTag == WG_MLP_KEY_FP16_SWIGLU ||
               dtypeTag == WG_MLP_KEY_FP16_MM2;
    }

    __aicore__ bool IsFp32() const
    {
        return dtypeTag == WG_MLP_STATIC_FLOAT32 || dtypeTag == WG_MLP_DYNAMIC_FLOAT32 ||
               dtypeTag == WG_MLP_KEY_FP32_MM1 || dtypeTag == WG_MLP_KEY_FP32_SWIGLU ||
               dtypeTag == WG_MLP_KEY_FP32_MM2 || dtypeTag == WG_MLP_KEY_FP32_FUSED;
    }

    __aicore__ bool IsBf16() const
    {
        return dtypeTag == WG_MLP_STATIC_BF16 || dtypeTag == WG_MLP_DYNAMIC_BF16 ||
               dtypeTag == WG_MLP_KEY_BF16_MM1 || dtypeTag == WG_MLP_KEY_BF16_SWIGLU ||
               dtypeTag == WG_MLP_KEY_BF16_MM2;
    }

    __aicore__ uint32_t GetRowsForCore(uint32_t blockIdx) const
    {
        if (usedCoreNum == 0 || blockIdx >= usedCoreNum) {
            return 0;
        }
        return baseRowsPerCore + ((blockIdx < tailRows) ? 1U : 0U);
    }

    __aicore__ uint64_t GetRowStartForCore(uint32_t blockIdx) const
    {
        if (usedCoreNum == 0 || blockIdx >= usedCoreNum) {
            return 0;
        }
        const uint64_t normalPart =
            static_cast<uint64_t>(blockIdx) * static_cast<uint64_t>(baseRowsPerCore);
        const uint64_t extraPart =
            static_cast<uint64_t>((blockIdx < tailRows) ? blockIdx : tailRows);
        return normalPart + extraPart;
    }

    uint32_t dtypeTag = WG_MLP_EMPTY_TILING_KEY;
    uint32_t stage = 0;
    uint32_t isDynamicShape = 0;
    uint32_t is32BAligned = 0;
    uint32_t isDoubleBuffer = 0;
    uint32_t usedCoreNum = 0;
    uint32_t dtypeSize = 0;

    uint64_t totalRows = 0;
    uint64_t hiddenSize = 0;
    uint64_t gateUpSize = 0;
    uint64_t intermediateSize = 0;
    uint64_t outSize = 0;

    uint32_t baseRowsPerCore = 0;
    uint32_t tailRows = 0;
    uint32_t tileRows = 0;

    uint32_t mm1BaseM = 0;
    uint32_t mm1BaseN = 0;
    uint32_t mm1BaseK = 0;

    uint32_t swiBaseRows = 0;
    uint32_t swiBaseCols = 0;

    uint32_t mm2BaseM = 0;
    uint32_t mm2BaseN = 0;
    uint32_t mm2BaseK = 0;
};

} // namespace swiglu_gated_mlp_kernel

using SwigluGatedMlpTilingData = swiglu_gated_mlp_kernel::SwigluGatedMlpTilingData;

#endif  // OP_KERNEL_GLU_TILING_HPP_
