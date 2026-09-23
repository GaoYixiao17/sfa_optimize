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

#ifndef OP_KERNEL_SWIGLU_GATED_MLP_SCALAR_MATMUL_HPP_
#define OP_KERNEL_SWIGLU_GATED_MLP_SCALAR_MATMUL_HPP_

namespace swiglu_gated_mlp_kernel {

template <typename T>
class SwigluGatedMlpScalarMatmul {
public:
    __aicore__ SwigluGatedMlpScalarMatmul() = default;
    __aicore__ ~SwigluGatedMlpScalarMatmul() = default;

    __aicore__ void Init(GM_ADDR xGm, GM_ADDR weightGm, GM_ADDR yGm,
        const SwigluGatedMlpTilingData &tilingData)
    {
        tiling_.Parse(tilingData);
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(xGm), tiling_.totalRows * tiling_.hiddenSize);
        weightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(weightGm), tiling_.hiddenSize * tiling_.outSize);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(yGm), tiling_.totalRows * tiling_.outSize);
    }

    __aicore__ void Process()
    {
        if (tiling_.totalRows == 0 || tiling_.hiddenSize == 0 || tiling_.outSize == 0) {
            return;
        }

        constexpr uint64_t rowBlock = 4U;
        constexpr uint64_t colBlock = 8U;
        const uint64_t rowTiles = DivCeil<uint64_t>(tiling_.totalRows, rowBlock);
        const uint64_t colTiles = DivCeil<uint64_t>(tiling_.outSize, colBlock);
        if (rowTiles == 0 || colTiles == 0) {
            return;
        }
        const uint64_t total = rowTiles * colTiles;
        const uint64_t blockNum = ScalarMax<uint64_t>(1U, static_cast<uint64_t>(tiling_.usedCoreNum));
        const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());

        for (uint64_t tileIdx = blockIdx; tileIdx < total; tileIdx += blockNum) {
            ProcessTile(tileIdx, colTiles, rowBlock, colBlock);
        }
    }

private:
    __aicore__ void ProcessTile(uint64_t tileIdx, uint64_t colTiles, uint64_t rowBlock, uint64_t colBlock)
    {
        if (colTiles == 0) {
            return;
        }
        const uint64_t rowTile = tileIdx / colTiles;
        const uint64_t rowStart = rowTile * rowBlock;
        const uint64_t colStart = (tileIdx - rowTile * colTiles) * colBlock;
        const uint64_t curRows = ScalarMin<uint64_t>(rowBlock, tiling_.totalRows - rowStart);
        const uint64_t curCols = ScalarMin<uint64_t>(colBlock, tiling_.outSize - colStart);

        if (curRows == rowBlock && curCols == colBlock) {
            ProcessFullTile(rowStart, colStart);
            return;
        }
        ProcessTailTile(rowStart, colStart, curRows, curCols);
    }

    __aicore__ void ProcessFullTile(uint64_t rowStart, uint64_t colStart)
    {
        float acc[4][8];
        InitFullTileAcc(acc);
        const uint64_t row0Base = rowStart * tiling_.hiddenSize;
        const uint64_t row1Base = row0Base + tiling_.hiddenSize;
        const uint64_t row2Base = row1Base + tiling_.hiddenSize;
        const uint64_t row3Base = row2Base + tiling_.hiddenSize;

        for (uint64_t k = 0; k < tiling_.hiddenSize; ++k) {
            const float a0 = ScalarToFloat<T>(xGm_.GetValue(row0Base + k));
            const float a1 = ScalarToFloat<T>(xGm_.GetValue(row1Base + k));
            const float a2 = ScalarToFloat<T>(xGm_.GetValue(row2Base + k));
            const float a3 = ScalarToFloat<T>(xGm_.GetValue(row3Base + k));
            AccumulateFullTile(acc, a0, a1, a2, a3, k * tiling_.outSize + colStart);
        }
        StoreFullTile(rowStart, colStart, acc);
    }

    __aicore__ void InitFullTileAcc(float (&acc)[4][8])
    {
        for (uint64_t r = 0; r < 4U; ++r) {
            for (uint64_t c = 0; c < 8U; ++c) {
                acc[r][c] = 0.0f;
            }
        }
    }

    __aicore__ void AccumulateFullTile(float (&acc)[4][8], float a0, float a1, float a2, float a3,
        uint64_t weightBase)
    {
        for (uint64_t c = 0; c < 8U; ++c) {
            const float w = ScalarToFloat<T>(weightGm_.GetValue(weightBase + c));
            acc[0][c] += a0 * w;
            acc[1][c] += a1 * w;
            acc[2][c] += a2 * w;
            acc[3][c] += a3 * w;
        }
    }

    __aicore__ void StoreFullTile(uint64_t rowStart, uint64_t colStart, float (&acc)[4][8])
    {
        for (uint64_t r = 0; r < 4U; ++r) {
            const uint64_t outBase = (rowStart + r) * tiling_.outSize + colStart;
            for (uint64_t c = 0; c < 8U; ++c) {
                yGm_.SetValue(outBase + c, ScalarFromFloat<T>(acc[r][c]));
            }
        }
    }

    __aicore__ void ProcessTailTile(uint64_t rowStart, uint64_t colStart, uint64_t curRows, uint64_t curCols)
    {
        for (uint64_t rowOffset = 0; rowOffset < curRows; ++rowOffset) {
            const uint64_t row = rowStart + rowOffset;
            float acc[8];
            InitTailAcc(acc);
            for (uint64_t k = 0; k < tiling_.hiddenSize; ++k) {
                const float a = ScalarToFloat<T>(xGm_.GetValue(row * tiling_.hiddenSize + k));
                AccumulateTail(acc, a, k * tiling_.outSize + colStart, curCols);
            }
            StoreTail(row, colStart, acc, curCols);
        }
    }

    __aicore__ void InitTailAcc(float (&acc)[8])
    {
        for (uint64_t c = 0; c < 8U; ++c) {
            acc[c] = 0.0f;
        }
    }

    __aicore__ void AccumulateTail(float (&acc)[8], float a, uint64_t weightBase, uint64_t curCols)
    {
        for (uint64_t c = 0; c < curCols; ++c) {
            acc[c] += a * ScalarToFloat<T>(weightGm_.GetValue(weightBase + c));
        }
    }

    __aicore__ void StoreTail(uint64_t row, uint64_t colStart, float (&acc)[8], uint64_t curCols)
    {
        const uint64_t outBase = row * tiling_.outSize + colStart;
        for (uint64_t c = 0; c < curCols; ++c) {
            yGm_.SetValue(outBase + c, ScalarFromFloat<T>(acc[c]));
        }
    }

    GluTilingInfo tiling_ {};
    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<T> weightGm_;
    AscendC::GlobalTensor<T> yGm_;
};

}  // namespace swiglu_gated_mlp_kernel

#endif  // OP_KERNEL_SWIGLU_GATED_MLP_SCALAR_MATMUL_HPP_
