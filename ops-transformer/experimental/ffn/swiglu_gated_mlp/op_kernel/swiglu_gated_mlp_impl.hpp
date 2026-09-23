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
 * \file swiglu_gated_mlp_impl.hpp
 * \brief Standalone staged SwigluGatedMlp implementation.
 */

#ifndef OP_KERNEL_SWIGLU_GATED_MLP_IMPL_HPP_
#define OP_KERNEL_SWIGLU_GATED_MLP_IMPL_HPP_

#include <type_traits>

#include "kernel_operator.h"
#include "common/matmul_act/matmul/kernel/kernel_matmul.h"
#include "glu_tiling_kernel.hpp"

namespace swiglu_gated_mlp_kernel {

constexpr uint64_t WG_MLP_SYS_WORKSPACE_BYTES = 16UL * 1024UL * 1024UL;

template <typename T>
__aicore__ inline T ZeroValue()
{
    return static_cast<T>(0.0f);
}

template <typename T>
__aicore__ inline float ScalarToFloat(T value)
{
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        return AscendC::ToFloat(value);
    } else {
        return static_cast<float>(value);
    }
}

template <typename T>
__aicore__ inline T ScalarFromFloat(float value)
{
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        return AscendC::ToBfloat16(value);
    } else {
        return static_cast<T>(value);
    }
}

template <typename T>
__aicore__ inline T ScalarMin(T a, T b)
{
    return (a < b) ? a : b;
}

template <typename T>
__aicore__ inline T ScalarMax(T a, T b)
{
    return (a > b) ? a : b;
}

__aicore__ inline float SwigluGatedMlpSigmoidApprox(float x)
{
    if (x >= 8.0f) {
        return 0.999664664f;
    }
    if (x <= -8.0f) {
        return 0.000335350f;
    }
    const float ax = (x >= 0.0f) ? x : -x;
    if (ax <= 1.0f) {
        const float x2 = x * x;
        return 0.5f + x * (0.25f - x2 * (0.020833333f - x2 * 0.002083333f));
    }
    return 0.5f * (x / (1.0f + ax)) + 0.5f;
}

template <typename T>
class SwigluGatedMlpCubeMatmulStage {
public:
    __aicore__ SwigluGatedMlpCubeMatmulStage() = default;
    __aicore__ ~SwigluGatedMlpCubeMatmulStage() = default;

    __aicore__ void Init(GM_ADDR xGm, GM_ADDR weightGm, GM_ADDR yGm,
        const SwigluGatedMlpTilingData &tilingData)
    {
        tiling_.Parse(tilingData);
        xGm_ = xGm;
        weightGm_ = weightGm;
        yGm_ = yGm;
    }

    __aicore__ void Process()
    {
        if ASCEND_IS_AIV {
            return;
        }
        if (tiling_.totalRows == 0 || tiling_.hiddenSize == 0 || tiling_.outSize == 0) {
            return;
        }

        if (tiling_.totalRows >= 128U) {
            RunMatmul<Act::Gemm::_128>();
        } else if (tiling_.totalRows >= 64U) {
            RunMatmul<Act::Gemm::_64>();
        } else if (tiling_.totalRows >= 32U) {
            RunMatmul<Act::Gemm::_32>();
        } else {
            RunMatmul<Act::Gemm::_16>();
        }
    }

private:
    template <typename MTile>
    __aicore__ void RunMatmul()
    {
        using L1TileShape = AscendC::Shape<MTile, Act::Gemm::_256, Act::Gemm::_64>;
        using L0TileShape = AscendC::Shape<MTile, Act::Gemm::_256, Act::Gemm::_64>;
        using Scheduler = Act::Gemm::IterateKScheduler;
        using BlockMmad = Act::Gemm::Block::BlockMmadBuilder<
            T, Act::Gemm::layout::RowMajor,
            T, Act::Gemm::layout::RowMajor,
            T, Act::Gemm::layout::RowMajor,
            T, Act::Gemm::layout::RowMajor,
            L1TileShape, L0TileShape, Scheduler,
            Act::Gemm::MatmulMultiBlockWithLayout<>>;
        using Epilogue = Act::Gemm::Block::BlockEpilogueEmpty;
        using MatmulKernel = Act::Gemm::Kernel::KernelMatmul<Act::Gemm::MatmulShape, BlockMmad, Epilogue, Scheduler>;
        using MmadArgs = typename BlockMmad::Arguments;
        using EpilogueArgs = typename Epilogue::Arguments;
        using Arguments = typename MatmulKernel::Arguments;
        using Params = typename MatmulKernel::Params;

        MmadArgs mmArgs{xGm_, weightGm_, yGm_, nullptr};
        EpilogueArgs epilogueArgs{};
        Arguments args{
            {static_cast<int64_t>(tiling_.totalRows), static_cast<int64_t>(tiling_.outSize),
             static_cast<int64_t>(tiling_.hiddenSize), 1},
            mmArgs,
            epilogueArgs};
        Params params = MatmulKernel::InitParams(args, nullptr);
        AscendC::TPipe pipe;
        MatmulKernel op;
        op(params);
    }

    GluTilingInfo tiling_ {};
    GM_ADDR xGm_ = nullptr;
    GM_ADDR weightGm_ = nullptr;
    GM_ADDR yGm_ = nullptr;
};

class SwigluGatedMlpFusedFloatStage {
public:
    __aicore__ SwigluGatedMlpFusedFloatStage() = default;
    __aicore__ ~SwigluGatedMlpFusedFloatStage() = default;

    __aicore__ void Init(GM_ADDR xGm, GM_ADDR gateUpWeightGm, GM_ADDR downWeightGm,
        GM_ADDR yGm, GM_ADDR workspaceGm, const SwigluGatedMlpTilingData &tilingData)
    {
        tiling_.Parse(tilingData);
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(xGm), tiling_.totalRows * tiling_.hiddenSize);
        gateUpWeightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gateUpWeightGm),
            tiling_.hiddenSize * tiling_.gateUpSize);
        downWeightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(downWeightGm),
            tiling_.intermediateSize * tiling_.outSize);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(yGm), tiling_.totalRows * tiling_.outSize);
        (void)workspaceGm;
    }

    __aicore__ void Process()
    {
        if (tiling_.totalRows == 0 || tiling_.hiddenSize == 0 ||
            tiling_.intermediateSize == 0 || tiling_.outSize == 0) {
            return;
        }

        const uint64_t blockNum = ScalarMax<uint64_t>(1U, static_cast<uint64_t>(tiling_.usedCoreNum));
        const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
        for (uint64_t row = blockIdx; row < tiling_.totalRows; row += blockNum) {
            ComputeOutput(row);
        }
    }

private:
    __aicore__ static float SigmoidScalar(float x)
    {
        return SwigluGatedMlpSigmoidApprox(x);
    }

    __aicore__ void ComputeOutput(uint64_t row)
    {
        for (uint64_t col = 0; col < tiling_.outSize; ++col) {
            float acc = 0.0f;
            for (uint64_t k = 0; k < tiling_.intermediateSize; ++k) {
                float gate = 0.0f;
                float up = 0.0f;
                for (uint64_t h = 0; h < tiling_.hiddenSize; ++h) {
                    const float x = xGm_.GetValue(row * tiling_.hiddenSize + h);
                    gate += x * gateUpWeightGm_.GetValue(h * tiling_.gateUpSize + k);
                    up += x * gateUpWeightGm_.GetValue(h * tiling_.gateUpSize + tiling_.intermediateSize + k);
                }
                acc += gate * SigmoidScalar(gate) * up * downWeightGm_.GetValue(k * tiling_.outSize + col);
            }
            yGm_.SetValue(row * tiling_.outSize + col, acc);
        }
    }

    GluTilingInfo tiling_ {};
    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> gateUpWeightGm_;
    AscendC::GlobalTensor<float> downWeightGm_;
    AscendC::GlobalTensor<float> yGm_;
};

template <typename T>
class SwigluGatedMlpScalarSwigluStage {
public:
    __aicore__ SwigluGatedMlpScalarSwigluStage() = default;
    __aicore__ ~SwigluGatedMlpScalarSwigluStage() = default;

    __aicore__ void Init(GM_ADDR gateUpGm, GM_ADDR hiddenGm,
        const SwigluGatedMlpTilingData &tilingData)
    {
        tiling_.Parse(tilingData);
        gateUpGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(gateUpGm), tiling_.totalRows * tiling_.gateUpSize);
        hiddenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(hiddenGm),
            tiling_.totalRows * tiling_.intermediateSize);
    }

    __aicore__ void Process()
    {
        if (tiling_.totalRows == 0 || tiling_.intermediateSize == 0) {
            return;
        }

        const uint64_t total = tiling_.totalRows * tiling_.intermediateSize;
        const uint64_t blockNum = ScalarMax<uint64_t>(1U, static_cast<uint64_t>(tiling_.usedCoreNum));
        const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());

        for (uint64_t idx = blockIdx; idx < total; idx += blockNum) {
            const uint64_t row = idx / tiling_.intermediateSize;
            const uint64_t col = idx - row * tiling_.intermediateSize;
            const uint64_t gateOffset = row * tiling_.gateUpSize + col;
            const float gate = ScalarToFloat<T>(gateUpGm_.GetValue(gateOffset));
            const float up = ScalarToFloat<T>(gateUpGm_.GetValue(gateOffset + tiling_.intermediateSize));
            const float out = gate * SwigluGatedMlpSigmoidApprox(gate) * up;
            hiddenGm_.SetValue(row * tiling_.intermediateSize + col, ScalarFromFloat<T>(out));
        }
    }

private:
    GluTilingInfo tiling_ {};
    AscendC::GlobalTensor<T> gateUpGm_;
    AscendC::GlobalTensor<T> hiddenGm_;
};

template <typename T, int BUFFER_NUM>
class SwigluGatedMlpSwigluStage {
public:
    __aicore__ SwigluGatedMlpSwigluStage() = default;
    __aicore__ ~SwigluGatedMlpSwigluStage() = default;

    __aicore__ void Init(GM_ADDR gateUpGm, GM_ADDR hiddenGm,
        const SwigluGatedMlpTilingData &tilingData)
    {
        tiling_.Parse(tilingData);
        gateUpGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(gateUpGm), tiling_.totalRows * tiling_.gateUpSize);
        hiddenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(hiddenGm),
            tiling_.totalRows * tiling_.intermediateSize);

        alignElems_ = (tiling_.dtypeSize == 0) ? 1U : ScalarMax<uint32_t>(1U, 32U / tiling_.dtypeSize);
        baseRows_ = ScalarMax<uint32_t>(1U, tiling_.swiBaseRows);
        baseCols_ = ScalarMax<uint32_t>(1U, tiling_.swiBaseCols);
        alignedBaseCols_ = AlignUp<uint32_t>(baseCols_, alignElems_);
        tileLength_ = static_cast<uint64_t>(baseRows_) * static_cast<uint64_t>(alignedBaseCols_);

        pipe_.InitBuffer(gateQueue_, BUFFER_NUM, tileLength_ * sizeof(T));
        pipe_.InitBuffer(upQueue_, BUFFER_NUM, tileLength_ * sizeof(T));
        pipe_.InitBuffer(outQueue_, BUFFER_NUM, tileLength_ * sizeof(T));
        pipe_.InitBuffer(calcBuf_, tileLength_ * sizeof(float) * 3U);
    }

    __aicore__ void Process()
    {
        if (tiling_.totalRows == 0 || tiling_.intermediateSize == 0) {
            return;
        }

        const uint64_t rowTiles = DivCeil<uint64_t>(tiling_.totalRows, static_cast<uint64_t>(baseRows_));
        const uint64_t colTiles = DivCeil<uint64_t>(tiling_.intermediateSize, static_cast<uint64_t>(baseCols_));
        if (rowTiles == 0 || colTiles == 0) {
            return;
        }
        const uint64_t totalTiles = rowTiles * colTiles;
        uint64_t blockNum = ScalarMax<uint64_t>(1U, static_cast<uint64_t>(GetBlockNum()));
        if ASCEND_IS_AIV {
            blockNum *= ScalarMax<uint64_t>(1U, static_cast<uint64_t>(GetTaskRation()));
        }
        const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());

        for (uint64_t tileIdx = blockIdx; tileIdx < totalTiles; tileIdx += blockNum) {
            const uint64_t rowTile = tileIdx / colTiles;
            const uint64_t colTile = tileIdx % colTiles;
            const uint64_t rowStart = rowTile * static_cast<uint64_t>(baseRows_);
            const uint64_t colStart = colTile * static_cast<uint64_t>(baseCols_);
            const uint32_t curRows = static_cast<uint32_t>(
                ScalarMin<uint64_t>(static_cast<uint64_t>(baseRows_), tiling_.totalRows - rowStart));
            const uint32_t curCols = static_cast<uint32_t>(
                ScalarMin<uint64_t>(static_cast<uint64_t>(baseCols_), tiling_.intermediateSize - colStart));
            const uint32_t alignedCols = AlignUp<uint32_t>(curCols, alignElems_);
            const uint64_t calcLen = static_cast<uint64_t>(curRows) * static_cast<uint64_t>(alignedCols);
            CopyIn(rowStart, colStart, curRows, curCols, alignedCols);
            Compute(calcLen);
            CopyOut(rowStart, colStart, curRows, curCols, alignedCols);
        }
    }

private:
    __aicore__ void CopyIn(uint64_t rowStart, uint64_t colStart, uint32_t curRows, uint32_t curCols,
        uint32_t alignedCols)
    {
        AscendC::LocalTensor<T> gateLocal = gateQueue_.template AllocTensor<T>();
        AscendC::LocalTensor<T> upLocal = upQueue_.template AllocTensor<T>();

        const uint64_t gateOffset = rowStart * tiling_.gateUpSize + colStart;
        const uint64_t upOffset = gateOffset + tiling_.intermediateSize;
        const uint32_t rightPad = static_cast<uint32_t>(alignedCols - curCols);
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(curRows),
            static_cast<uint32_t>(curCols * sizeof(T)),
            static_cast<uint32_t>((tiling_.gateUpSize - curCols) * sizeof(T)),
            static_cast<uint32_t>(rightPad * sizeof(T)),
            0};
        AscendC::DataCopyPadExtParams<T> padParams{
            rightPad != 0U, 0, static_cast<uint8_t>(rightPad), ZeroValue<T>()};

        DataCopyPad(gateLocal, gateUpGm_[gateOffset], copyParams, padParams);
        DataCopyPad(upLocal, gateUpGm_[upOffset], copyParams, padParams);
        gateQueue_.template EnQue<T>(gateLocal);
        upQueue_.template EnQue<T>(upLocal);
    }

    __aicore__ void Compute(uint64_t calcLen)
    {
        AscendC::LocalTensor<T> gateLocal = gateQueue_.template DeQue<T>();
        AscendC::LocalTensor<T> upLocal = upQueue_.template DeQue<T>();
        AscendC::LocalTensor<T> outLocal = outQueue_.template AllocTensor<T>();

        if constexpr (std::is_same_v<T, float>) {
            AscendC::LocalTensor<float> tmp = calcBuf_.Get<float>();
            Sigmoid(tmp, gateLocal, calcLen);
            PipeBarrier<PIPE_V>();
            Mul(tmp, tmp, gateLocal, calcLen);
            PipeBarrier<PIPE_V>();
            Mul(outLocal, tmp, upLocal, calcLen);
            PipeBarrier<PIPE_V>();
        } else {
            AscendC::LocalTensor<float> gateFp32 = calcBuf_.Get<float>();
            AscendC::LocalTensor<float> upFp32 = gateFp32[calcLen];
            AscendC::LocalTensor<float> tmp = upFp32[calcLen];

            Cast(gateFp32, gateLocal, AscendC::RoundMode::CAST_NONE, calcLen);
            PipeBarrier<PIPE_V>();
            Cast(upFp32, upLocal, AscendC::RoundMode::CAST_NONE, calcLen);
            PipeBarrier<PIPE_V>();
            Sigmoid(tmp, gateFp32, calcLen);
            PipeBarrier<PIPE_V>();
            Mul(gateFp32, gateFp32, tmp, calcLen);
            PipeBarrier<PIPE_V>();
            Mul(tmp, gateFp32, upFp32, calcLen);
            PipeBarrier<PIPE_V>();
            if constexpr (std::is_same_v<T, bfloat16_t>) {
                Cast(outLocal, tmp, AscendC::RoundMode::CAST_RINT, calcLen);
            } else {
                Cast(outLocal, tmp, AscendC::RoundMode::CAST_NONE, calcLen);
            }
            PipeBarrier<PIPE_V>();
        }

        gateQueue_.template FreeTensor<T>(gateLocal);
        upQueue_.template FreeTensor<T>(upLocal);
        outQueue_.template EnQue<T>(outLocal);
    }

    __aicore__ void CopyOut(uint64_t rowStart, uint64_t colStart, uint32_t curRows, uint32_t curCols,
        uint32_t alignedCols)
    {
        AscendC::LocalTensor<T> outLocal = outQueue_.template DeQue<T>();
        const uint64_t outOffset = rowStart * tiling_.intermediateSize + colStart;
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(curRows),
            static_cast<uint32_t>(curCols * sizeof(T)),
            static_cast<uint32_t>((alignedCols - curCols) * sizeof(T)),
            static_cast<uint32_t>((tiling_.intermediateSize - curCols) * sizeof(T)),
            0};
        DataCopyPad(hiddenGm_[outOffset], outLocal, copyParams);
        outQueue_.template FreeTensor<T>(outLocal);
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> gateQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> upQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf_;

    GluTilingInfo tiling_ {};
    AscendC::GlobalTensor<T> gateUpGm_;
    AscendC::GlobalTensor<T> hiddenGm_;

    uint32_t alignElems_ = 1;
    uint32_t baseRows_ = 1;
    uint32_t baseCols_ = 1;
    uint32_t alignedBaseCols_ = 1;
    uint64_t tileLength_ = 1;
};

template <int BUFFER_NUM>
class SwigluGatedMlpKernelFp16 : public SwigluGatedMlpSwigluStage<half, BUFFER_NUM> {
};

template <int BUFFER_NUM>
class SwigluGatedMlpKernelFp32 : public SwigluGatedMlpSwigluStage<float, BUFFER_NUM> {
};

template <int BUFFER_NUM>
class SwigluGatedMlpKernelBf16 : public SwigluGatedMlpSwigluStage<bfloat16_t, BUFFER_NUM> {
};

}  // namespace swiglu_gated_mlp_kernel

#include "swiglu_gated_mlp_scalar_matmul.hpp"

#endif  // OP_KERNEL_SWIGLU_GATED_MLP_IMPL_HPP_
