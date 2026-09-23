/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
 * \file fusion_mul.h
 * \brief
 */

#ifndef EPILOGUE_FUSION_FUSION_MUL_H
#define EPILOGUE_FUSION_FUSION_MUL_H
#include "kernel_operator.h"
#include "../../utils/common_utils.h"
#include "../../utils/device_utils.h"

namespace Act {
namespace Gemm {
namespace Block {
template <typename DataTypeOut_, typename DataTypeIn_>
class FusionMul {
public:
    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    __aicore__ inline FusionMul(){};

    struct Arguments {
        GM_ADDR inputGmAddr{nullptr};
    };

    struct Params {
        GM_ADDR inputGmAddr{nullptr};
    };

    AscendC::LocalTensor<DataTypeIn> inputLocal_;
    AscendC::GlobalTensor<DataTypeIn> inputGlobal_;
    int64_t stageSize_{0};
    int64_t ubCalcM_{0};
    int64_t ubCalcN_{0};
    int64_t strideN_{0};

    template <class LocalTensor>
    __aicore__ inline void Init(Params const& params, LocalTensor ubTensor, int64_t ubCalcM, int64_t ubCalcN,
                                int64_t& ubOffset, int64_t& stageSize)
    {
        static constexpr int64_t stageNum = 2;
        inputGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataTypeIn*>(params.inputGmAddr));
        int64_t lastUBSize = AscendC::TOTAL_UB_SIZE - ubOffset * sizeof(DataTypeIn);
        ASCENDC_ASSERT((lastUBSize > ubCalcN * sizeof(DataTypeIn)), {
            KERNEL_LOG(KERNEL_ERROR, "ub size limit %ld, %ld!", lastUBSize, ubCalcN * sizeof(DataTypeIn));
        });
        stageSize_ = AscendC::Std::min(
            static_cast<int64_t>(lastUBSize / stageNum / sizeof(DataTypeIn_) / ubCalcN * ubCalcN), ubCalcM * ubCalcN);
        inputLocal_ = ubTensor[ubOffset];
        ubOffset += stageSize_;
        stageSize = stageSize_;
    }

    __aicore__ inline void operator()(const AscendC::LocalTensor<DataTypeIn>& srcLocal,
                                      AscendC::LocalTensor<DataTypeOut>& outputLocal, int64_t offset, int64_t curAivM,
                                      int64_t curAivN, int strideN, int64_t stageSize)
    {
        int64_t curAivNAlign = AlignBlock<DataTypeIn>(curAivN);
        TPipeSetWaitFlag<AscendC::HardEvent::MTE3_MTE2>();
        AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(stageSize / curAivNAlign),
                                     static_cast<uint32_t>(curAivN * sizeof(DataTypeOut)),
                                     static_cast<uint32_t>((strideN - curAivN) * sizeof(DataTypeIn)), 0, 0};
        AscendC::DataCopyPadExtParams<DataTypeIn> padParams{true, 0, 0, 0};
        AscendC::DataCopyPad(inputLocal_, inputGlobal_[offset], copyParams, padParams);
        TPipeSetWaitFlag<AscendC::HardEvent::MTE2_V>();
        AscendC::Mul(outputLocal, inputLocal_, srcLocal, stageSize);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __host_aicore__ static Params InitParams(Arguments const& args, GM_ADDR /* workspaceGm */)
    {
        return {args.inputGmAddr};
    }
};
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif // ACT_INCLUDE_EPILOGUE_FUSION_MUL_OP_H
