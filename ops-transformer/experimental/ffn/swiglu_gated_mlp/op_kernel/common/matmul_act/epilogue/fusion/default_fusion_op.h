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
 * \file default_fusion_op.h
 * \brief
 */

#ifndef EPILOGUE_FUSION_DEFAULT_FUSION_OP_H
#define EPILOGUE_FUSION_DEFAULT_FUSION_OP_H
#include "kernel_operator.h"
#include "../../utils/common_utils.h"
#include "../../utils/device_utils.h"

namespace Act {
namespace Gemm {
namespace Block {
template <typename DataTypeOut_, typename DataTypeIn_>
class DefaultFusionOp {
public:
    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    __aicore__ inline DefaultFusionOp(){};

    struct Arguments {};

    struct Params {};

    __aicore__ inline void Init(Params const& params, int64_t calcM, int64_t calcN, int64_t n) {}
    __aicore__ inline void InitBuffers() {}
    __aicore__ inline int64_t GetUbSizeOneM(int64_t ubCalcN)
    {
        return 0;
    }
    __aicore__ inline void Run(AscendC::LocalTensor<DataTypeOut>& dstLocal, AscendC::LocalTensor<DataTypeIn>& srcLocal,
                               int64_t curAivM, int64_t curAivN, int64_t mIdx, int64_t nIdx)
    {
        dstLocal = srcLocal;
        return;
    }

    __aicore__ inline void operator()(AscendC::LocalTensor<DataTypeOut>& dstLocal,
                                      AscendC::LocalTensor<DataTypeIn>& srcLocal, int64_t curAivM, int64_t curAivN,
                                      int64_t mIdx, int64_t nIdx)
    {
        Run(dstLocal, srcLocal, curAivM, curAivN, mIdx, nIdx);
    }
};
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif // EPILOGUE_FUSION_DEFAULT_FUSION_OP_H
