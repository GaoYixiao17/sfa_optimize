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
 * \file block_epilogue_empty.h
 * \brief
 */

#ifndef EPILOGUE_BLOCK_EPILOGUE_EMPTY_H
#define EPILOGUE_BLOCK_EPILOGUE_EMPTY_H
#include "kernel_operator.h"
#include "fusion/default_fusion_op.h"
#include "../utils/common_utils.h"
#include "../utils/device_utils.h"
#include "../utils/status_utils.h"

namespace Act {
namespace Gemm {
namespace Block {

class BlockEpilogueEmpty {
public:
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;

    struct Arguments {
        Arguments() = default;
    };

    struct Params {
        Params() = default;
    };

    __aicore__ inline BlockEpilogueEmpty() {}

    __aicore__ inline void Run()
    {
        return;
    }

    __aicore__ inline void operator()(Arguments const& params)
    {
        Run();
    }

    __host_aicore__ static Params InitParams(Arguments const& args, GM_ADDR workspaceGm)
    {
        Params params = {};
        return params;
    }

    __host_aicore__ static size_t GetWorkspaceSize(int64_t blockNum, int64_t l1M, int64_t l1N)
    {
        return 0;
    }

    __host_aicore__ static Status CanImplement(Arguments const& args)
    {
        return Status::success;
    }

    __aicore__ inline void operator()(BlockShape const& blockShape, BlockCoord const& blockCoord,
                                      int64_t dstStartOffset = 0, int64_t srcStartOffset = 0)
    {
        return;
    }
};
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif
