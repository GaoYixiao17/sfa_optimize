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
 * \file common_def.h
 * \brief
 */

#ifndef COMMON_DEF_H
#define COMMON_DEF_H

#include "lib/hccl/hccl.h"

namespace MatmulReduceScatterV2Impl {
using A_DTYPE = DTYPE_X1;
using B_DTYPE = DTYPE_X1;
using C_DTYPE = DTYPE_Y;
using BIAS_DTYPE = DTYPE_Y;
constexpr uint8_t MAX_HANDLE = 64;          // 最大handle数
constexpr uint8_t MC2_DEBUG_ONLY_CUBE = 1;  // 只计算不通信
constexpr uint8_t MC2_DEBUG_ONLY_AICPU = 4; // 只通信不计算
constexpr uint64_t BLOCK_SIZE = 128;

constexpr int CCU_COMM_MODE = 0;
constexpr int AICPU_COMM_MODE = 1;

template <int commMode>
struct HcclTypeSelector {
    using type = AscendC::Hccl<AscendC::HcclServerType::HCCL_SERVER_TYPE_CCU>;
};
template <>
struct HcclTypeSelector<AICPU_COMM_MODE> {
    using type = AscendC::Hccl<AscendC::HcclServerType::HCCL_SERVER_TYPE_AICPU>;
};
template <class T>
struct BiasType {
    using type = float;
};
template <>
struct BiasType<half> {
    using type = half;
};
template <>
struct BiasType<bfloat16_t> {
    using type = bfloat16_t;
};
__aicore__ inline uint64_t CeilDiv(uint64_t x, uint64_t y)
{
    return y == 0 ? x : (x + y - 1) / y;
}

} // namespace MatmulReduceScatterV2Impl
#endif // COMMON_DEF_H
