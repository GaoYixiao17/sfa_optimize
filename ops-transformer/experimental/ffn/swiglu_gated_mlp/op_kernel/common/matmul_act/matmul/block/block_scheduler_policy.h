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
 * \file block_scheduler_policy.h
 * \brief
 */

#ifndef MATMUL_BLOCK_BLOCK_SCHEDULER_POLICY_H
#define MATMUL_BLOCK_BLOCK_SCHEDULER_POLICY_H

namespace Act {
namespace Gemm {

// FULL_LOAD_MODE_表示全载模式 0: 非全载 2：B全载
template <uint64_t FULL_LOAD_MODE_ = 0>
struct BuiltInAswtScheduler {
    static constexpr uint64_t FULL_LOAD_MODE = FULL_LOAD_MODE_;
};
struct BuiltInStreamKScheduler {};
struct BuiltInIterBatchScheduler {};
struct BuiltInBatchMatmulToMulScheduler {};
struct IterateKScheduler {};
struct QuantIterateKScheduler {};
} // namespace Gemm
} // namespace Act
#endif