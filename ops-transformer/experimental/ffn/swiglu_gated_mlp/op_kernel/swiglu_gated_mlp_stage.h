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

#ifndef OP_KERNEL_SWIGLU_GATED_MLP_STAGE_H_
#define OP_KERNEL_SWIGLU_GATED_MLP_STAGE_H_

#include <cstdint>

constexpr int64_t WG_MLP_STAGE_MM1 = 101;
constexpr int64_t WG_MLP_STAGE_SWIGLU = 102;
constexpr int64_t WG_MLP_STAGE_MM2 = 103;
constexpr int64_t WG_MLP_STAGE_FUSED = 104;
constexpr int64_t WG_MLP_STAGE_DTYPE_STRIDE = 1000;

#endif  // OP_KERNEL_SWIGLU_GATED_MLP_STAGE_H_
