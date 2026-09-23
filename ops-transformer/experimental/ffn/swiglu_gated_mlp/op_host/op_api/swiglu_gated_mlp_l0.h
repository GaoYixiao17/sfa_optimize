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

#ifndef OP_API_INC_LEVEL0_SWIGLU_GATED_MLP_L0_H_
#define OP_API_INC_LEVEL0_SWIGLU_GATED_MLP_L0_H_

#include "opdev/op_executor.h"

namespace l0op {

const aclTensor* SwigluGatedMlp(
    const aclTensor* x,
    const aclTensor* gateUpWeight,
    const aclTensor* downWeight,
    int64_t cubeMathType,
    aclOpExecutor* executor);

}  // namespace l0op

#endif  // OP_API_INC_LEVEL0_SWIGLU_GATED_MLP_L0_H_
