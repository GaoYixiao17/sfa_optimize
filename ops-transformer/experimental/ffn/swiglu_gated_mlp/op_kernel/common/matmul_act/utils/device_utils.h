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
 * \file device_utils.h
 * \brief
 */

#ifndef UTILS_DEVICE_UTILS_H
#define UTILS_DEVICE_UTILS_H
#include "kernel_operator.h"
namespace Act {
namespace Gemm {

template <AscendC::HardEvent event>
__aicore__ inline void TPipeSetWaitFlag()
{
    auto eventID = GetTPipePtr()->FetchEventID(event);
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

// blockAlign 32B
template <typename DataType, int64_t blockSize = 32>
__aicore__ inline int64_t AlignBlock(const int64_t& t)
{
    return AscendC::Align(t, static_cast<int64_t>(blockSize / sizeof(DataType)));
}

} // namespace Gemm
} // namespace Act
#endif