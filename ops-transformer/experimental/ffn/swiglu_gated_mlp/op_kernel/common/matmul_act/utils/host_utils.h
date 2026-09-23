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
 * \file host_utils.h
 * \brief
 */

#ifndef UTILS_HOST_UTILS_H
#define UTILS_HOST_UTILS_H
#ifndef __NPU_ARCH__
#include "tiling/platform/platform_ascendc.h"
#endif
namespace Act {
namespace Gemm {

static int64_t GetCoreNum()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (ascendcPlatform == nullptr) {
        return 0;
    } else {
        return ascendcPlatform->GetCoreNumAic();
    }
}

static size_t GetSysWorkspaceSize()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (ascendcPlatform == nullptr) {
        return 0;
    } else {
        return static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
    }
}
} // namespace Gemm
} // namespace Act
#endif
