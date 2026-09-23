/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SCATTER_PA_KV_CACHE_WITH_K_SCALE_TILING_DATA_H
#define SCATTER_PA_KV_CACHE_WITH_K_SCALE_TILING_DATA_H

#include <cstdint>

struct ScatterPaKvCacheWithKScaleTilingData {
    int64_t needCoreNum;
    int64_t numTokens;
    int64_t numHead;
    int64_t kHeadSize;
    int64_t vHeadSize;
    int64_t numBlocks;
    int64_t blockSize;
    int64_t maxSlot;
    int64_t keyStride[3];
    int64_t valueStride[3];
    int64_t keyCacheStride[4];
    int64_t valueCacheStride[4];
    int64_t slotMappingStride[1];
    int64_t keyScaleStride[2];
    int64_t keyScaleCacheStride[4];
};

#endif // SCATTER_PA_KV_CACHE_WITH_K_SCALE_TILING_DATA_H
