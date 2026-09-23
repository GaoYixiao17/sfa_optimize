/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "scatter_pa_kv_cache_with_k_scale_tiling_def.h"
#include "../../../../op_kernel/scatter_pa_kv_cache_with_k_scale_apt.cpp"
#include "../../../../op_kernel/arch35/scatter_pa_kv_cache_with_k_scale_tiling_key.h"

using namespace std;

extern "C" __global__ __aicore__ void scatter_pa_kv_cache_with_k_scale(
    GM_ADDR key, GM_ADDR value,
    GM_ADDR key_cache, GM_ADDR value_cache,
    GM_ADDR slot_mapping, GM_ADDR key_scale, GM_ADDR key_scale_cache,
    GM_ADDR key_cache_out, GM_ADDR value_cache_out, GM_ADDR key_scale_cache_out,
    GM_ADDR workspace, GM_ADDR tiling);

class scatter_pa_kv_cache_with_k_scale_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "scatter_pa_kv_cache_with_k_scale_test SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "scatter_pa_kv_cache_with_k_scale_test TearDown" << endl;
    }
};

TEST_F(scatter_pa_kv_cache_with_k_scale_test, test_case_specialized_fp8_e4m3_int32)
{
    int64_t numTokens = 4;
    int64_t numHead = 2;
    int64_t kHeadSize = 128;
    int64_t vHeadSize = 128;
    int64_t blockSize = 16;
    int64_t numBlocks = 2;
    int64_t maxSlot = numBlocks * blockSize;

    size_t keySize = numTokens * numHead * kHeadSize * sizeof(uint8_t);
    size_t valueSize = numTokens * numHead * vHeadSize * sizeof(uint8_t);
    size_t keyCacheSize = numBlocks * blockSize * numHead * kHeadSize * sizeof(uint8_t);
    size_t valueCacheSize = numBlocks * blockSize * numHead * vHeadSize * sizeof(uint8_t);
    size_t slotMappingSize = numTokens * sizeof(int32_t);
    size_t keyScaleSize = numTokens * numHead * sizeof(float);
    size_t keyScaleCacheSize = numBlocks * blockSize * numHead * sizeof(float);

    uint8_t* key = (uint8_t*)AscendC::GmAlloc(keySize);
    uint8_t* value = (uint8_t*)AscendC::GmAlloc(valueSize);
    uint8_t* keyCache = (uint8_t*)AscendC::GmAlloc(keyCacheSize);
    uint8_t* valueCache = (uint8_t*)AscendC::GmAlloc(valueCacheSize);
    uint8_t* slotMapping = (uint8_t*)AscendC::GmAlloc(slotMappingSize);
    uint8_t* keyScale = (uint8_t*)AscendC::GmAlloc(keyScaleSize);
    uint8_t* keyScaleCache = (uint8_t*)AscendC::GmAlloc(keyScaleCacheSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(ScatterPaKvCacheWithKScaleTilingData));

    ScatterPaKvCacheWithKScaleTilingData* tilingData = reinterpret_cast<ScatterPaKvCacheWithKScaleTilingData*>(tiling);
    tilingData->needCoreNum = 1;
    tilingData->numTokens = numTokens;
    tilingData->numHead = numHead;
    tilingData->kHeadSize = kHeadSize;
    tilingData->vHeadSize = vHeadSize;
    tilingData->numBlocks = numBlocks;
    tilingData->blockSize = blockSize;
    tilingData->maxSlot = maxSlot;
    tilingData->keyStride[0] = numHead * kHeadSize;
    tilingData->keyStride[1] = kHeadSize;
    tilingData->keyStride[2] = 1;
    tilingData->valueStride[0] = numHead * vHeadSize;
    tilingData->valueStride[1] = vHeadSize;
    tilingData->valueStride[2] = 1;
    tilingData->keyCacheStride[0] = blockSize * numHead * kHeadSize;
    tilingData->keyCacheStride[1] = kHeadSize;
    tilingData->keyCacheStride[2] = numHead * kHeadSize;
    tilingData->keyCacheStride[3] = 1;
    tilingData->valueCacheStride[0] = blockSize * numHead * vHeadSize;
    tilingData->valueCacheStride[1] = vHeadSize;
    tilingData->valueCacheStride[2] = numHead * vHeadSize;
    tilingData->valueCacheStride[3] = 1;
    tilingData->slotMappingStride[0] = 1;
    tilingData->keyScaleStride[0] = numHead;
    tilingData->keyScaleStride[1] = 1;
    tilingData->keyScaleCacheStride[0] = blockSize * numHead;
    tilingData->keyScaleCacheStride[1] = 1;
    tilingData->keyScaleCacheStride[2] = numHead;
    tilingData->keyScaleCacheStride[3] = 1;

    for (size_t i = 0; i < keySize; ++i) {
        key[i] = static_cast<uint8_t>(i % 256);
    }
    for (size_t i = 0; i < valueSize; ++i) {
        value[i] = static_cast<uint8_t>((i + 128) % 256);
    }
    for (size_t i = 0; i < keyCacheSize; ++i) {
        keyCache[i] = 0;
    }
    for (size_t i = 0; i < valueCacheSize; ++i) {
        valueCache[i] = 0;
    }
    int32_t* slotMappingData = reinterpret_cast<int32_t*>(slotMapping);
    slotMappingData[0] = 0;
    slotMappingData[1] = 16;
    slotMappingData[2] = 1;
    slotMappingData[3] = 17;

    float* keyScaleData = reinterpret_cast<float*>(keyScale);
    for (int64_t i = 0; i < numTokens * numHead; ++i) {
        keyScaleData[i] = 1.0f + static_cast<float>(i) * 0.1f;
    }
    float* keyScaleCacheData = reinterpret_cast<float*>(keyScaleCache);
    for (int64_t i = 0; i < numBlocks * blockSize * numHead; ++i) {
        keyScaleCacheData[i] = 0.0f;
    }

    ICPU_SET_TILING_KEY(1000001);
    
    auto scatterPaKvCacheWithKScaleWrapper = [] (GM_ADDR key, GM_ADDR value,
        GM_ADDR key_cache, GM_ADDR value_cache, GM_ADDR slot_mapping,
        GM_ADDR key_scale, GM_ADDR key_scale_cache,
        GM_ADDR key_cache_out, GM_ADDR value_cache_out, GM_ADDR key_scale_cache_out,
        GM_ADDR workspace, GM_ADDR tiling) {
        scatter_pa_kv_cache_with_k_scale<SCATTER_KV_CACHE_SCENE_SPECIALIZED>(
            key, value, key_cache, value_cache, slot_mapping, key_scale, key_scale_cache,
            key_cache_out, value_cache_out, key_scale_cache_out, workspace, tiling);
    };
    
    ICPU_RUN_KF(scatterPaKvCacheWithKScaleWrapper, 1,
        key, value, keyCache, valueCache, slotMapping, keyScale, keyScaleCache,
        keyCache, valueCache, keyScaleCache, workspace, tiling);

    AscendC::GmFree(key);
    AscendC::GmFree(value);
    AscendC::GmFree(keyCache);
    AscendC::GmFree(valueCache);
    AscendC::GmFree(slotMapping);
    AscendC::GmFree(keyScale);
    AscendC::GmFree(keyScaleCache);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(scatter_pa_kv_cache_with_k_scale_test, test_case_generalized_fp8_e4m3_int32)
{
    int64_t numTokens = 4;
    int64_t numHead = 2;
    int64_t kHeadSize = 128;
    int64_t vHeadSize = 64;
    int64_t blockSize = 16;
    int64_t numBlocks = 2;
    int64_t maxSlot = numBlocks * blockSize;

    size_t keySize = numTokens * numHead * kHeadSize * sizeof(uint8_t);
    size_t valueSize = numTokens * numHead * vHeadSize * sizeof(uint8_t);
    size_t keyCacheSize = numBlocks * blockSize * numHead * kHeadSize * sizeof(uint8_t);
    size_t valueCacheSize = numBlocks * blockSize * numHead * vHeadSize * sizeof(uint8_t);
    size_t slotMappingSize = numTokens * sizeof(int32_t);
    size_t keyScaleSize = numTokens * numHead * sizeof(float);
    size_t keyScaleCacheSize = numBlocks * blockSize * numHead * sizeof(float);

    uint8_t* key = (uint8_t*)AscendC::GmAlloc(keySize);
    uint8_t* value = (uint8_t*)AscendC::GmAlloc(valueSize);
    uint8_t* keyCache = (uint8_t*)AscendC::GmAlloc(keyCacheSize);
    uint8_t* valueCache = (uint8_t*)AscendC::GmAlloc(valueCacheSize);
    uint8_t* slotMapping = (uint8_t*)AscendC::GmAlloc(slotMappingSize);
    uint8_t* keyScale = (uint8_t*)AscendC::GmAlloc(keyScaleSize);
    uint8_t* keyScaleCache = (uint8_t*)AscendC::GmAlloc(keyScaleCacheSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(ScatterPaKvCacheWithKScaleTilingData));

    ScatterPaKvCacheWithKScaleTilingData* tilingData = reinterpret_cast<ScatterPaKvCacheWithKScaleTilingData*>(tiling);
    tilingData->needCoreNum = 1;
    tilingData->numTokens = numTokens;
    tilingData->numHead = numHead;
    tilingData->kHeadSize = kHeadSize;
    tilingData->vHeadSize = vHeadSize;
    tilingData->numBlocks = numBlocks;
    tilingData->blockSize = blockSize;
    tilingData->maxSlot = maxSlot;
    tilingData->keyStride[0] = numHead * kHeadSize;
    tilingData->keyStride[1] = kHeadSize;
    tilingData->keyStride[2] = 1;
    tilingData->valueStride[0] = numHead * vHeadSize;
    tilingData->valueStride[1] = vHeadSize;
    tilingData->valueStride[2] = 1;
    tilingData->keyCacheStride[0] = blockSize * numHead * kHeadSize;
    tilingData->keyCacheStride[1] = kHeadSize;
    tilingData->keyCacheStride[2] = numHead * kHeadSize;
    tilingData->keyCacheStride[3] = 1;
    tilingData->valueCacheStride[0] = blockSize * numHead * vHeadSize;
    tilingData->valueCacheStride[1] = vHeadSize;
    tilingData->valueCacheStride[2] = numHead * vHeadSize;
    tilingData->valueCacheStride[3] = 1;
    tilingData->slotMappingStride[0] = 1;
    tilingData->keyScaleStride[0] = numHead;
    tilingData->keyScaleStride[1] = 1;
    tilingData->keyScaleCacheStride[0] = blockSize * numHead;
    tilingData->keyScaleCacheStride[1] = 1;
    tilingData->keyScaleCacheStride[2] = numHead;
    tilingData->keyScaleCacheStride[3] = 1;

    for (size_t i = 0; i < keySize; ++i) {
        key[i] = static_cast<uint8_t>(i % 256);
    }
    for (size_t i = 0; i < valueSize; ++i) {
        value[i] = static_cast<uint8_t>((i + 128) % 256);
    }
    for (size_t i = 0; i < keyCacheSize; ++i) {
        keyCache[i] = 0;
    }
    for (size_t i = 0; i < valueCacheSize; ++i) {
        valueCache[i] = 0;
    }
    int32_t* slotMappingData = reinterpret_cast<int32_t*>(slotMapping);
    slotMappingData[0] = 0;
    slotMappingData[1] = 16;
    slotMappingData[2] = 1;
    slotMappingData[3] = 17;

    float* keyScaleData = reinterpret_cast<float*>(keyScale);
    for (int64_t i = 0; i < numTokens * numHead; ++i) {
        keyScaleData[i] = 1.0f + static_cast<float>(i) * 0.1f;
    }
    float* keyScaleCacheData = reinterpret_cast<float*>(keyScaleCache);
    for (int64_t i = 0; i < numBlocks * blockSize * numHead; ++i) {
        keyScaleCacheData[i] = 0.0f;
}

    ICPU_SET_TILING_KEY(1000001);
    
    auto scatterPaKvCacheWithKScaleWrapper = [] (GM_ADDR key, GM_ADDR value,
        GM_ADDR key_cache, GM_ADDR value_cache, GM_ADDR slot_mapping,
        GM_ADDR key_scale, GM_ADDR key_scale_cache,
        GM_ADDR key_cache_out, GM_ADDR value_cache_out, GM_ADDR key_scale_cache_out,
        GM_ADDR workspace, GM_ADDR tiling) {
        scatter_pa_kv_cache_with_k_scale<SCATTER_KV_CACHE_SCENE_GENERALIZED>(
            key, value, key_cache, value_cache, slot_mapping, key_scale, key_scale_cache,
            key_cache_out, value_cache_out, key_scale_cache_out, workspace, tiling);
    };
    
    ICPU_RUN_KF(scatterPaKvCacheWithKScaleWrapper, 1,
        key, value, keyCache, valueCache, slotMapping, keyScale, keyScaleCache,
        keyCache, valueCache, keyScaleCache, workspace, tiling);

    AscendC::GmFree(key);
    AscendC::GmFree(value);
    AscendC::GmFree(keyCache);
    AscendC::GmFree(valueCache);
    AscendC::GmFree(slotMapping);
    AscendC::GmFree(keyScale);
    AscendC::GmFree(keyScaleCache);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(scatter_pa_kv_cache_with_k_scale_test, test_case_specialized_fp8_e5m2_int32)
{
    int64_t numTokens = 8;
    int64_t numHead = 4;
    int64_t kHeadSize = 64;
    int64_t vHeadSize = 64;
    int64_t blockSize = 16;
    int64_t numBlocks = 4;
    int64_t maxSlot = numBlocks * blockSize;

    size_t keySize = numTokens * numHead * kHeadSize * sizeof(uint8_t);
    size_t valueSize = numTokens * numHead * vHeadSize * sizeof(uint8_t);
    size_t keyCacheSize = numBlocks * blockSize * numHead * kHeadSize * sizeof(uint8_t);
    size_t valueCacheSize = numBlocks * blockSize * numHead * vHeadSize * sizeof(uint8_t);
    size_t slotMappingSize = numTokens * sizeof(int32_t);
    size_t keyScaleSize = numTokens * numHead * sizeof(float);
    size_t keyScaleCacheSize = numBlocks * blockSize * numHead * sizeof(float);

    uint8_t* key = (uint8_t*)AscendC::GmAlloc(keySize);
    uint8_t* value = (uint8_t*)AscendC::GmAlloc(valueSize);
    uint8_t* keyCache = (uint8_t*)AscendC::GmAlloc(keyCacheSize);
    uint8_t* valueCache = (uint8_t*)AscendC::GmAlloc(valueCacheSize);
    uint8_t* slotMapping = (uint8_t*)AscendC::GmAlloc(slotMappingSize);
    uint8_t* keyScale = (uint8_t*)AscendC::GmAlloc(keyScaleSize);
    uint8_t* keyScaleCache = (uint8_t*)AscendC::GmAlloc(keyScaleCacheSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(ScatterPaKvCacheWithKScaleTilingData));

    ScatterPaKvCacheWithKScaleTilingData* tilingData = reinterpret_cast<ScatterPaKvCacheWithKScaleTilingData*>(tiling);
    tilingData->needCoreNum = 1;
    tilingData->numTokens = numTokens;
    tilingData->numHead = numHead;
    tilingData->kHeadSize = kHeadSize;
    tilingData->vHeadSize = vHeadSize;
    tilingData->numBlocks = numBlocks;
    tilingData->blockSize = blockSize;
    tilingData->maxSlot = maxSlot;
    tilingData->keyStride[0] = numHead * kHeadSize;
    tilingData->keyStride[1] = kHeadSize;
    tilingData->keyStride[2] = 1;
    tilingData->valueStride[0] = numHead * vHeadSize;
    tilingData->valueStride[1] = vHeadSize;
    tilingData->valueStride[2] = 1;
    tilingData->keyCacheStride[0] = blockSize * numHead * kHeadSize;
    tilingData->keyCacheStride[1] = kHeadSize;
    tilingData->keyCacheStride[2] = numHead * kHeadSize;
    tilingData->keyCacheStride[3] = 1;
    tilingData->valueCacheStride[0] = blockSize * numHead * vHeadSize;
    tilingData->valueCacheStride[1] = vHeadSize;
    tilingData->valueCacheStride[2] = numHead * vHeadSize;
    tilingData->valueCacheStride[3] = 1;
    tilingData->slotMappingStride[0] = 1;
    tilingData->keyScaleStride[0] = numHead;
    tilingData->keyScaleStride[1] = 1;
    tilingData->keyScaleCacheStride[0] = blockSize * numHead;
    tilingData->keyScaleCacheStride[1] = 1;
    tilingData->keyScaleCacheStride[2] = numHead;
    tilingData->keyScaleCacheStride[3] = 1;

    for (size_t i = 0; i < keySize; ++i) {
        key[i] = static_cast<uint8_t>(i % 256);
    }
    for (size_t i = 0; i < valueSize; ++i) {
        value[i] = static_cast<uint8_t>((i + 64) % 256);
    }
    for (size_t i = 0; i < keyCacheSize; ++i) {
        keyCache[i] = 0;
    }
    for (size_t i = 0; i < valueCacheSize; ++i) {
        valueCache[i] = 0;
    }
    int32_t* slotMappingData = reinterpret_cast<int32_t*>(slotMapping);
    for (int64_t i = 0; i < numTokens; ++i) {
        slotMappingData[i] = static_cast<int32_t>(i % maxSlot);
    }

    float* keyScaleData = reinterpret_cast<float*>(keyScale);
    for (int64_t i = 0; i < numTokens * numHead; ++i) {
        keyScaleData[i] = 1.0f + static_cast<float>(i) * 0.05f;
    }
    float* keyScaleCacheData = reinterpret_cast<float*>(keyScaleCache);
    for (int64_t i = 0; i < numBlocks * blockSize * numHead; ++i) {
        keyScaleCacheData[i] = 0.0f;
    }

    ICPU_SET_TILING_KEY(1000001);
    
    auto scatterPaKvCacheWithKScaleWrapper = [] (GM_ADDR key, GM_ADDR value,
        GM_ADDR key_cache, GM_ADDR value_cache, GM_ADDR slot_mapping,
        GM_ADDR key_scale, GM_ADDR key_scale_cache,
        GM_ADDR key_cache_out, GM_ADDR value_cache_out, GM_ADDR key_scale_cache_out,
        GM_ADDR workspace, GM_ADDR tiling) {
        scatter_pa_kv_cache_with_k_scale<SCATTER_KV_CACHE_SCENE_SPECIALIZED>(
            key, value, key_cache, value_cache, slot_mapping, key_scale, key_scale_cache,
            key_cache_out, value_cache_out, key_scale_cache_out, workspace, tiling);
    };
    
    ICPU_RUN_KF(scatterPaKvCacheWithKScaleWrapper, 1,
        key, value, keyCache, valueCache, slotMapping, keyScale, keyScaleCache,
        keyCache, valueCache, keyScaleCache, workspace, tiling);

    AscendC::GmFree(key);
    AscendC::GmFree(value);
    AscendC::GmFree(keyCache);
    AscendC::GmFree(valueCache);
    AscendC::GmFree(slotMapping);
    AscendC::GmFree(keyScale);
    AscendC::GmFree(keyScaleCache);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(scatter_pa_kv_cache_with_k_scale_test, test_case_invalid_slot_int32)
{
    int64_t numTokens = 4;
    int64_t numHead = 2;
    int64_t kHeadSize = 128;
    int64_t vHeadSize = 128;
    int64_t blockSize = 16;
    int64_t numBlocks = 2;
    int64_t maxSlot = numBlocks * blockSize;

    size_t keySize = numTokens * numHead * kHeadSize * sizeof(uint8_t);
    size_t valueSize = numTokens * numHead * vHeadSize * sizeof(uint8_t);
    size_t keyCacheSize = numBlocks * blockSize * numHead * kHeadSize * sizeof(uint8_t);
    size_t valueCacheSize = numBlocks * blockSize * numHead * vHeadSize * sizeof(uint8_t);
    size_t slotMappingSize = numTokens * sizeof(int32_t);
    size_t keyScaleSize = numTokens * numHead * sizeof(float);
    size_t keyScaleCacheSize = numBlocks * blockSize * numHead * sizeof(float);

    uint8_t* key = (uint8_t*)AscendC::GmAlloc(keySize);
    uint8_t* value = (uint8_t*)AscendC::GmAlloc(valueSize);
    uint8_t* keyCache = (uint8_t*)AscendC::GmAlloc(keyCacheSize);
    uint8_t* valueCache = (uint8_t*)AscendC::GmAlloc(valueCacheSize);
    uint8_t* slotMapping = (uint8_t*)AscendC::GmAlloc(slotMappingSize);
    uint8_t* keyScale = (uint8_t*)AscendC::GmAlloc(keyScaleSize);
    uint8_t* keyScaleCache = (uint8_t*)AscendC::GmAlloc(keyScaleCacheSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(ScatterPaKvCacheWithKScaleTilingData));

    ScatterPaKvCacheWithKScaleTilingData* tilingData = reinterpret_cast<ScatterPaKvCacheWithKScaleTilingData*>(tiling);
    tilingData->needCoreNum = 1;
    tilingData->numTokens = numTokens;
    tilingData->numHead = numHead;
    tilingData->kHeadSize = kHeadSize;
    tilingData->vHeadSize = vHeadSize;
    tilingData->numBlocks = numBlocks;
    tilingData->blockSize = blockSize;
    tilingData->maxSlot = maxSlot;
    tilingData->keyStride[0] = numHead * kHeadSize;
    tilingData->keyStride[1] = kHeadSize;
    tilingData->keyStride[2] = 1;
    tilingData->valueStride[0] = numHead * vHeadSize;
    tilingData->valueStride[1] = vHeadSize;
    tilingData->valueStride[2] = 1;
    tilingData->keyCacheStride[0] = blockSize * numHead * kHeadSize;
    tilingData->keyCacheStride[1] = kHeadSize;
    tilingData->keyCacheStride[2] = numHead * kHeadSize;
    tilingData->keyCacheStride[3] = 1;
    tilingData->valueCacheStride[0] = blockSize * numHead * vHeadSize;
    tilingData->valueCacheStride[1] = vHeadSize;
    tilingData->valueCacheStride[2] = numHead * vHeadSize;
    tilingData->valueCacheStride[3] = 1;
    tilingData->slotMappingStride[0] = 1;
    tilingData->keyScaleStride[0] = numHead;
    tilingData->keyScaleStride[1] = 1;
    tilingData->keyScaleCacheStride[0] = blockSize * numHead;
    tilingData->keyScaleCacheStride[1] = 1;
    tilingData->keyScaleCacheStride[2] = numHead;
    tilingData->keyScaleCacheStride[3] = 1;

    for (size_t i = 0; i < keyCacheSize; ++i) {
        keyCache[i] = 0;
    }
    for (size_t i = 0; i < valueCacheSize; ++i) {
        valueCache[i] = 0;
    }
    int32_t* slotMappingData = reinterpret_cast<int32_t*>(slotMapping);
    slotMappingData[0] = 0;
    slotMappingData[1] = -1;
    slotMappingData[2] = maxSlot + 10;
    slotMappingData[3] = 5;

    float* keyScaleCacheData = reinterpret_cast<float*>(keyScaleCache);
    for (int64_t i = 0; i < numBlocks * blockSize * numHead; ++i) {
        keyScaleCacheData[i] = 0.0f;
    }

    ICPU_SET_TILING_KEY(1000001);
    
    auto scatterPaKvCacheWithKScaleWrapper = [] (GM_ADDR key, GM_ADDR value,
        GM_ADDR key_cache, GM_ADDR value_cache, GM_ADDR slot_mapping,
        GM_ADDR key_scale, GM_ADDR key_scale_cache,
        GM_ADDR key_cache_out, GM_ADDR value_cache_out, GM_ADDR key_scale_cache_out,
        GM_ADDR workspace, GM_ADDR tiling) {
        scatter_pa_kv_cache_with_k_scale<SCATTER_KV_CACHE_SCENE_SPECIALIZED>(
            key, value, key_cache, value_cache, slot_mapping, key_scale, key_scale_cache,
            key_cache_out, value_cache_out, key_scale_cache_out, workspace, tiling);
    };
    
    ICPU_RUN_KF(scatterPaKvCacheWithKScaleWrapper, 1,
        key, value, keyCache, valueCache, slotMapping, keyScale, keyScaleCache,
        keyCache, valueCache, keyScaleCache, workspace, tiling);

    AscendC::GmFree(key);
    AscendC::GmFree(value);
    AscendC::GmFree(keyCache);
    AscendC::GmFree(valueCache);
    AscendC::GmFree(slotMapping);
    AscendC::GmFree(keyScale);
    AscendC::GmFree(keyScaleCache);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}