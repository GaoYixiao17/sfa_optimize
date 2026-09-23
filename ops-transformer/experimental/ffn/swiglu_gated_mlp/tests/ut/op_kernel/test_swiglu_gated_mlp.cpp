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

#include <cstdint>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>

#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "swiglu_gated_mlp_tiling_def.h"

using namespace std;

namespace platform_ascendc {
uint32_t PlatformAscendC::GetCoreNumAic() const
{
    return 0;
}

uint32_t PlatformAscendC::GetLibApiWorkSpaceSize() const
{
    return 0;
}

PlatformAscendC *PlatformAscendCManager::platformInfo = nullptr;
std::mutex PlatformAscendCManager::platformInitMtx;

PlatformAscendC *PlatformAscendCManager::PlatformAscendCManagerInit(const char *customSocVersion)
{
    (void)customSocVersion;
    return nullptr;
}
}  // namespace platform_ascendc

extern "C" __global__ __aicore__ void swiglu_gated_mlp(
    GM_ADDR xGm,
    GM_ADDR gateUpWeightGm,
    GM_ADDR downWeightGm,
    GM_ADDR yGm,
    GM_ADDR workspace,
    GM_ADDR tiling);

class swiglu_gated_mlp_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "swiglu_gated_mlp_test SetUp\n" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "swiglu_gated_mlp_test TearDown\n" << endl;
    }
};

template <typename T>
struct DataTypeName {
    static constexpr const char *value = "unknown";
};

template <>
struct DataTypeName<half> {
    static constexpr const char *value = "float16";
};

template <>
struct DataTypeName<float> {
    static constexpr const char *value = "float32";
};

template <>
struct DataTypeName<bfloat16_t> {
    static constexpr const char *value = "bfloat16_t";
};

template <typename T>
float ToFloat(T value)
{
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        return AscendC::ToFloat(value);
    } else {
        return static_cast<float>(value);
    }
}

template <typename T>
T FromFloat(float value)
{
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        return AscendC::ToBfloat16(value);
    } else {
        return static_cast<T>(value);
    }
}

static float Sigmoid(float value)
{
    return 1.0f / (1.0f + std::exp(-value));
}

template <typename T>
void RunSwigluGatedMlpKernelCase(uint64_t tilingKey)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    const char *dtypeStr = DataTypeName<T>::value;
    cout << ">>> Current Test Type: " << dtypeStr << endl;

    constexpr uint64_t totalRows = 2;
    constexpr uint64_t intermediateSize = 16;
    constexpr uint64_t gateUpSize = intermediateSize * 2;
    constexpr uint32_t usedCoreNum = 1;
    constexpr uint32_t baseRows = 1;
    constexpr uint32_t baseCols = 16;

    size_t gateUpByteSize = totalRows * gateUpSize * sizeof(T);
    size_t hiddenByteSize = totalRows * intermediateSize * sizeof(T);
    size_t workspaceByteSize = 32;
    size_t tilingByteSize = sizeof(swiglu_gated_mlp_kernel::SwigluGatedMlpTilingData);

    uint8_t *gateUp = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(gateUpByteSize));
    uint8_t *hidden = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(hiddenByteSize));
    uint8_t *downWeight = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(sizeof(T)));
    uint8_t *workspace = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(workspaceByteSize));
    uint8_t *tiling = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(tilingByteSize));

    memset(gateUp, 0, gateUpByteSize);
    memset(hidden, 0, hiddenByteSize);
    memset(downWeight, 0, sizeof(T));
    memset(workspace, 0, workspaceByteSize);
    memset(tiling, 0, tilingByteSize);

    T *gateUpData = reinterpret_cast<T *>(gateUp);
    T *hiddenData = reinterpret_cast<T *>(hidden);
    float golden[totalRows * intermediateSize] = {};

    for (uint64_t row = 0; row < totalRows; ++row) {
        for (uint64_t col = 0; col < intermediateSize; ++col) {
            const uint64_t gateOffset = row * gateUpSize + col;
            const uint64_t upOffset = gateOffset + intermediateSize;
            const float gateValue = -1.0f + 0.125f * static_cast<float>(gateOffset % 17U);
            const float upValue = 0.5f - 0.0625f * static_cast<float>(upOffset % 13U);
            gateUpData[gateOffset] = FromFloat<T>(gateValue);
            gateUpData[upOffset] = FromFloat<T>(upValue);

            const float gate = ToFloat<T>(gateUpData[gateOffset]);
            const float up = ToFloat<T>(gateUpData[upOffset]);
            golden[row * intermediateSize + col] = ToFloat<T>(FromFloat<T>(gate * Sigmoid(gate) * up));
        }
    }

    auto *tilingData = reinterpret_cast<swiglu_gated_mlp_kernel::SwigluGatedMlpTilingData *>(tiling);
    tilingData->dtypeTag = static_cast<uint32_t>(tilingKey);
    tilingData->stage = WG_MLP_STAGE_SWIGLU;
    tilingData->isDynamicShape = 0;
    tilingData->is32BAligned = 1;
    tilingData->isDoubleBuffer = 1;
    tilingData->usedCoreNum = usedCoreNum;
    tilingData->dtypeSize = sizeof(T);
    tilingData->totalRows = totalRows;
    tilingData->hiddenSize = gateUpSize;
    tilingData->gateUpSize = gateUpSize;
    tilingData->intermediateSize = intermediateSize;
    tilingData->outSize = intermediateSize;
    tilingData->baseRowsPerCore = totalRows;
    tilingData->tailRows = 0;
    tilingData->tileRows = totalRows;
    tilingData->mm1BaseM = 0;
    tilingData->mm1BaseN = 0;
    tilingData->mm1BaseK = 0;
    tilingData->swiBaseRows = baseRows;
    tilingData->swiBaseCols = baseCols;
    tilingData->mm2BaseM = 0;
    tilingData->mm2BaseN = 0;
    tilingData->mm2BaseK = 0;

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(swiglu_gated_mlp, usedCoreNum, gateUp, nullptr, downWeight, hidden, workspace, tiling);

    std::string outputName = std::string(dtypeStr) + "_output_swiglu_gated_mlp.bin";
    cout << "=== Preview first 5 elements for " << outputName << " ===" << endl;
    cout << std::fixed << std::setprecision(30);
    bool comparePassed = true;
    uint64_t firstMismatch = totalRows * intermediateSize;
    const float tolerance = std::is_same_v<T, float> ? 1e-3f : (std::is_same_v<T, half> ? 5e-3f : 5e-2f);
    for (uint64_t idx = 0; idx < totalRows * intermediateSize; ++idx) {
        const float output = ToFloat<T>(hiddenData[idx]);
        const float expected = golden[idx];
        if (idx < 5U) {
            cout << "index:" << idx << ",output:" << output << ",golden:" << expected << endl;
        }
        if (std::fabs(output - expected) > tolerance) {
            comparePassed = false;
            if (firstMismatch == totalRows * intermediateSize) {
                firstMismatch = idx;
            }
        }
    }

    if (comparePassed) {
        cout << "Comparison passed for " << outputName << endl;
    } else {
        cout << "Comparison failed for " << outputName << endl;
        for (uint64_t idx = firstMismatch; idx < totalRows * intermediateSize && idx < firstMismatch + 5U; ++idx) {
            cout << "index:" << idx << ",output:" << ToFloat<T>(hiddenData[idx])
                 << ", golden:" << golden[idx] << endl;
        }
    }
    cout.unsetf(std::ios::floatfield);
    EXPECT_TRUE(comparePassed);

    AscendC::GmFree(gateUp);
    AscendC::GmFree(hidden);
    AscendC::GmFree(downWeight);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(swiglu_gated_mlp_test, test_case_fp16_swiglu_gated_mlp)
{
    RunSwigluGatedMlpKernelCase<half>(WG_MLP_KEY_FP16_SWIGLU);
}

TEST_F(swiglu_gated_mlp_test, test_case_fp32_swiglu_gated_mlp)
{
    RunSwigluGatedMlpKernelCase<float>(WG_MLP_KEY_FP32_SWIGLU);
}

TEST_F(swiglu_gated_mlp_test, test_case_bf16_swiglu_gated_mlp)
{
    RunSwigluGatedMlpKernelCase<bfloat16_t>(WG_MLP_KEY_BF16_SWIGLU);
}
