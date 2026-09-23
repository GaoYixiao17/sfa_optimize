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
 * \file test_matmul_swiglu_kernel_run.cpp
 * \brief MatmulSwiglu op_kernel smoke UT —— 真实调用 kernel(ICPU_RUN_KF)。
 *
 * 目的: 真跑一遍 kernel 计算路径(tiling 解析、双 matmul 对象、GetTensorC->UB、
 *       SwiGLU epilogue、DataCopyPad 写回), 校验其能编译并执行不崩溃。
 *
 * 为何不做逐元素 golden 比对:
 *   本 kernel 基于 Matmul 高阶 API。tikicpulib 的 CPU mock 下 matmul 的 GetTensorC 不能
 *   可靠/确定地写满 L0C->UB 输出(实测同一 kernel 多次运行残留元素数在 2~100+ 间波动),
 *   因此逐元素比对不稳定。这与本仓库 matmul 类 kernel UT 的惯例一致
 *   (见 matmul/mat_mul_v3/tests/ut/op_kernel/test_mat_mul_v3.cpp 亦只跑不校验输出)。
 *   kernel 的逐元素精度由真机 ST / examples(PreSmoke) 覆盖; 数值参考本身的正确性见
 *   test_matmul_swiglu_kernel_golden.cpp。
 *
 * 注意: kernel 默认 DTYPE_X=half(见 op_kernel/matmul_swiglu.cpp 兜底宏), 故走 fp16 路径。
 *       TCubeTiling 各字段按 mat_mul_v3 UT 范式手填。
 */

#include <vector>
#include <cstdint>
#include <cstring>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "matmul_swiglu_tiling_def.h"
#include "data_utils.h"

using namespace std;

extern "C" __global__ __aicore__ void matmul_swiglu(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y,
                                                       GM_ADDR workspace, GM_ADDR tiling);

namespace {
// 手填单边 [M,N,K] 的 TCubeTiling(单块, base 覆盖整个 shape)。字段参照 mat_mul_v3 UT。
void FillCubeTiling(TCubeTiling& t, int64_t M, int64_t N, int64_t K)
{
    t.usedCoreNum = 1;
    t.M = static_cast<int32_t>(M);
    t.N = static_cast<int32_t>(N);
    t.Ka = static_cast<int32_t>(K);
    t.Kb = static_cast<int32_t>(K);
    t.singleCoreM = static_cast<int32_t>(M);
    t.singleCoreN = static_cast<int32_t>(N);
    t.singleCoreK = static_cast<int32_t>(K);
    t.baseM = static_cast<int32_t>(M);
    t.baseN = static_cast<int32_t>(N);
    t.baseK = static_cast<int32_t>(K);
    t.depthA1 = 2;
    t.depthB1 = 2;
    t.stepM = 1;
    t.stepN = 1;
    t.stepKa = 1;
    t.stepKb = 1;
    t.isBias = 0;
    t.transLength = 0;
    t.iterateOrder = 0;
    t.shareL1Size = 512 * 1024;
    t.shareL0CSize = 128 * 1024;
    t.shareUbSize = 0;
    t.batchM = 1;
    t.batchN = 1;
    t.singleBatchM = 1;
    t.singleBatchN = 1;
    t.depthAL1CacheUB = 0;
    t.depthBL1CacheUB = 0;
    t.dbL0A = 2;
    t.dbL0B = 2;
    t.dbL0C = 2;
}

// 构造输入 + 手填 tiling + 真跑 kernel。能执行完不崩溃即视为通过。
void RunKernelSmoke(int64_t M, int64_t K, int64_t N)
{
    const int64_t twoN = 2 * N;
    const size_t xBytes = static_cast<size_t>(M * K) * sizeof(half);
    const size_t wBytes = static_cast<size_t>(K * twoN) * sizeof(half);
    const size_t yBytes = static_cast<size_t>(M * N) * sizeof(half);

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(xBytes);
    uint8_t* wGM = (uint8_t*)AscendC::GmAlloc(wBytes);
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(twoN * sizeof(float));  // 无 bias, 占位
    uint8_t* yGM = (uint8_t*)AscendC::GmAlloc(yBytes);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(optiling::MatmulSwigluTilingData));

    half* xh = reinterpret_cast<half*>(xGM);
    half* wh = reinterpret_cast<half*>(wGM);
    for (int i = 0; i < static_cast<int>(M * K); ++i) {
        xh[i] = static_cast<half>(static_cast<float>((i % 7) - 3) * 0.1f);
    }
    for (int i = 0; i < static_cast<int>(K * twoN); ++i) {
        wh[i] = static_cast<half>(static_cast<float>((i % 5) - 2) * 0.1f);
    }
    memset(biasGM, 0, twoN * sizeof(float));
    memset(yGM, 0, yBytes);

    auto* td = reinterpret_cast<optiling::MatmulSwigluTilingData*>(tiling);
    memset(td, 0, sizeof(optiling::MatmulSwigluTilingData));
    td->hasBias = 0;
    td->tileN = static_cast<uint32_t>(N);  // 小 N: 整行不切分。transpose 经 TilingKey(默认 0=非转置)
    // m/k/2N 由 kernel 从 mmTiling 派生, 故 mmTiling.N 需为 2N(matmul 输出宽度)
    FillCubeTiling(td->mmTiling, M, twoN, K);

    ICPU_RUN_KF(matmul_swiglu, 1, xGM, wGM, biasGM, yGM, workspace, tiling);

    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)wGM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)yGM);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}
}  // namespace

class matmul_swiglu_kernel_run : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "matmul_swiglu_kernel_run SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "matmul_swiglu_kernel_run TearDown\n" << endl; }
};

// packed weight, 无 bias, 单块, fp16: 真跑 kernel 路径, 不崩溃即通过。
TEST_F(matmul_swiglu_kernel_run, smoke_packed_no_bias_16x16x16)
{
    RunKernelSmoke(16, 16, 16);
}
