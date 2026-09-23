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
 * \file test_matmul_swiglu_kernel_golden.cpp
 * \brief MatmulSwiglu 数值参考(golden) UT
 *
 * kernel 走 Matmul 高阶 API, CPU mock 下 cube 输出不保证逐元素可比, 故按本仓库
 * 惯例(参见 matmul/mat_mul_v3 的 op_kernel UT) kernel 输出精度由 ST/examples 覆盖。
 * 本文件只校验"kernel 须对齐的数值参考"本身: 两条结构不同的实现(融合式 SwigluGolden
 * 与整块 SwigluRefFull)交叉比对, 覆盖 packed/transposed weight、有无 bias、多种 shape
 * 及 SiLU 边界。语义: [gate|up] = x @ weight (+bias), y = SiLU(gate) * up, weight [K, 2N]。
 */

#include <cmath>
#include <vector>
#include <random>

#include <gtest/gtest.h>

namespace {
float Silu(float x)
{
    return x / (1.0f + std::exp(-x));
}

// 参考实现 A: 融合式, 按 (row, col) 直接累加 gate/up 两路点积。
// 对应 kernel 的"按输出块逐列计算 gate/up"思路。
std::vector<float> SwigluGolden(const std::vector<float>& x, const std::vector<float>& weight,
                                const std::vector<float>& bias, int64_t m, int64_t k, int64_t n,
                                bool transposeWeight, bool hasBias)
{
    std::vector<float> y(static_cast<size_t>(m * n), 0.0f);
    for (int64_t row = 0; row < m; ++row) {
        for (int64_t col = 0; col < n; ++col) {
            float gate = 0.0f;
            float up = 0.0f;
            for (int64_t kk = 0; kk < k; ++kk) {
                const float xVal = x[static_cast<size_t>(row * k + kk)];
                if (transposeWeight) {  // weight [2N, K]
                    gate += xVal * weight[static_cast<size_t>(col * k + kk)];
                    up += xVal * weight[static_cast<size_t>((n + col) * k + kk)];
                } else {                // weight [K, 2N]
                    gate += xVal * weight[static_cast<size_t>(kk * 2 * n + col)];
                    up += xVal * weight[static_cast<size_t>(kk * 2 * n + n + col)];
                }
            }
            if (hasBias) {
                gate += bias[static_cast<size_t>(col)];
                up += bias[static_cast<size_t>(n + col)];
            }
            y[static_cast<size_t>(row * n + col)] = Silu(gate) * up;
        }
    }
    return y;
}

// 参考实现 B: 朴素整块, 先算完整 [M, 2N] = x @ weight (+bias), 再切分 gate/up。
// 结构与 A 不同(先整块乘再切分), 作为交叉校验, 避免"自己证明自己"。
std::vector<float> SwigluRefFull(const std::vector<float>& x, const std::vector<float>& weight,
                                 const std::vector<float>& bias, int64_t m, int64_t k, int64_t n,
                                 bool transposeWeight, bool hasBias)
{
    const int64_t twoN = 2 * n;
    std::vector<float> mm(static_cast<size_t>(m * twoN), 0.0f);
    for (int64_t row = 0; row < m; ++row) {
        for (int64_t c2 = 0; c2 < twoN; ++c2) {
            float acc = 0.0f;
            for (int64_t kk = 0; kk < k; ++kk) {
                const float xVal = x[static_cast<size_t>(row * k + kk)];
                const float wVal = transposeWeight
                                       ? weight[static_cast<size_t>(c2 * k + kk)]   // [2N, K]
                                       : weight[static_cast<size_t>(kk * twoN + c2)]; // [K, 2N]
                acc += xVal * wVal;
            }
            if (hasBias) {
                acc += bias[static_cast<size_t>(c2)];
            }
            mm[static_cast<size_t>(row * twoN + c2)] = acc;
        }
    }
    std::vector<float> y(static_cast<size_t>(m * n), 0.0f);
    for (int64_t row = 0; row < m; ++row) {
        for (int64_t col = 0; col < n; ++col) {
            const float gate = mm[static_cast<size_t>(row * twoN + col)];
            const float up = mm[static_cast<size_t>(row * twoN + n + col)];
            y[static_cast<size_t>(row * n + col)] = Silu(gate) * up;
        }
    }
    return y;
}

// 生成随机数据并填好 weight(按 transpose 与否布局)/bias。
void GenData(int64_t m, int64_t k, int64_t n, bool transposeWeight, bool hasBias, uint32_t seed,
             std::vector<float>& x, std::vector<float>& weight, std::vector<float>& bias)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    x.resize(static_cast<size_t>(m * k));
    for (auto& v : x) {
        v = dist(rng);
    }
    weight.resize(static_cast<size_t>(k * 2 * n));  // 元素总数与布局无关, 均为 K*2N
    for (auto& v : weight) {
        v = dist(rng);
    }
    bias.clear();
    if (hasBias) {
        bias.resize(static_cast<size_t>(2 * n));
        for (auto& v : bias) {
            v = dist(rng);
        }
    }
}

void CrossCheck(int64_t m, int64_t k, int64_t n, bool transposeWeight, bool hasBias, uint32_t seed)
{
    std::vector<float> x, weight, bias;
    GenData(m, k, n, transposeWeight, hasBias, seed, x, weight, bias);
    const auto ya = SwigluGolden(x, weight, bias, m, k, n, transposeWeight, hasBias);
    const auto yb = SwigluRefFull(x, weight, bias, m, k, n, transposeWeight, hasBias);
    ASSERT_EQ(ya.size(), yb.size());
    ASSERT_EQ(ya.size(), static_cast<size_t>(m * n));
    for (size_t i = 0; i < ya.size(); ++i) {
        EXPECT_NEAR(ya[i], yb[i], 1e-4f) << "ref mismatch at " << i;
    }
}
}  // namespace

// ---- 精确小用例: 与手算结果比对 (packed weight) ----
TEST(MatmulSwigluKernelGolden, packed_weight_reference)
{
    const int64_t m = 1;
    const int64_t k = 2;
    const int64_t n = 2;
    const std::vector<float> x = {1.0f, 2.0f};
    const std::vector<float> weight = {
        1.0f, 0.0f, 2.0f, 1.0f,
        0.0f, 1.0f, 3.0f, -1.0f,
    };

    const auto y = SwigluGolden(x, weight, {}, m, k, n, false, false);

    ASSERT_EQ(y.size(), 2U);
    EXPECT_NEAR(y[0], Silu(1.0f) * 8.0f, 1e-5f);
    EXPECT_NEAR(y[1], Silu(2.0f) * -1.0f, 1e-5f);
}

// ---- 精确小用例: transposed weight [2N, K] ----
TEST(MatmulSwigluKernelGolden, transposed_weight_reference)
{
    const int64_t m = 1;
    const int64_t k = 2;
    const int64_t n = 2;
    const std::vector<float> x = {1.0f, 2.0f};
    const std::vector<float> weight = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        2.0f, 3.0f,
        1.0f, -1.0f,
    };

    const auto y = SwigluGolden(x, weight, {}, m, k, n, true, false);

    ASSERT_EQ(y.size(), 2U);
    EXPECT_NEAR(y[0], Silu(1.0f) * 8.0f, 1e-5f);
    EXPECT_NEAR(y[1], Silu(2.0f) * -1.0f, 1e-5f);
}

// ---- 精确小用例: 带 bias ----
TEST(MatmulSwigluKernelGolden, packed_weight_with_bias)
{
    const int64_t m = 1;
    const int64_t k = 2;
    const int64_t n = 1;
    const std::vector<float> x = {1.0f, 2.0f};
    // weight [K, 2N] = [2, 2]: 列0=gate, 列1=up
    const std::vector<float> weight = {
        1.0f, 2.0f,   // k=0
        3.0f, 1.0f,   // k=1
    };
    const std::vector<float> bias = {0.5f, -1.0f};  // [gate_bias, up_bias]

    // gate = 1*1 + 2*3 + 0.5 = 7.5 ; up = 1*2 + 2*1 - 1.0 = 3.0
    const auto y = SwigluGolden(x, weight, bias, m, k, n, false, true);
    ASSERT_EQ(y.size(), 1U);
    EXPECT_NEAR(y[0], Silu(7.5f) * 3.0f, 1e-4f);
}

// ---- 两条参考实现交叉校验: 多 shape / transpose / bias ----
TEST(MatmulSwigluKernelGolden, cross_check_packed_no_bias)
{
    CrossCheck(8, 16, 8, /*transpose=*/false, /*bias=*/false, 1u);
    CrossCheck(32, 64, 16, false, false, 2u);
    CrossCheck(5, 7, 3, false, false, 3u);  // 非对齐 shape
}

TEST(MatmulSwigluKernelGolden, cross_check_packed_with_bias)
{
    CrossCheck(8, 16, 8, false, true, 11u);
    CrossCheck(17, 33, 9, false, true, 12u);
}

TEST(MatmulSwigluKernelGolden, cross_check_transposed)
{
    CrossCheck(8, 16, 8, /*transpose=*/true, /*bias=*/false, 21u);
    CrossCheck(16, 24, 12, true, true, 22u);
}

// ---- SiLU 语义边界 ----
TEST(MatmulSwigluKernelGolden, silu_semantics)
{
    EXPECT_NEAR(Silu(0.0f), 0.0f, 1e-6f);
    EXPECT_NEAR(Silu(20.0f), 20.0f, 1e-3f);     // 大正数趋近 x
    EXPECT_NEAR(Silu(-20.0f), 0.0f, 1e-3f);     // 大负数趋近 0
    EXPECT_LT(Silu(-1.0f), 0.0f);               // SiLU 在负区可为负
}