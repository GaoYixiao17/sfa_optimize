/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_inplace_partial_rotary_mul_grad_tiling.cpp
 * @brief OpHost UT for InplacePartialRotaryMulGrad tiling (Ascend950, RegBase)
 *
 * Coverage per template (in priority order):
 *   ABA&BA (10000): BNSD  → TilingKey ABA=201 / BA=202
 *   BAB    (20000): BSND  → TilingKey BAB=203
 *   AB     (25000): SBND  → TilingKey AB=204
 *   A&B    (40000): NO_BROADCAST / BROADCAST_BSN → TilingKey A=205 / B=206
 *
 * Dtype combinations: fp16+fp16, fp32+fp32, fp16+fp32, bf16+bf16
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/inplace_partial_rotary_mul_grad_regbase_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class InplacePartialRotaryMulGradTilingTest : public testing::Test {
protected:
    static constexpr uint64_t k950CoreNum = 64;
    static constexpr uint64_t k950UbSize = 253952; // 248KB
    static constexpr const char *kSocVersion = "Ascend950";
    optiling::InplacePartialRotaryMulGradCompileInfo compileInfo_ = {};

    static void SetUpTestCase()
    {
        cout << "InplacePartialRotaryMulGradTilingTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "InplacePartialRotaryMulGradTilingTest TearDown" << endl;
    }
};

// ==================== BAB (203): BSND, cosb_==1 ====================
TEST_F(InplacePartialRotaryMulGradTilingTest, BAB_BSND_fp32)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dy
            {{{1, 2, 1, 4}, {1, 2, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // cos
            {{{1, 2, 1, 4}, {1, 2, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // sin
        },
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dy out
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({2, 6})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 203;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

TEST_F(InplacePartialRotaryMulGradTilingTest, BAB_BSND_fp16_fp32_mixed)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 2, 1, 4}, {1, 2, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 2, 1, 4}, {1, 2, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 2, 1, 0}, {1, 2, 1, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 2, 1, 0}, {1, 2, 1, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({2, 6})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 203;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// ==================== AB (204): SBND ====================
TEST_F(InplacePartialRotaryMulGradTilingTest, AB_SBND_fp32)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dy [S,B,N,D]
            {{{2, 1, 1, 4}, {2, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // cos [S,1,1,cosD]
            {{{2, 1, 1, 4}, {2, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // sin
        },
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 1, 1, 0}, {2, 1, 1, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 1, 1, 0}, {2, 1, 1, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({2, 6})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 204;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// ==================== ABA (201): BNSD, cosb_!=1 ====================
TEST_F(InplacePartialRotaryMulGradTilingTest, ABA_BNSD_fp32)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dy [B,N,S,D]
            {{{2, 1, 2, 4}, {2, 1, 2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // cos [B,1,S,cosD]
            {{{2, 1, 2, 4}, {2, 1, 2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // sin
        },
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 1, 2, 0}, {2, 1, 2, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 1, 2, 0}, {2, 1, 2, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({2, 6})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 201;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// ==================== BA (202): BNSD, cosb_==1 ====================
TEST_F(InplacePartialRotaryMulGradTilingTest, BA_BNSD_fp32)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dy [B,N,S,D]
            {{{1, 1, 2, 4}, {1, 1, 2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // cos [1,1,S,cosD]
            {{{1, 1, 2, 4}, {1, 1, 2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // sin
        },
        {
            {{{2, 2, 2, 8}, {2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 2, 0}, {1, 1, 2, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 2, 0}, {1, 1, 2, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({2, 6})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 202;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// ==================== A (205): NO_BROADCAST ====================
TEST_F(InplacePartialRotaryMulGradTilingTest, A_NO_BROADCAST_fp32)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 4}, {2, 2, 2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dy [B,N,S,D]
            {{{2, 2, 2, 4}, {2, 2, 2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // cos same shape
            {{{2, 2, 2, 4}, {2, 2, 2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // sin
        },
        {
            {{{2, 2, 2, 4}, {2, 2, 2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 2, 0}, {2, 2, 2, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 2, 0}, {2, 2, 2, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 4})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 205;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}

// ==================== B (206): BROADCAST_BSN ====================
TEST_F(InplacePartialRotaryMulGradTilingTest, B_BROADCAST_BSN_fp32)
{
    gert::TilingContextPara tilingPara(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 2, 2, 4}, {2, 2, 2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dy [B,N,S,D]
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // cos [1,1,1,D]
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // sin
        },
        {
            {{{2, 2, 2, 4}, {2, 2, 2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 0}, {1, 1, 1, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 0}, {1, 1, 1, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 4})},
        },
        &compileInfo_, kSocVersion, k950CoreNum, k950UbSize);

    uint64_t expectTilingKey = 206;
    vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingPara, ge::GRAPH_SUCCESS, expectTilingKey, "", expectWorkspaces);
}
