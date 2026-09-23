/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

using namespace std;
class InplacePartialRotaryMulGradInferShape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "InplacePartialRotaryMulGrad InferShape Test SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "InplacePartialRotaryMulGrad InferShape Test TearDown" << std::endl;
    }
};

// ==================== Basic: BSND, 4 inputs (x optional present), fp16 ====================
TEST_F(InplacePartialRotaryMulGradInferShape, basic_BSND_with_x_fp16)
{
    gert::InfershapeContextPara ctx(
        "InplacePartialRotaryMulGrad",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // dy
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // cos
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // sin
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // dy out
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 64})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 64, 2, 64}};
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== Without x (3 inputs), fp32 ====================
TEST_F(InplacePartialRotaryMulGradInferShape, basic_no_x_fp32)
{
    gert::InfershapeContextPara ctx(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 32, 4, 128}, {2, 32, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dy
            {{{1, 32, 1, 64}, {1, 32, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},   // cos
            {{{1, 32, 1, 64}, {1, 32, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},   // sin
            // x absent
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({32, 96})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 32, 4, 128}};
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== SBND layout, fp32 ====================
TEST_F(InplacePartialRotaryMulGradInferShape, SBND_fp32)
{
    gert::InfershapeContextPara ctx(
        "InplacePartialRotaryMulGrad",
        {
            {{{4, 2, 8, 128}, {4, 2, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dy [S,B,N,D]
            {{{4, 1, 1, 64}, {4, 1, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},   // cos [S,1,1,cosD]
            {{{4, 1, 1, 64}, {4, 1, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},   // sin
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 64})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 2, 8, 128}};
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== BNSD layout, fp16 ====================
TEST_F(InplacePartialRotaryMulGradInferShape, BNSD_fp16)
{
    gert::InfershapeContextPara ctx(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 8, 4, 128}, {2, 8, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // dy [B,N,S,D]
            {{{2, 1, 4, 64}, {2, 1, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // cos [B,1,S,cosD]
            {{{2, 1, 4, 64}, {2, 1, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // sin
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({32, 96})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 8, 4, 128}};
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== NO_BROADCAST, bf16 ====================
TEST_F(InplacePartialRotaryMulGradInferShape, NO_BROADCAST_bf16)
{
    gert::InfershapeContextPara ctx(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 4, 8, 64}, {2, 4, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND}, // dy
            {{{2, 4, 8, 64}, {2, 4, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND}, // cos same shape
            {{{2, 4, 8, 64}, {2, 4, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND}, // sin
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 64})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 4, 8, 64}};
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== BROADCAST_BSN, fp32 ====================
TEST_F(InplacePartialRotaryMulGradInferShape, BROADCAST_BSN_fp32)
{
    gert::InfershapeContextPara ctx(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 4, 8, 128}, {2, 4, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dy
            {{{1, 1, 1, 128}, {1, 1, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}, // cos [1,1,1,D]
            {{{1, 1, 1, 128}, {1, 1, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}, // sin
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 128})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 4, 8, 128}};
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== Mixed precision: dy=fp16, cos=fp32 ====================
TEST_F(InplacePartialRotaryMulGradInferShape, mixed_fp16_fp32)
{
    gert::InfershapeContextPara ctx(
        "InplacePartialRotaryMulGrad",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // dy
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},   // cos
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},   // sin
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 64})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 64, 2, 64}};
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== Mixed precision: dy=fp32, cos=fp16 ====================
TEST_F(InplacePartialRotaryMulGradInferShape, mixed_fp32_fp16)
{
    gert::InfershapeContextPara ctx(
        "InplacePartialRotaryMulGrad",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dy
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // cos
            {{{1, 64, 1, 64}, {1, 64, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // sin
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({0, 64})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 64, 2, 64}};
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== Empty slice (valid, no computation) ====================
TEST_F(InplacePartialRotaryMulGradInferShape, empty_slice)
{
    gert::InfershapeContextPara ctx(
        "InplacePartialRotaryMulGrad",
        {
            {{{2, 32, 4, 128}, {2, 32, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 32, 1, 0}, {2, 32, 1, 0}}, ge::DT_FLOAT, ge::FORMAT_ND}, // cos D=0 (empty)
            {{{2, 32, 1, 0}, {2, 32, 1, 0}}, ge::DT_FLOAT, ge::FORMAT_ND}, // sin D=0
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"partial_slice", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({64, 64})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 32, 4, 128}};
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, expectOutputShape);
}
