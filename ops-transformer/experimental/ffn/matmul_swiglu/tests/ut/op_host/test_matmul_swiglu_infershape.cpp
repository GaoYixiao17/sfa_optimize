/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>

#include <gtest/gtest.h>

#include "base/registry/op_impl_space_registry_v2.h"
#include "infer_datatype_context_faker.h"
#include "infershape_case_executor.h"

class MatmulSwigluInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MatmulSwigluInfershape SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "MatmulSwigluInfershape TearDown" << std::endl; }
};

namespace {
std::vector<gert::InfershapeContextPara::OpAttr> MakeAttrs(bool transposeWeight)
{
    return {
        {"transpose_weight", Ops::NN::AnyValue::CreateFrom<bool>(transposeWeight)},
    };
}

void ExecuteDataTypeCase(ge::DataType dtype)
{
    auto contextFaker = gert::InferDataTypeContextFaker();
    auto contextHolder = contextFaker.NodeIoNum(2, 1)
                             .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                             .SetOpType("MatmulSwiglu")
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MatmulSwiglu");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(opImpl->infer_datatype(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), dtype);
}
}  // namespace

// x[M,K], weight[K,2N] -> y[M,N]; 这里 M=32,K=128,2N=128 => N=64
TEST_F(MatmulSwigluInfershape, matmul_swiglu_infershape_basic)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MatmulSwiglu",
        {
            {{{32, 128}, {32, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},   // x
            {{{128, 128}, {128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // weight [K, 2N]
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                 // y
        },
        MakeAttrs(false));
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {32, 64},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 动态 M
TEST_F(MatmulSwigluInfershape, matmul_swiglu_infershape_dynamic_m)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MatmulSwiglu",
        {
            {{{-1, 256}, {-1, 256}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{256, 512}, {256, 512}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        MakeAttrs(false));
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, 256},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 不显式传 attr 时使用 op_def 里的默认 transpose_weight=false
TEST_F(MatmulSwigluInfershape, matmul_swiglu_infershape_default_attr)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MatmulSwiglu",
        {
            {{{8, 64}, {8, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 32}, {64, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {8, 16},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// transpose_weight=true 时 weight 形状为 [2N,K]
TEST_F(MatmulSwigluInfershape, matmul_swiglu_infershape_transpose_weight)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MatmulSwiglu",
        {
            {{{16, 128}, {16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256, 128}, {256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        MakeAttrs(true));
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {16, 128},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// x 支持 rank>=2, 输出保留前置维度并替换最后一维为 N
TEST_F(MatmulSwigluInfershape, matmul_swiglu_infershape_high_rank_x)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MatmulSwiglu",
        {
            {{{2, 3, 128}, {2, 3, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{128, 64}, {128, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        MakeAttrs(false));
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 32},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MatmulSwigluInfershape, matmul_swiglu_infershape_odd_two_n_failed)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MatmulSwiglu",
        {
            {{{32, 128}, {32, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{128, 127}, {128, 127}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        MakeAttrs(false));
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(MatmulSwigluInfershape, matmul_swiglu_infershape_invalid_weight_rank_failed)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MatmulSwiglu",
        {
            {{{32, 128}, {32, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{128, 64, 2}, {128, 64, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        MakeAttrs(false));
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(MatmulSwigluInfershape, matmul_swiglu_inferdatatype_follows_x)
{
    ExecuteDataTypeCase(ge::DT_FLOAT16);
    ExecuteDataTypeCase(ge::DT_BF16);
    ExecuteDataTypeCase(ge::DT_FLOAT);
}
