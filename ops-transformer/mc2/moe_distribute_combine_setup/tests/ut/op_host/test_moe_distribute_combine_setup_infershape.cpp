/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_moe_distribute_combine_setup_infershape.cpp
 * \brief infershape ut
 */

#include <iostream>
#include <gtest/gtest.h>
#include "mc2_infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "infer_shape_context_faker.h"

namespace MoeDistributeCombineSetupInferShapeUT {

static const std::string OP_NAME = "MoeDistributeCombineSetup";

class MoeDistributeCombineSetupInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeDistributeCombineSetupInferShapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeDistributeCombineSetupInferShapeTest TearDown" << std::endl;
    }
};

TEST_F(MoeDistributeCombineSetupInferShapeTest, InferShapeNormal1)
{
    gert::StorageShape expandXShape = {{192, 4096}, {}};
    gert::StorageShape expertIdsShape = {{16, 6}, {}};
    gert::StorageShape assistInfoShape = {{24576}, {}};
    gert::StorageShape quantExpandXOutShape = {{192, 6144}, {}};
    gert::StorageShape commCmdInfoOutShape = {{3104}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        OP_NAME,
        {{expandXShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {assistInfoShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{quantExpandXOutShape, ge::DT_INT8, ge::FORMAT_ND}, {commCmdInfoOutShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("test_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{192, 6144}, {3104}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeDistributeCombineSetupInferShapeTest, InferShapeNormal2)
{
    gert::StorageShape expandXShape = {{512, 4096}, {}};
    gert::StorageShape expertIdsShape = {{16, 6}, {}};
    gert::StorageShape assistInfoShape = {{65536}, {}};
    gert::StorageShape quantExpandXOutShape = {{512, 6144}, {}};
    gert::StorageShape commCmdInfoOutShape = {{8320}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        OP_NAME,
        {{expandXShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {assistInfoShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{quantExpandXOutShape, ge::DT_INT8, ge::FORMAT_ND}, {commCmdInfoOutShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("test_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{512, 6144}, {8320}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeDistributeCombineSetupInferShapeTest, InferShapeNormal3)
{
    gert::StorageShape expandXShape = {{192, 7168}, {}};
    gert::StorageShape expertIdsShape = {{16, 8}, {}};
    gert::StorageShape assistInfoShape = {{24576}, {}};
    gert::StorageShape quantExpandXOutShape = {{192, 10752}, {}};
    gert::StorageShape commCmdInfoOutShape = {{3104}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        OP_NAME,
        {{expandXShape, ge::DT_BF16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {assistInfoShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{quantExpandXOutShape, ge::DT_INT8, ge::FORMAT_ND}, {commCmdInfoOutShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("test_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{192, 10752}, {3104}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeDistributeCombineSetupInferShapeTest, InferShapeNormal4)
{
    gert::StorageShape expandXShape = {{512, 7168}, {}};
    gert::StorageShape expertIdsShape = {{16, 8}, {}};
    gert::StorageShape assistInfoShape = {{65536}, {}};
    gert::StorageShape quantExpandXOutShape = {{512, 10752}, {}};
    gert::StorageShape commCmdInfoOutShape = {{8320}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        OP_NAME,
        {{expandXShape, ge::DT_BF16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {assistInfoShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{quantExpandXOutShape, ge::DT_INT8, ge::FORMAT_ND}, {commCmdInfoOutShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("test_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{512, 10752}, {8320}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeDistributeCombineSetupInferShapeTest, InferShapeNormal5)
{
    gert::StorageShape expandXShape = {{96, 4096}, {}};
    gert::StorageShape expertIdsShape = {{8, 6}, {}};
    gert::StorageShape assistInfoShape = {{12288}, {}};
    gert::StorageShape quantExpandXOutShape = {{96, 6144}, {}};
    gert::StorageShape commCmdInfoOutShape = {{1568}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        OP_NAME,
        {{expandXShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {assistInfoShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{quantExpandXOutShape, ge::DT_INT8, ge::FORMAT_ND}, {commCmdInfoOutShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("test_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{96, 6144}, {1568}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeDistributeCombineSetupInferShapeTest, InferShapeNormal6)
{
    gert::StorageShape expandXShape = {{4096, 7168}, {}};
    gert::StorageShape expertIdsShape = {{256, 8}, {}};
    gert::StorageShape assistInfoShape = {{524288}, {}};
    gert::StorageShape quantExpandXOutShape = {{4096, 10752}, {}};
    gert::StorageShape commCmdInfoOutShape = {{65568}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        OP_NAME,
        {{expandXShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {assistInfoShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{quantExpandXOutShape, ge::DT_INT8, ge::FORMAT_ND}, {commCmdInfoOutShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("test_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(512)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{4096, 10752}, {65568}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeDistributeCombineSetupInferShapeTest, InferDataTypeNormal)
{
    auto contextHolder = gert::InferDataTypeContextFaker()
                             .SetOpType(OP_NAME)
                             .NodeIoNum(3, 2)
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl(OP_NAME.c_str());
    ASSERT_NE(opImpl, nullptr);
    auto inferDtypeFunc = opImpl->infer_datatype;
    ASSERT_NE(inferDtypeFunc, nullptr);
    auto ret = inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>());
    ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_INT8);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), ge::DT_INT32);
}

} // namespace MoeDistributeCombineSetupInferShapeUT
