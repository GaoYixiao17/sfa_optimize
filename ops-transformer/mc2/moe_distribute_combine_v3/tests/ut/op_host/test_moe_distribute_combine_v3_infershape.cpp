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
 * \file test_moe_distribute_combine_v3.cpp
 * \brief infershape ut
 */

#include <iostream>
#include <gtest/gtest.h>
#include "mc2_infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace MoeDistributeCombineV3InfershapeUT {

class MoeDistributeCombineV3Infershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeDistributeCombineV3Infershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeDistributeCombineV3Infershape TearDown" << std::endl;
    }
};

TEST_F(MoeDistributeCombineV3Infershape, InferShape0)
{
    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape expandXShape = {{576, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};
    gert::StorageShape assistInfoForCombineShape = {{32 * 8}, {}};
    gert::StorageShape epSendCountsShape = {{288}, {}};
    gert::StorageShape expertScalesShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeCombineV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {expandXShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {assistInfoForCombineShape, ge::DT_INT32, ge::FORMAT_ND},
         {epSendCountsShape, ge::DT_INT32, ge::FORMAT_ND},
         {expertScalesShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> xOutputShape = {{32, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, xOutputShape);
}

TEST_F(MoeDistributeCombineV3Infershape, InferShape1DExpertIds)
{
    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape expandXShape = {{576, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32 * 8}, {}};
    gert::StorageShape assistInfoForCombineShape = {{32 * 8}, {}};
    gert::StorageShape epSendCountsShape = {{288}, {}};
    gert::StorageShape expertScalesShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeCombineV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {expandXShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {assistInfoForCombineShape, ge::DT_INT32, ge::FORMAT_ND},
         {epSendCountsShape, ge::DT_INT32, ge::FORMAT_ND},
         {expertScalesShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> xOutputShape = {{-1, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, xOutputShape);
}

TEST_F(MoeDistributeCombineV3Infershape, InferShape1DExpandX)
{
    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape expandXShape = {{576 * 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};
    gert::StorageShape assistInfoForCombineShape = {{32 * 8}, {}};
    gert::StorageShape epSendCountsShape = {{288}, {}};
    gert::StorageShape expertScalesShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeCombineV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {expandXShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {assistInfoForCombineShape, ge::DT_INT32, ge::FORMAT_ND},
         {epSendCountsShape, ge::DT_INT32, ge::FORMAT_ND},
         {expertScalesShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> xOutputShape = {{32, -1}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, xOutputShape);
}

TEST_F(MoeDistributeCombineV3Infershape, InferShapeBfloat16)
{
    gert::StorageShape contextShape = {{1, 2052}, {}};
    gert::StorageShape expandXShape = {{576, 7168}, {}};
    gert::StorageShape expertIdsShape = {{32, 8}, {}};
    gert::StorageShape assistInfoForCombineShape = {{32 * 8}, {}};
    gert::StorageShape epSendCountsShape = {{288}, {}};
    gert::StorageShape expertScalesShape = {{32, 8}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeCombineV3",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {expandXShape, ge::DT_BF16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {assistInfoForCombineShape, ge::DT_INT32, ge::FORMAT_ND},
         {epSendCountsShape, ge::DT_INT32, ge::FORMAT_ND},
         {expertScalesShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"tp_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"tp_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"zero_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"copy_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"const_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> xOutputShape = {{32, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, xOutputShape);
}

TEST_F(MoeDistributeCombineV3Infershape, InferDtype0)
{
    ge::DataType contextType = ge::DT_INT32;
    ge::DataType expandXType = ge::DT_FLOAT16;
    ge::DataType expertIdsType = ge::DT_INT32;
    ge::DataType assistInfoForCombineType = ge::DT_INT32;
    ge::DataType epSendCountsType = ge::DT_INT32;
    ge::DataType expertScalesType = ge::DT_FLOAT;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(6, 1)
                             .InputDataTypes({&contextType, &expandXType, &expertIdsType, &assistInfoForCombineType,
                                              &epSendCountsType, &expertScalesType})
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto opImpl = spaceRegistry->GetOpImpl("MoeDistributeCombineV3");
    ASSERT_NE(opImpl, nullptr);
    auto inferDtypeFunc = opImpl->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST_F(MoeDistributeCombineV3Infershape, InferDtypeBfloat16)
{
    ge::DataType contextType = ge::DT_INT32;
    ge::DataType expandXType = ge::DT_BF16;
    ge::DataType expertIdsType = ge::DT_INT32;
    ge::DataType assistInfoForCombineType = ge::DT_INT32;
    ge::DataType epSendCountsType = ge::DT_INT32;
    ge::DataType expertScalesType = ge::DT_FLOAT;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(6, 1)
                             .InputDataTypes({&contextType, &expandXType, &expertIdsType, &assistInfoForCombineType,
                                              &epSendCountsType, &expertScalesType})
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto opImpl = spaceRegistry->GetOpImpl("MoeDistributeCombineV3");
    ASSERT_NE(opImpl, nullptr);
    auto inferDtypeFunc = opImpl->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_BF16);
}

} // namespace MoeDistributeCombineV3InfershapeUT
