/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <gtest/gtest.h>
#include "mc2_infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace MoeDistributeCombineTeardownInferShapeUT {

class MoeDistributeCombineTeardownInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeDistributeCombineTeardownInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeDistributeCombineTeardownInfershape TearDown" << std::endl;
    }
};

TEST_F(MoeDistributeCombineTeardownInfershape, InferShape_fp16_normal_2d)
{
    gert::StorageShape expandXShape = {{48, 4096}, {}};
    gert::StorageShape quantExpandXShape = {{48, 4096}, {}};
    gert::StorageShape expertIdsShape = {{8, 6}, {}};
    gert::StorageShape expandIdxShape = {{48}, {}};
    gert::StorageShape expertScalesShape = {{8, 6}, {}};
    gert::StorageShape commCmdInfoShape = {{128}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeCombineTeardown",
        {{expandXShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {quantExpandXShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {expandIdxShape, ge::DT_INT32, ge::FORMAT_ND},
         {expertScalesShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {commCmdInfoShape, ge::DT_INT32, ge::FORMAT_ND}},
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep",
          Ops::Transformer::AnyValue::CreateFrom<std::string>("moe_distribute_combine_teardown_test_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> xOutputShape = {{8, 4096}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, xOutputShape);
}

TEST_F(MoeDistributeCombineTeardownInfershape, InferShape_bf16_normal_2d)
{
    gert::StorageShape expandXShape = {{64, 7168}, {}};
    gert::StorageShape quantExpandXShape = {{64, 7168}, {}};
    gert::StorageShape expertIdsShape = {{8, 8}, {}};
    gert::StorageShape expandIdxShape = {{64}, {}};
    gert::StorageShape expertScalesShape = {{8, 8}, {}};
    gert::StorageShape commCmdInfoShape = {{128}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeCombineTeardown",
        {{expandXShape, ge::DT_BF16, ge::FORMAT_ND},
         {quantExpandXShape, ge::DT_BF16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {expandIdxShape, ge::DT_INT32, ge::FORMAT_ND},
         {expertScalesShape, ge::DT_BF16, ge::FORMAT_ND},
         {commCmdInfoShape, ge::DT_INT32, ge::FORMAT_ND}},
        {
            {{}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"group_ep",
          Ops::Transformer::AnyValue::CreateFrom<std::string>("moe_distribute_combine_teardown_test_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> xOutputShape = {{8, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, xOutputShape);
}

TEST_F(MoeDistributeCombineTeardownInfershape, InferShape_globalBs_nonzero)
{
    gert::StorageShape expandXShape = {{128, 7168}, {}};
    gert::StorageShape quantExpandXShape = {{128, 7168}, {}};
    gert::StorageShape expertIdsShape = {{8, 8}, {}};
    gert::StorageShape expandIdxShape = {{64}, {}};
    gert::StorageShape expertScalesShape = {{8, 8}, {}};
    gert::StorageShape commCmdInfoShape = {{2080}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeCombineTeardown",
        {{expandXShape, ge::DT_BF16, ge::FORMAT_ND},
         {quantExpandXShape, ge::DT_BF16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {expandIdxShape, ge::DT_INT32, ge::FORMAT_ND},
         {expertScalesShape, ge::DT_BF16, ge::FORMAT_ND},
         {commCmdInfoShape, ge::DT_INT32, ge::FORMAT_ND}},
        {
            {{}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("hccl_world_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> xOutputShape = {{8, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, xOutputShape);
}

TEST_F(MoeDistributeCombineTeardownInfershape, InferDtype_fp16)
{
    ge::DataType expandXType = ge::DT_FLOAT16;
    ge::DataType quantExpandXType = ge::DT_FLOAT16;
    ge::DataType expertIdsType = ge::DT_INT32;
    ge::DataType expandIdxType = ge::DT_INT32;
    ge::DataType expertScalesType = ge::DT_FLOAT16;
    ge::DataType commCmdInfoType = ge::DT_INT32;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(6, 1)
                             .InputDataTypes({&expandXType, &quantExpandXType, &expertIdsType, &expandIdxType,
                                              &expertScalesType, &commCmdInfoType})
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .Build();
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("MoeDistributeCombineTeardown")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST_F(MoeDistributeCombineTeardownInfershape, InferDtype_bf16)
{
    ge::DataType expandXType = ge::DT_BF16;
    ge::DataType quantExpandXType = ge::DT_BF16;
    ge::DataType expertIdsType = ge::DT_INT32;
    ge::DataType expandIdxType = ge::DT_INT32;
    ge::DataType expertScalesType = ge::DT_BF16;
    ge::DataType commCmdInfoType = ge::DT_INT32;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(6, 1)
                             .InputDataTypes({&expandXType, &quantExpandXType, &expertIdsType, &expandIdxType,
                                              &expertScalesType, &commCmdInfoType})
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .Build();
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("MoeDistributeCombineTeardown")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_BF16);
}

TEST_F(MoeDistributeCombineTeardownInfershape, InferShape_expandX_1dim)
{
    gert::StorageShape expandXShape = {{4096}, {}};
    gert::StorageShape quantExpandXShape = {{4096, 6144}, {}};
    gert::StorageShape expertIdsShape = {{8, 6}, {}};
    gert::StorageShape expandIdxShape = {{48}, {}};
    gert::StorageShape expertScalesShape = {{8, 6}, {}};
    gert::StorageShape commCmdInfoShape = {{1568}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeCombineTeardown",
        {{expandXShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {quantExpandXShape, ge::DT_INT8, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {expandIdxShape, ge::DT_INT32, ge::FORMAT_ND},
         {expertScalesShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {commCmdInfoShape, ge::DT_INT32, ge::FORMAT_ND}},
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep",
          Ops::Transformer::AnyValue::CreateFrom<std::string>("moe_distribute_combine_teardown_test_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> xOutputShape = {{8, -1}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, xOutputShape);
}

TEST_F(MoeDistributeCombineTeardownInfershape, InferShape_expertIds_1dim)
{
    gert::StorageShape expandXShape = {{96, 4096}, {}};
    gert::StorageShape quantExpandXShape = {{96, 6144}, {}};
    gert::StorageShape expertIdsShape = {{48}, {}};
    gert::StorageShape expandIdxShape = {{48}, {}};
    gert::StorageShape expertScalesShape = {{8, 6}, {}};
    gert::StorageShape commCmdInfoShape = {{1568}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeCombineTeardown",
        {{expandXShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {quantExpandXShape, ge::DT_INT8, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {expandIdxShape, ge::DT_INT32, ge::FORMAT_ND},
         {expertScalesShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {commCmdInfoShape, ge::DT_INT32, ge::FORMAT_ND}},
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep",
          Ops::Transformer::AnyValue::CreateFrom<std::string>("moe_distribute_combine_teardown_test_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> xOutputShape = {{-1, 4096}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, xOutputShape);
}

TEST_F(MoeDistributeCombineTeardownInfershape, InferShape_debug_log)
{
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "0", 1);
    gert::StorageShape expandXShape = {{48, 4096}, {}};
    gert::StorageShape quantExpandXShape = {{48, 4096}, {}};
    gert::StorageShape expertIdsShape = {{8, 6}, {}};
    gert::StorageShape expandIdxShape = {{48}, {}};
    gert::StorageShape expertScalesShape = {{8, 6}, {}};
    gert::StorageShape commCmdInfoShape = {{128}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "MoeDistributeCombineTeardown",
        {{expandXShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {quantExpandXShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
         {expandIdxShape, ge::DT_INT32, ge::FORMAT_ND},
         {expertScalesShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {commCmdInfoShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"group_ep",
          Ops::Transformer::AnyValue::CreateFrom<std::string>("moe_distribute_combine_teardown_test_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> xOutputShape = {{8, 4096}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, xOutputShape);
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "3", 1);
}

} // namespace MoeDistributeCombineTeardownInferShapeUT
