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
#include "mc2_infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "infer_datatype_context_faker.h"

class DistributeBarrierInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "DistributeBarrierInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "DistributeBarrierInfershape TearDown" << std::endl;
    }
};

TEST_F(DistributeBarrierInfershape, InferShape0)
{
    gert::StorageShape xRefShape = {{32, 7168}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "DistributeBarrier", {{xRefShape, ge::DT_FLOAT16, ge::FORMAT_ND}}, {{{}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{32, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(DistributeBarrierInfershape, InferDataType)
{
    auto opImpl = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("DistributeBarrier");
    ASSERT_NE(opImpl, nullptr);
    auto data_type_func = opImpl->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);

    ge::DataType input_0 = ge::DT_FLOAT16;
    ge::DataType output_0 = ge::DT_FLOAT16;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .NodeIoNum(1, 1)
                              .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&input_0})
                              .OutputDataTypes({&output_0})
                              .Build();

    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST_F(DistributeBarrierInfershape, InferDataTypeBf16)
{
    auto opImpl = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("DistributeBarrier");
    ASSERT_NE(opImpl, nullptr);
    auto data_type_func = opImpl->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);

    ge::DataType input_0 = ge::DT_BF16;
    ge::DataType output_0 = ge::DT_BF16;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .NodeIoNum(1, 1)
                              .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&input_0})
                              .OutputDataTypes({&output_0})
                              .Build();

    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
}
