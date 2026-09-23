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
#include "matmul_all_reduce_add_rms_norm_host_ut_param.h"
#include "mc2_infer_shape_case_executor.h"

namespace MatmulAllReduceAddRmsNormUT {

class InferShapeMatmulAllReduceAddRmsNormTest
    : public testing::TestWithParam<MatmulAllReduceAddRmsNormInferShapeUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulAllReduceAddRmsNorm InferShape SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "MatmulAllReduceAddRmsNorm InferShape TearDown" << std::endl;
    }
};

TEST_P(InferShapeMatmulAllReduceAddRmsNormTest, param)
{
    auto param = GetParam();
    std::vector<gert::InfershapeContextPara::TensorDescription> inputTensorDesc;
    if (param.inputInstance.size() > 0 && param.inputInstance[0] == 1)
        inputTensorDesc.emplace_back(param.x1);
    if (param.inputInstance.size() > 1 && param.inputInstance[1] == 1)
        inputTensorDesc.emplace_back(param.x2);
    if (param.inputInstance.size() > 2 && param.inputInstance[2] == 1)
        inputTensorDesc.emplace_back(param.bias);
    if (param.inputInstance.size() > 3 && param.inputInstance[3] == 1)
        inputTensorDesc.emplace_back(param.residual);
    if (param.inputInstance.size() > 4 && param.inputInstance[4] == 1)
        inputTensorDesc.emplace_back(param.gamma);
    if (param.inputInstance.size() > 5 && param.inputInstance[5] == 1)
        inputTensorDesc.emplace_back(param.antiquant_scale);
    if (param.inputInstance.size() > 6 && param.inputInstance[6] == 1)
        inputTensorDesc.emplace_back(param.antiquant_offset);
    if (param.inputInstance.size() > 7 && param.inputInstance[7] == 1)
        inputTensorDesc.emplace_back(param.dequant_scale);
    std::vector<gert::InfershapeContextPara::TensorDescription> outputTensorDesc;
    if (param.outputInstance.size() > 0 && param.outputInstance[0] == 1)
        outputTensorDesc.emplace_back(param.y);
    if (param.outputInstance.size() > 1 && param.outputInstance[1] == 1)
        outputTensorDesc.emplace_back(param.norm_out);

    gert::InfershapeContextPara inferShapeContextPara(
        "MatmulAllReduceAddRmsNorm", inputTensorDesc, outputTensorDesc,
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.group)},
         {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.reduce_op)},
         {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_trans_a)},
         {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_trans_b)},
         {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.comm_turn)},
         {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.antiquant_group_size)},
         {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(param.epsilon)}},
        param.inputInstance, param.outputInstance);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", param.ranksize}};
    Mc2ExecuteTestCase(inferShapeContextPara, hcomTopologyMockValues, param.expectResult, param.expectOutputShape);
}

INSTANTIATE_TEST_SUITE_P(
    MatmulAllReduceAddRmsNorm, InferShapeMatmulAllReduceAddRmsNormTest,
    testing::ValuesIn(GetCasesFromCsv<MatmulAllReduceAddRmsNormInferShapeUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<MatmulAllReduceAddRmsNormInferShapeUtParam>);

} // namespace MatmulAllReduceAddRmsNormUT
