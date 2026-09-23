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
#include "inplace_matmul_all_reduce_add_rms_norm_host_ut_param.h"
#include "mc2_infer_shape_case_executor.h"

namespace InplaceMatmulAllReduceAddRmsNormUT {

class InferShapeInplaceMatmulAllReduceAddRmsNormTest
    : public testing::TestWithParam<InplaceMatmulAllReduceAddRmsNormInferShapeUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "InplaceMatmulAllReduceAddRmsNorm InferShapeInplaceMatmulAllReduceAddRmsNormTest SetUp"
                  << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "InplaceMatmulAllReduceAddRmsNorm InferShapeInplaceMatmulAllReduceAddRmsNormTest TearDown"
                  << std::endl;
    }
};

TEST_P(InferShapeInplaceMatmulAllReduceAddRmsNormTest, param)
{
    auto param = GetParam();
    if (param.inputInstance.size() < 8 || param.outputInstance.size() < 2) {
        return;
    }
    std::vector<gert::InfershapeContextPara::TensorDescription> inputTensorDesc;
    if (param.inputInstance[0] == 1)
        inputTensorDesc.emplace_back(param.x1);
    if (param.inputInstance[1] == 1)
        inputTensorDesc.emplace_back(param.x2);
    if (param.inputInstance[2] == 1)
        inputTensorDesc.emplace_back(param.bias);
    if (param.inputInstance[3] == 1)
        inputTensorDesc.emplace_back(param.residual);
    if (param.inputInstance[4] == 1)
        inputTensorDesc.emplace_back(param.gamma);
    if (param.inputInstance[5] == 1)
        inputTensorDesc.emplace_back(param.antiquant_scale);
    if (param.inputInstance[6] == 1)
        inputTensorDesc.emplace_back(param.antiquant_offset);
    if (param.inputInstance[7] == 1)
        inputTensorDesc.emplace_back(param.dequant_scale);
    std::vector<gert::InfershapeContextPara::TensorDescription> outputTensorDesc;
    if (param.outputInstance[0] == 1)
        outputTensorDesc.emplace_back(param.y);
    if (param.outputInstance[1] == 1)
        outputTensorDesc.emplace_back(param.norm_out);
    gert::InfershapeContextPara inferShapeContextPara(
        "InplaceMatmulAllReduceAddRmsNorm", inputTensorDesc, outputTensorDesc,
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

INSTANTIATE_TEST_SUITE_P(InplaceMatmulAllReduceAddRmsNorm, InferShapeInplaceMatmulAllReduceAddRmsNormTest,
                         testing::ValuesIn(GetCasesFromCsv<InplaceMatmulAllReduceAddRmsNormInferShapeUtParam>(
                             ReplaceFileExtension2Csv(__FILE__))),
                         PrintCaseInfoString<InplaceMatmulAllReduceAddRmsNormInferShapeUtParam>);

} // namespace InplaceMatmulAllReduceAddRmsNormUT
