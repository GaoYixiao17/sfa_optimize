/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class FfnWorkerBatchingInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FfnWorkerBatchingInfershapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FfnWorkerBatchingInfershapeTest TearDown" << std::endl;
    }
};

TEST_F(FfnWorkerBatchingInfershapeTest, ffn_worker_batching_infershape_test01)
{
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};

    gert::InfershapeContextPara infershapeContextPara(
        "FfnWorkerBatching",
        {// input
         {schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND}},
        {
            // output
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},   // group_list
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // session_ids
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // micro_batch_ids
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // token_ids
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // expert_offsets
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dynamic_scale
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},   // actual_token_num
        },
        {{"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"max_out_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({16, 8, 9, 4096})},
         {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layer_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1152, 4096}, {8, 2}, {1152}, {1152},
                                                           {1152},       {1152}, {1152}, {1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ---------------------------------------------------------------------------------------------------------------------
// 迭代一 A5(arch35/ascend950) infershape 规格覆盖
//
// InferShape 与代次无关（A2/A3/A5 共用 InferShape4FfnWorkerBatching，不改），故不区分 socVersion。
// 这些用例覆盖 A5 对外 shape 规格：Y = A*BS*K、H = max_out_shape[3]、group_list = {expert_num, 2}、
// 其余 5 个索引输出与 dynamic_scale = {Y}、actual_token_num = {1}。规格边界对齐 spec.yaml
// （A≤1024、K≤64、expert_num≤8192）。
// ---------------------------------------------------------------------------------------------------------------------

// A5 规格：最大 session 数（A=1024）+ 单 topK（K=1），H=2048 -> Y=1024*1*1=1024。
TEST_F(FfnWorkerBatchingInfershapeTest, ffn_worker_batching_infershape_arch35_max_session)
{
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};

    gert::InfershapeContextPara infershapeContextPara(
        "FfnWorkerBatching",
        {// input
         {schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND}},
        {
            // output
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},   // group_list
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // session_ids
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // micro_batch_ids
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // token_ids
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},   // expert_offsets
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},   // dynamic_scale
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},   // actual_token_num
        },
        {{"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
         {"max_out_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1024, 1, 1, 2048})},
         {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"layer_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});
    // Y=1024, H=2048; group_list={256,2}
    std::vector<std::vector<int64_t>> expectOutputShape = {{1024, 2048}, {256, 2}, {1024}, {1024},
                                                           {1024},       {1024},   {1024}, {1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// A5 规格：最大 topK（K=64），A=16、BS=8、H=4096 -> Y=16*8*64=8192。
TEST_F(FfnWorkerBatchingInfershapeTest, ffn_worker_batching_infershape_arch35_max_topk)
{
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};

    gert::InfershapeContextPara infershapeContextPara(
        "FfnWorkerBatching",
        {// input
         {schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND}},
        {
            // output
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},  // y (token_dtype=1)
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND}, // group_list
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // session_ids
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // micro_batch_ids
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // token_ids
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // expert_offsets
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dynamic_scale
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND}, // actual_token_num
        },
        {{"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"max_out_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({16, 8, 64, 4096})},
         {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layer_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});
    // Y=8192, H=4096; group_list={8,2}
    std::vector<std::vector<int64_t>> expectOutputShape = {{8192, 4096}, {8, 2}, {8192}, {8192},
                                                           {8192},       {8192}, {8192}, {1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
