/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"

// 测试类：专门测试 BSASelectBlockMask 的 InferShape 逻辑
class BSASelectBlockMaskInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "--- BSASelectBlockMask InferShape UT SetUp ---" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "--- BSASelectBlockMask InferShape UT TearDown ---" << std::endl;
    }
};

// ============================================================================
// 测试用例 1: BNSD Layout 下的 Shape 推导
// ============================================================================
TEST_F(BSASelectBlockMaskInferShapeTest, infershape_bnsd_layout)
{
    int64_t b = 2, n = 8, s = 256, s_kv = 128, d = 128;

    gert::InfershapeContextPara infershapeContextPara(
        "BSASelectBlockMask",
        {
            // 0: query [B, N, S, D]
            {{{b, n, s, d}, {b, n, s, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key [B, N, S_kv, D]
            {{{b, n, s_kv, d}, {b, n, s_kv, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: block_shape (OPTIONAL)
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
            // 3: post_block_shape (OPTIONAL)
            {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            // 4: actual_seq_lengths (OPTIONAL)
            {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND},
            // 5: actual_seq_lengths_kv (OPTIONAL)
            {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND},
            // 6: actual_block_len_query (OPTIONAL)
            {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            // 7: actual_block_len_key (OPTIONAL)
            {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND}
        },
        {
            // 预期的输出列表占位 (block_sparse_mask_out)
            {{{-1}, {-1}}, ge::DT_INT8, ge::FORMAT_ND}
        },
        {
            // 属性占位 (对齐 OpDef 里的 5 个属性顺序)
            {"q_input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"kv_input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(n)},
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
            {"sparsity", Ops::Transformer::AnyValue::CreateFrom<float>(0.5f)}
        }
    );

    // 断言期待结果：mask_out = [B, N, S, S_kv]
    // 注意：infershape 中 maskShape 的 dim2/dim3 设置为 sqLen/skvLen（原始序列长度），
    //       实际 block 数在 tiling 阶段计算。
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {b, n, s, s_kv}  // 0: block_sparse_mask_out
    };

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ============================================================================
// 测试用例 2: TND Layout 下的 Shape 推导
// ============================================================================
TEST_F(BSASelectBlockMaskInferShapeTest, infershape_tnd_layout)
{
    int64_t total_q = 1024, total_kv = 2048, n = 8, d = 128;
    int64_t batch = 1;

    gert::InfershapeContextPara infershapeContextPara(
        "BSASelectBlockMask",
        {
            // 补齐 8 个输入
            // 0: query [T, N, D]
            {{{total_q, n, d}, {total_q, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 1: key [T_kv, N, D]
            {{{total_kv, n, d}, {total_kv, n, d}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // 2: block_shape (OPTIONAL)
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
            // 3: post_block_shape (OPTIONAL)
            {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            // 4: actual_seq_lengths (OPTIONAL)
            {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
            // 5: actual_seq_lengths_kv (OPTIONAL)
            {{{batch}, {batch}}, ge::DT_INT64, ge::FORMAT_ND},
            // 6: actual_block_len_query (OPTIONAL)
            {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            // 7: actual_block_len_key (OPTIONAL)
            {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND}
        },
        {
            {{{-1}, {-1}}, ge::DT_INT8, ge::FORMAT_ND}
        },
        {
            {"q_input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"kv_input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("TND")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(n)},
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
            {"sparsity", Ops::Transformer::AnyValue::CreateFrom<float>(0.5f)}
        }
    );

    // TND 布局下：batchSize=1, numHeads=n, sqLen=total_q, skvLen=total_kv
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, n, total_q, total_kv}  // 0: block_sparse_mask_out
    };

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ============================================================================
// 测试用例 3: BF16 数据类型
// ============================================================================
TEST_F(BSASelectBlockMaskInferShapeTest, infershape_bf16_dtype)
{
    int64_t b = 1, n = 4, s = 512, s_kv = 512, d = 128;

    gert::InfershapeContextPara infershapeContextPara(
        "BSASelectBlockMask",
        {
            // 0: query [B, N, S, D] BF16
            {{{b, n, s, d}, {b, n, s, d}}, ge::DT_BF16, ge::FORMAT_ND},
            // 1: key [B, N, S_kv, D] BF16
            {{{b, n, s_kv, d}, {b, n, s_kv, d}}, ge::DT_BF16, ge::FORMAT_ND},
            // 2: block_shape (OPTIONAL)
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
            // 3: post_block_shape (OPTIONAL)
            {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            // 4: actual_seq_lengths (OPTIONAL)
            {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND},
            // 5: actual_seq_lengths_kv (OPTIONAL)
            {{{b}, {b}}, ge::DT_INT64, ge::FORMAT_ND},
            // 6: actual_block_len_query (OPTIONAL)
            {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND},
            // 7: actual_block_len_key (OPTIONAL)
            {{{-1}, {-1}}, ge::DT_UNDEFINED, ge::FORMAT_ND}
        },
        {
            {{{-1}, {-1}}, ge::DT_INT8, ge::FORMAT_ND}
        },
        {
            {"q_input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"kv_input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BNSD")},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(n)},
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
            {"sparsity", Ops::Transformer::AnyValue::CreateFrom<float>(0.5f)}
        }
    );

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {b, n, s, s_kv}  // 0: block_sparse_mask_out
    };

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
