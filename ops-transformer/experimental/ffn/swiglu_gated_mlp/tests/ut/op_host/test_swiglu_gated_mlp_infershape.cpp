/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Copyright (c) 2026 联通（广东）产业互联网有限公司.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>
#include <iostream>

#include "infershape_test_util.h"
#include "log/log.h"
#include "ut_op_common.h"
#include "../../../op_graph/swiglu_gated_mlp_proto.h"

class SwigluGatedMlpInferShape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SwigluGatedMlpInferShape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SwigluGatedMlpInferShape TearDown" << std::endl;
    }
};

TEST_F(SwigluGatedMlpInferShape, full_mlp_infershape)
{
    ge::op::SwigluGatedMlp op;
    op.UpdateInputDesc("x", create_desc({2, 3}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gate_up_weight", create_desc({3, 8}, ge::DT_FLOAT16));
    op.UpdateInputDesc("down_weight", create_desc({4, 2}, ge::DT_FLOAT16));
    op.SetAttr("cube_math_type", static_cast<int64_t>(1));

    Runtime2TestParam param{{"cube_math_type"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto yDesc = op.GetOutputDesc(0);
    std::vector<int64_t> expectedShape = {2, 2};
    EXPECT_EQ(yDesc.GetShape().GetDims(), expectedShape);

    EXPECT_EQ(InferDataTypeTest(op), ge::GRAPH_SUCCESS);
    yDesc = op.GetOutputDesc(0);
    EXPECT_EQ(yDesc.GetDataType(), ge::DT_FLOAT16);
}

TEST_F(SwigluGatedMlpInferShape, swiglu_stage_infershape)
{
    ge::op::SwigluGatedMlp op;
    op.UpdateInputDesc("x", create_desc({2, 8}, ge::DT_FLOAT));
    op.UpdateInputDesc("gate_up_weight", create_desc({3, 8}, ge::DT_FLOAT));
    op.UpdateInputDesc("down_weight", create_desc({4, 2}, ge::DT_FLOAT));
    op.SetAttr("cube_math_type", static_cast<int64_t>(102));

    Runtime2TestParam param{{"cube_math_type"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto yDesc = op.GetOutputDesc(0);
    std::vector<int64_t> expectedShape = {2, 4};
    EXPECT_EQ(yDesc.GetShape().GetDims(), expectedShape);

    EXPECT_EQ(InferDataTypeTest(op), ge::GRAPH_SUCCESS);
    yDesc = op.GetOutputDesc(0);
    EXPECT_EQ(yDesc.GetDataType(), ge::DT_FLOAT);
}

TEST_F(SwigluGatedMlpInferShape, illegal_gate_up_shape)
{
    ge::op::SwigluGatedMlp op;
    op.UpdateInputDesc("x", create_desc({2, 3}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gate_up_weight", create_desc({3, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("down_weight", create_desc({4, 2}, ge::DT_FLOAT16));
    op.SetAttr("cube_math_type", static_cast<int64_t>(1));

    Runtime2TestParam param{{"cube_math_type"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}
