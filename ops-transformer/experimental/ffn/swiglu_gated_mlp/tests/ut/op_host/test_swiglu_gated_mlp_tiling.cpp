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
#include <map>
#include <string>

#include "array_ops.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "test_cube_util.h"
#include "ut_op_util.h"
#include "../../../op_host/swiglu_gated_mlp_tiling.h"

using namespace ge;
using namespace std;
using namespace ut_util;

class SwigluGatedMlpTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SwigluGatedMlpTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SwigluGatedMlpTiling TearDown" << std::endl;
    }
};

namespace {

constexpr int64_t WG_MLP_STAGE_ATTR_FP16_MM1 = optiling::WG_MLP_STAGE_MM1;
constexpr int64_t WG_MLP_STAGE_ATTR_FP32_SWIGLU =
    optiling::WG_MLP_STAGE_DTYPE_STRIDE + optiling::WG_MLP_STAGE_SWIGLU;

void InitPlatform(fe::PlatFormInfos& platformInfo)
{
    string compileInfo = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 40}
    })";
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compileInfo.c_str(), socInfos, aicoreSpec, intrinsics);

    platformInfo.Init();
    platformInfo.SetPlatformRes("SoCInfo", socInfos);
    platformInfo.SetPlatformRes("AICoreSpec", aicoreSpec);
    platformInfo.SetCoreNumByCoreType("AICore");
    platformInfo.SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
}

}  // namespace

TEST_F(SwigluGatedMlpTiling, mm1_stage_tiling)
{
    std::string opType("SwigluGatedMlp");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape gateUpWeightShape = {{3, 8}, {3, 8}};
    gert::StorageShape downWeightShape = {{4, 2}, {4, 2}};
    gert::StorageShape yShape = {{2, 8}, {2, 8}};

    fe::PlatFormInfos platformInfo;
    InitPlatform(platformInfo);
    optiling::SwigluGatedMlpCompileInfo compileInfo;

    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    ASSERT_NE(tilingData, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType("SwigluGatedMlp")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &gateUpWeightShape, &downWeightShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"cube_math_type", Ops::NN::AnyValue::CreateFrom<int64_t>(
                                                       WG_MLP_STAGE_ATTR_FP16_MM1)}})
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();

    auto context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(tilingFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetTilingKey(), optiling::WG_MLP_KEY_FP16_MM1);
}

TEST_F(SwigluGatedMlpTiling, swiglu_stage_tiling)
{
    std::string opType("SwigluGatedMlp");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    gert::StorageShape gateUpShape = {{2, 8}, {2, 8}};
    gert::StorageShape gateUpWeightShape = {{3, 8}, {3, 8}};
    gert::StorageShape downWeightShape = {{4, 2}, {4, 2}};
    gert::StorageShape hiddenShape = {{2, 4}, {2, 4}};

    fe::PlatFormInfos platformInfo;
    InitPlatform(platformInfo);
    optiling::SwigluGatedMlpCompileInfo compileInfo;

    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    ASSERT_NE(tilingData, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType("SwigluGatedMlp")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&gateUpShape, &gateUpWeightShape, &downWeightShape})
                      .OutputShapes({&hiddenShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      // The stage attr encodes dtype offset plus stage id; 1000 + 102 means FP32 SWIGLU.
                      .NodeAttrs({{"cube_math_type", Ops::NN::AnyValue::CreateFrom<int64_t>(
                                                       WG_MLP_STAGE_ATTR_FP32_SWIGLU)}})
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();

    auto context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(tilingFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetTilingKey(), optiling::WG_MLP_KEY_FP32_SWIGLU);
}
