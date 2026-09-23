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
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/ffn_worker_batching_tiling.h"

using namespace ge;
using namespace optiling;

class FfnWorkerBatchingTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FfnWorkerBatchingTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FfnWorkerBatchingTilingTest TearDown" << std::endl;
    }
};

TEST_F(FfnWorkerBatchingTilingTest, ffn_worker_batching_tiling_test01)
{
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};

    gert::StorageShape y_shape = {{1024, 4096}, {1024, 4096}};
    gert::StorageShape group_list_shape = {{8, 2}, {8, 2}};
    gert::StorageShape session_ids_shape = {{1024}, {1024}};
    gert::StorageShape micro_batch_ids_shape = {{1024}, {1024}};
    gert::StorageShape token_ids_shape = {{1024}, {1024}};
    gert::StorageShape expert_offsets_shape = {{1024}, {1024}};
    gert::StorageShape dynamic_scale_shape = {{1024}, {1024}};
    gert::StorageShape actual_token_num_shape = {{1}, {1}};

    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "FfnWorkerBatching",
        {// input
         {schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND}},
        {
            // output
            {y_shape, ge::DT_INT8, ge::FORMAT_ND},                 // y
            {group_list_shape, ge::DT_INT64, ge::FORMAT_ND},       // group_list
            {session_ids_shape, ge::DT_INT32, ge::FORMAT_ND},      // session_ids
            {micro_batch_ids_shape, ge::DT_INT32, ge::FORMAT_ND},  // micro_batch_ids
            {token_ids_shape, ge::DT_INT32, ge::FORMAT_ND},        // token_ids
            {expert_offsets_shape, ge::DT_INT32, ge::FORMAT_ND},   // expert_offsets
            {dynamic_scale_shape, ge::DT_FLOAT, ge::FORMAT_ND},    // dynamic_scale
            {actual_token_num_shape, ge::DT_INT64, ge::FORMAT_ND}, // actual_token_num
        },
        {{"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"max_out_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({16, 8, 9, 4096})},
         {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layer_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo);
    int64_t expectTilingKey = 101;
    std::string expectTilingData = "1152 4096 0 8 32 261888 8160 1152 ";
    std::vector<size_t> expectWorkspaces = {16819232};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// ---------------------------------------------------------------------------------------------------------------------
// 迭代一 A5(arch35/ascend950) 核心路径 UT
//
// 路由机制（本组用例的判定依据）：
//   socVersion="Ascend950" -> PlatformAscendC.GetCurNpuArch()==DAV_3510 -> IsRegbaseSocVersion()==true
//   -> Tiling4FfnWorkerBatching 走 TilingRegistry::DoTilingImpl -> 命中 1000 优先级模板
//      FfnWorkerBatchingRegbaseTiling(_tiling_arch35.cpp)。
//   socVersion="Ascend910B"/"Ascend910_93" -> DAV_2201 -> IsRegbaseSocVersion()==false
//   -> 走原 monolithic FfnWorkerBatchingTiling::RunFfnWorkerBatchingTiling（A2/A3，逐字不动）。
//
// arch35 TilingData 平铺 struct FfnWorkerBatchingArch35TilingData 与 A2 FfnWorkerBatchingTilingData 字段一一对应
// （8×int64：Y H tokenDtype expertNum coreNum ubSize sortLoopMaxElement sortNumWorkSpace），A5 功能对齐 A2，
// 故相同输入下 golden 值与 A2 一致 —— 本身即是「arch35 复算等价于 A2」的交叉校验。
// ---------------------------------------------------------------------------------------------------------------------

namespace {
// 构造 8 输出 desc（shape 仅供 faker 建 tensor，tiling 不读输出 shape，dtype 按 def 顺序）。
std::vector<gert::TilingContextPara::TensorDescription> MakeA5OutputDesc()
{
    static gert::StorageShape y_shape = {{1152, 4096}, {1152, 4096}};
    static gert::StorageShape group_list_shape = {{8, 2}, {8, 2}};
    static gert::StorageShape idx_shape = {{1152}, {1152}};
    static gert::StorageShape scale_shape = {{1152}, {1152}};
    static gert::StorageShape actual_token_num_shape = {{1}, {1}};
    return {
        {y_shape, ge::DT_INT8, ge::FORMAT_ND},                 // y
        {group_list_shape, ge::DT_INT64, ge::FORMAT_ND},       // group_list
        {idx_shape, ge::DT_INT32, ge::FORMAT_ND},              // session_ids
        {idx_shape, ge::DT_INT32, ge::FORMAT_ND},              // micro_batch_ids
        {idx_shape, ge::DT_INT32, ge::FORMAT_ND},              // token_ids
        {idx_shape, ge::DT_INT32, ge::FORMAT_ND},              // expert_offsets
        {scale_shape, ge::DT_FLOAT, ge::FORMAT_ND},            // dynamic_scale
        {actual_token_num_shape, ge::DT_INT64, ge::FORMAT_ND}, // actual_token_num
    };
}
} // namespace

// P0-核心路径①：Ascend950 + need_schedule=0(NORM) -> arch35 1000 档，TilingKey=100。
// NORM/100 分支是相对现有基线（need_schedule=1/101）全新的覆盖点；coreNum 不受 RECV 32 限核，保持 aivNum=64。
TEST_F(FfnWorkerBatchingTilingTest, ffn_worker_batching_tiling_arch35_ascend950_norm_key100)
{
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "FfnWorkerBatching",
        {// input
         {schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND}},
        MakeA5OutputDesc(),
        {{"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"max_out_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({16, 8, 9, 4096})},
         {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"layer_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo,
        "Ascend950"); // -> DAV_3510 -> arch35 1000 档

    // 机器证据：先低阶执行并打印 arch35 实际 tiling 结果（tilingKey/blockDim/workspace/8×int64）。
    TilingInfo info;
    bool ok = ExecuteTiling(tilingContextPara, info);
    ASSERT_TRUE(ok);
    std::cout << "[arch35][ascend950][NORM] tilingKey=" << info.tilingKey << " blockDim=" << info.blockNum
              << " workspace=" << (info.workspaceSizes.empty() ? -1 : info.workspaceSizes[0]) << " tilingData=";
    {
        const int64_t *d = reinterpret_cast<const int64_t *>(info.tilingData.get());
        for (size_t i = 0; i < info.tilingDataSize / sizeof(int64_t); ++i) {
            std::cout << d[i] << " ";
        }
        std::cout << std::endl;
    }

    int64_t expectTilingKey = 100; // NORM
    // 字段: Y H tokenDtype expertNum coreNum ubSize sortLoopMaxElement sortNumWorkSpace
    // coreNum=64（NORM 不限核）。GetCoreMemSize(UB)=262144：arch35(DAV_3510) 扣减 32KB 系统预留 -> ubSize=229376，
    // 与 A2(DAV_2201) 预留 256B 不同口径（A2 扣 256B 得 261888；arch35 扣 32768 得 229376）。
    // sortLoopMaxElement=(262144-32768)/(4*2*4)/32*32=7168。运行时取 UB 后按固定系统预留扣减，非写死 arch 常量。
    std::string expectTilingData = "1152 4096 0 8 64 229376 7168 1152 ";
    std::vector<size_t> expectWorkspaces = {16819232};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// P0-核心路径②：Ascend950 + need_schedule=1(RECV) -> arch35 1000 档，TilingKey=101，RECV 限核 coreNum=min(64,32)=32。
// 与基线 test01（Ascend910B/101）golden 完全一致 -> 证明 arch35 RECV 复算等价于 A2。
TEST_F(FfnWorkerBatchingTilingTest, ffn_worker_batching_tiling_arch35_ascend950_recv_key101)
{
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "FfnWorkerBatching",
        {// input
         {schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND}},
        MakeA5OutputDesc(),
        {{"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"max_out_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({16, 8, 9, 4096})},
         {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layer_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 101; // RECV
    // arch35 GetCoreMemSize(UB)=262144 扣 32KB 系统预留 -> ubSize=229376 / sortLoopMaxElement=7168（同 NORM
    // 用例说明）；coreNum=32 为 RECV 限核。
    std::string expectTilingData = "1152 4096 0 8 32 229376 7168 1152 ";
    std::vector<size_t> expectWorkspaces = {16819232};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 规格覆盖：Ascend950 + BF16(token_dtype=1) profile，NORM -> tokenDtype 字段=1，TilingKey=100。
TEST_F(FfnWorkerBatchingTilingTest, ffn_worker_batching_tiling_arch35_ascend950_bf16_norm)
{
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "FfnWorkerBatching", {{schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        {{"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"max_out_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({16, 8, 9, 4096})},
         {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}, // BF16
         {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"layer_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 100;
    std::string expectTilingData = "1152 4096 1 8 64 229376 7168 1152 ";
    std::vector<size_t> expectWorkspaces = {16819232};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 规格覆盖：Ascend950 + dynamic_quant_int8(token_dtype=2) profile，RECV -> tokenDtype 字段=2，TilingKey=101。
TEST_F(FfnWorkerBatchingTilingTest, ffn_worker_batching_tiling_arch35_ascend950_int8_recv)
{
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "FfnWorkerBatching", {{schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        {{"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"max_out_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({16, 8, 9, 4096})},
         {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}, // dynamic_quant_int8
         {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layer_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend950");

    int64_t expectTilingKey = 101;
    std::string expectTilingData = "1152 4096 2 8 32 229376 7168 1152 ";
    std::vector<size_t> expectWorkspaces = {16819232};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 零回归①：Ascend910B(A2) + need_schedule=0(NORM) -> monolithic 路径 TilingKey=100，coreNum=64。
// 验证 IsRegbaseSocVersion 对 A2 返回 false（未误走 arch35），且 A2 NORM 分支输出正确。
TEST_F(FfnWorkerBatchingTilingTest, ffn_worker_batching_tiling_ascend910b_norm_no_regression)
{
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "FfnWorkerBatching", {{schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        {{"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"max_out_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({16, 8, 9, 4096})},
         {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"layer_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo,
        "Ascend910B"); // -> DAV_2201 -> monolithic A2 路径

    int64_t expectTilingKey = 100;
    std::string expectTilingData = "1152 4096 0 8 64 261888 8160 1152 ";
    std::vector<size_t> expectWorkspaces = {16819232};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 零回归②：Ascend910_93(A3) + need_schedule=1(RECV) -> monolithic 路径 TilingKey=101，coreNum=32。
// 与基线 test01（Ascend910B）语义一致，验证 A3 走原路径无回归。
TEST_F(FfnWorkerBatchingTilingTest, ffn_worker_batching_tiling_ascend910_93_recv_no_regression)
{
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "FfnWorkerBatching", {{schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        {{"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"max_out_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({16, 8, 9, 4096})},
         {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"layer_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo,
        "Ascend910_93"); // -> DAV_2201 -> monolithic A3 路径

    int64_t expectTilingKey = 101;
    std::string expectTilingData = "1152 4096 0 8 32 261888 8160 1152 ";
    std::vector<size_t> expectWorkspaces = {16819232};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// =====================================================================================================================
// 迭代二 A2 Tiling 分支 UT 扩充（op_host Tiling 分支覆盖达标）
//
// 覆盖点（迭代二验收）：
//   1) NORM(need_schedule=0)->TilingKey 100 / RECV(need_schedule=1)->TilingKey 101 分派正确（arch35 1000 档）
//   2) 单核/多核 sort 阈值分支：小 shape(Y<=sortLoopMaxElement) 单核 / 大 shape(Y>sortLoopMaxElement) 多核，
//      tiling 侧驱动量（Y、sortLoopMaxElement）正确
//   3) RECV 限核阈值 TH_RECV_CORE_NUM=32 / TH_RECV_Y_NUM=200000 生效（Y<=200000 限核 32；Y>200000 不限核）
//   4) UB 阈值 sortLoopMaxElement 运行时派生正确，且对齐 issue_20260708 收敛口径
//      GetCoreMemSize(UB)=248KB(253952 B) —— 真实 Ascend950 platform_config ub_size，扣 32KB 系统预留后派生
//   5) A2/A3(ascend910b/ascend910_93) 仍走原 monolithic 路径、零回归（守卫验证）
//
// 关键派生关系（与 _tiling_arch35.cpp / _tiling.cpp 一致）：
//   sortLoopMaxElement = ubSize / (sizeof(int32)*2*4) / 32 * 32 = ubSize/32 向下 32 对齐
//     - arch35(DAV_3510) 扣减 32KB 系统预留：GetCoreMemSize 返回值 - 32768 后派生
//       → 248KB=253952 扣后 221184 → 6912；256KB=262144 扣后 229376 → 7168
//     - A2/A3(DAV_2201) 预留 256B：注入 262144 → GetCoreMemSize=261888 → 8160
//   workspace = 128*4 + Y*4 + Y*4*2*4 + expertNum*4 + sysWorkspaceSize(16777216) = 16777760 + 36*Y (expertNum=8)
//   RECV 限核：need_schedule==1 && Y<=200000 → coreNum=min(aivNum,32)；否则 coreNum=aivNum
//
// 注：faker 经 TilingContextPara(coreNum,ubSize) 注入平台值（UB_SIZE/CORE_NUM），本组用 253952 显式建模真实
//     Ascend950 248KB 收敛口径；上方迭代一用例保留 faker 默认 262144（同为「运行时派生、非写死」的等价证明）。
// =====================================================================================================================

namespace {
constexpr uint64_t kArch35Ub248K = 253952; // GetCoreMemSize(UB) 真实 Ascend950 返回值（issue_20260708 收敛口径）
constexpr uint64_t kAivNum64 = 64;

// 按 (tokenDtype, needSchedule, A, BS, K, H) 构造 attrs；Y=A*BS*K，max_out_shape={A,BS,K,H}。
std::vector<gert::TilingContextPara::OpAttr> MakeArch35Attrs(int64_t tokenDtype, int64_t needSchedule, int64_t A,
                                                             int64_t BS, int64_t K, int64_t H, int64_t expertNum = 8)
{
    return {{"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertNum)},
            {"max_out_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({A, BS, K, H})},
            {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tokenDtype)},
            {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(needSchedule)},
            {"layer_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}};
}
} // namespace

// A1｜NORM 分派 + 248KB UB 收敛 + 单核阈值：Ascend950 + need_schedule=0 -> TilingKey=100，coreNum=64（NORM 不限核）。
// GetCoreMemSize(UB)=253952(248KB)，扣 32KB 系统预留 -> ubSize=221184(216KB) -> sortLoopMaxElement=6912；
// Y=1152 <= 6912 -> kernel 侧走单核 sort（tiling 驱动量正确）。扣减防多核 sort 4-buffer footprint 溢出（穿刺2 实测）。
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_norm_key100_ub248k_singlecore)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        "FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        MakeArch35Attrs(/*tokenDtype*/ 0, /*needSchedule*/ 0, /*A*/ 16, /*BS*/ 8, /*K*/ 9, /*H*/ 4096), &compileInfo,
        "Ascend950", kAivNum64, kArch35Ub248K);

    // 机器证据：先打印 arch35 实际 tiling，再断言 golden。
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    const int64_t *d = reinterpret_cast<const int64_t *>(info.tilingData.get());
    std::cout << "[A1][NORM][ub248k] tilingKey=" << info.tilingKey << " blockDim=" << info.blockNum << " Y=" << d[0]
              << " coreNum=" << d[4] << " ubSize=" << d[5] << " sortLoopMaxElement=" << d[6] << std::endl;
    EXPECT_EQ(info.tilingKey, 100);
    EXPECT_EQ(info.blockNum, 64u); // NORM 不限核
    EXPECT_EQ(d[6], 6912);         // sortLoopMaxElement = (253952-32768)/32 对齐（扣系统预留）
    EXPECT_LE(d[0], d[6]);         // Y(1152) <= sortLoopMaxElement -> 单核 sort 分支

    std::string expectTilingData = "1152 4096 0 8 64 221184 6912 1152 ";
    std::vector<size_t> expectWorkspaces = {16819232};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 100, expectTilingData, expectWorkspaces);
}

// A2｜RECV 分派 + 248KB UB + 限核：Ascend950 + need_schedule=1 -> TilingKey=101，Y=1152<=200000 ->
// coreNum=min(64,32)=32。
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_recv_key101_ub248k_corelimit)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        "FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        MakeArch35Attrs(/*tokenDtype*/ 0, /*needSchedule*/ 1, /*A*/ 16, /*BS*/ 8, /*K*/ 9, /*H*/ 4096), &compileInfo,
        "Ascend950", kAivNum64, kArch35Ub248K);

    std::string expectTilingData = "1152 4096 0 8 32 221184 6912 1152 "; // coreNum=32 限核；ubSize 扣 32KB 系统预留
    std::vector<size_t> expectWorkspaces = {16819232};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 101, expectTilingData, expectWorkspaces);
}

// A3｜UB 阈值运行时派生证明：同一 shape/NORM/Ascend950，仅 ubSize 不同 -> sortLoopMaxElement 随之变化（非写死 arch
// 常量）。
//     扣 32KB 系统预留后：253952(248KB)->(253952-32768)/32=6912；262144(256KB)->(262144-32768)/32=7168。
//     对齐 MEMORY「UB size no hardcode」（运行时取值 + 固定系统预留扣减，非写死 UB 容量常量）。
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_sortloopmax_runtime_derived)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};

    gert::TilingContextPara para248("FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
                                    MakeArch35Attrs(0, 0, 16, 8, 9, 4096), &compileInfo, "Ascend950", kAivNum64,
                                    253952);
    gert::TilingContextPara para256("FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
                                    MakeArch35Attrs(0, 0, 16, 8, 9, 4096), &compileInfo, "Ascend950", kAivNum64,
                                    262144);

    TilingInfo info248;
    TilingInfo info256;
    ASSERT_TRUE(ExecuteTiling(para248, info248));
    ASSERT_TRUE(ExecuteTiling(para256, info256));
    int64_t slme248 = reinterpret_cast<const int64_t *>(info248.tilingData.get())[6];
    int64_t slme256 = reinterpret_cast<const int64_t *>(info256.tilingData.get())[6];
    std::cout << "[A3][runtime-derived] ub253952->slme=" << slme248 << " ub262144->slme=" << slme256 << std::endl;
    EXPECT_EQ(slme248, 6912);    // (253952-32768)/32
    EXPECT_EQ(slme256, 7168);    // (262144-32768)/32
    EXPECT_NE(slme248, slme256); // ubSize 变则阈值变 -> 运行时派生，非写死
}

// B1｜多核 sort 阈值分支（大 shape）：Ascend950 + NORM + Y=65536 > sortLoopMaxElement(6912) -> kernel 走多核 sort。
//     tiling 侧 coreNum=64（NORM 不限核）、sortNumWorkSpace=Y 正确；workspace 随 Y 增大（16777760+36*Y）。
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_norm_multicore_largeshape)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        "FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        MakeArch35Attrs(/*tokenDtype*/ 0, /*needSchedule*/ 0, /*A*/ 1024, /*BS*/ 1, /*K*/ 64, /*H*/ 4096), &compileInfo,
        "Ascend950", kAivNum64, kArch35Ub248K);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    const int64_t *d = reinterpret_cast<const int64_t *>(info.tilingData.get());
    std::cout << "[B1][NORM][multicore] Y=" << d[0] << " sortLoopMaxElement=" << d[6] << " coreNum=" << d[4]
              << std::endl;
    EXPECT_EQ(info.tilingKey, 100);
    EXPECT_GT(d[0], d[6]); // Y(65536) > sortLoopMaxElement(6912) -> 多核 sort 分支
    EXPECT_EQ(d[4], 64);   // NORM 不限核

    // Y=65536: workspace = 16777760 + 36*65536 = 19137056
    std::string expectTilingData = "65536 4096 0 8 64 221184 6912 65536 ";
    std::vector<size_t> expectWorkspaces = {19137056};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 100, expectTilingData, expectWorkspaces);
}

// C1｜RECV 限核边界（Y 低于 aivNum×TH_RECV_MIN_ROWS_PER_CORE=64×64=4096）：Ascend950 + RECV + Y=4032(<4096)
//     -> 每核分不满 64 行、gather 非瓶颈，保留 A2 的 32 核限核，coreNum=min(64,32)=32。
//     （A5 新策略取代旧 flat TH_RECV_Y_NUM=200000 限核：仅小 Y 限核，大 Y 放开满核打满 gather 带宽。）
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_recv_corelimit_small_y_32)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        "FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        MakeArch35Attrs(/*tokenDtype*/ 0, /*needSchedule*/ 1, /*A*/ 63, /*BS*/ 1, /*K*/ 64, /*H*/ 4096), // Y=4032
        &compileInfo, "Ascend950", kAivNum64, kArch35Ub248K);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    const int64_t *d = reinterpret_cast<const int64_t *>(info.tilingData.get());
    std::cout << "[C1][RECV][small Y=4032] coreNum=" << d[4] << " blockDim=" << info.blockNum << std::endl;
    EXPECT_EQ(info.tilingKey, 101);
    EXPECT_EQ(d[4], 32);           // Y<4096 -> 限核 32
    EXPECT_EQ(info.blockNum, 32u); // SetBlockDim 同步为 32

    // Y=4032: workspace = 16777760 + 36*4032 = 16922912
    std::string expectTilingData = "4032 4096 0 8 32 221184 6912 4032 ";
    std::vector<size_t> expectWorkspaces = {16922912};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 101, expectTilingData, expectWorkspaces);
}

// C2｜RECV 限核边界（Y 恰达阈值 4096）：Ascend950 + RECV + Y=4096(>=4096) -> 每核可分满 64 行，放开满核 coreNum=64。
//     与 C1 成对，精确卡住「Y < aivNum×64 才限核」的 < 边界（4032->32 / 4096->64）。
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_recv_corelimit_large_y_64)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        "FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        MakeArch35Attrs(/*tokenDtype*/ 0, /*needSchedule*/ 1, /*A*/ 64, /*BS*/ 1, /*K*/ 64, /*H*/ 4096), // Y=4096
        &compileInfo, "Ascend950", kAivNum64, kArch35Ub248K);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    const int64_t *d = reinterpret_cast<const int64_t *>(info.tilingData.get());
    std::cout << "[C2][RECV][large Y=4096] coreNum=" << d[4] << " blockDim=" << info.blockNum << std::endl;
    EXPECT_EQ(info.tilingKey, 101);
    EXPECT_EQ(d[4], 64); // Y>=4096 -> 放开满核 64
    EXPECT_EQ(info.blockNum, 64u);

    // Y=4096: workspace = 16777760 + 36*4096 = 16925216
    std::string expectTilingData = "4096 4096 0 8 64 221184 6912 4096 ";
    std::vector<size_t> expectWorkspaces = {16925216};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 101, expectTilingData, expectWorkspaces);
}

// C3｜大 expert_num（>1024）：Ascend950 + RECV + expert_num=5120 -> tokenDtype/coreNum/workspace 正确派生。
//     直接对应 scan_group_listing UB buffer 分批 flush 场景（expert_num 超 GROUP_LIST_UB_CAPACITY=1024 时
//     kernel 须分批搬出）：tiling 侧验证 workspace 的 expertNum*4 项随大 expert_num 正确增长（非写死 8）。
//     Y=1152(<sortLoopMaxElement 6912) 单核 sort，RECV Y<=200000 限核 coreNum=32。
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_recv_large_expert_num_5120)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara para("FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
                                 MakeArch35Attrs(/*tokenDtype*/ 0, /*needSchedule*/ 1, /*A*/ 16, /*BS*/ 8, /*K*/ 9,
                                                 /*H*/ 4096, /*expertNum*/ 5120),
                                 &compileInfo, "Ascend950", kAivNum64, kArch35Ub248K);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    const int64_t *d = reinterpret_cast<const int64_t *>(info.tilingData.get());
    std::cout << "[C3][RECV][expert_num=5120] Y=" << d[0] << " expertNum=" << d[3] << " coreNum=" << d[4] << std::endl;
    EXPECT_EQ(info.tilingKey, 101);
    EXPECT_EQ(d[3], 5120); // expertNum 字段透传正确（>1024，触发 kernel 分批 flush 场景）
    EXPECT_EQ(d[4], 32);   // RECV Y=1152<=200000 -> 限核 32

    // Y=1152, expertNum=5120: workspace = 16777728 + 36*1152 + 5120*4 = 16839680（expertNum*4 项随大专家数增长）
    std::string expectTilingData = "1152 4096 0 5120 32 221184 6912 1152 ";
    std::vector<size_t> expectWorkspaces = {16839680};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 101, expectTilingData, expectWorkspaces);
}

// D1｜A3 零回归（NORM）：Ascend910_93 + need_schedule=0 -> monolithic 路径 TilingKey=100，coreNum=64。
//     补齐现有 A3 RECV 无回归用例缺失的 A3 NORM 分支；DAV_2201 预留 256B -> ubSize=261888、sortLoopMaxElement=8160
//     （与 arch35 221184/6912 不同，证明 A3 未误走 arch35 1000 档）。
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_ascend910_93_norm_no_regression)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        "FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        MakeArch35Attrs(/*tokenDtype*/ 0, /*needSchedule*/ 0, /*A*/ 16, /*BS*/ 8, /*K*/ 9, /*H*/ 4096), &compileInfo,
        "Ascend910_93"); // 默认 coreNum=64/ubSize=262144 -> DAV_2201 monolithic

    std::string expectTilingData = "1152 4096 0 8 64 261888 8160 1152 ";
    std::vector<size_t> expectWorkspaces = {16819232};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 100, expectTilingData, expectWorkspaces);
}

// D2｜A2 零回归（大 shape 多核 + RECV 不限核边界）：Ascend910B + RECV + Y=204800(>200000) -> monolithic 路径
//     TilingKey=101、coreNum=64（不限核）。验证 A2 monolithic 与 arch35 共享同一 RECV 限核 <= 边界语义、
//     且大 shape 下仍走 DAV_2201 路径（ubSize=261888、sortLoopMaxElement=8160），零回归。
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_ascend910b_recv_largeshape_no_regression)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        "FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        MakeArch35Attrs(/*tokenDtype*/ 0, /*needSchedule*/ 1, /*A*/ 800, /*BS*/ 4, /*K*/ 64, /*H*/ 4096), // Y=204800
        &compileInfo, "Ascend910B");

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    const int64_t *d = reinterpret_cast<const int64_t *>(info.tilingData.get());
    std::cout << "[D2][A2][RECV largeshape] coreNum=" << d[4] << " ubSize=" << d[5] << " sortLoopMaxElement=" << d[6]
              << std::endl;
    EXPECT_EQ(info.tilingKey, 101);
    EXPECT_EQ(d[4], 64);     // Y>200000 不限核（A2 monolithic 同语义）
    EXPECT_EQ(d[5], 261888); // DAV_2201 预留 256B -> 未走 arch35（arch35 扣 32KB 会是 221184）

    // Y=204800: workspace = 16777760 + 36*204800 = 24150560
    std::string expectTilingData = "204800 4096 0 8 64 261888 8160 204800 ";
    std::vector<size_t> expectWorkspaces = {24150560};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 101, expectTilingData, expectWorkspaces);
}

// =====================================================================================================================
// arch35 硬校验错误路径 UT（对齐 spec.yaml boundary_conditions 的 raises_error 契约）
//
// 覆盖 arch35 FfnWorkerBatchingRegbaseTiling::CheckInputParam / GetAttrsInfo 的硬校验分支——这是与 A2 monolithic
// 校验相互独立的全新代码路径。全部经 socVersion="Ascend950" 路由到 1000 档 arch35 tiling，期望返回 ge::GRAPH_FAILED。
// 逐条对应 spec.yaml boundary_conditions：expert_num>8192 / topK+1>64 / A>1024 / max_out_shape 长度≠4 /
// token_dtype∉[0,2] / need_schedule∉[0,1]。校验阈值(EXPERT_IDX_MAX/MAX_SESSION_NUM/MAX_K_NUM)为对外规格，与代次无关。
// =====================================================================================================================

// E1｜expert_num 越界(>8192) -> arch35 GetAttrsInfo 报错（spec: index_out_of_bounds / attribute_value_out_of_range）。
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_expert_num_out_of_range_fail)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara para("FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
                                 MakeArch35Attrs(/*tokenDtype*/ 0, /*needSchedule*/ 0, /*A*/ 16, /*BS*/ 8, /*K*/ 9,
                                                 /*H*/ 4096, /*expertNum*/ 8193),
                                 &compileInfo, "Ascend950", kAivNum64, kArch35Ub248K);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// E2｜topK+1 = max_out_shape[2] 越界(>64) -> arch35 报错（spec MAX_K_NUM=64）。
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_topk_out_of_range_fail)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        "FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        MakeArch35Attrs(/*tokenDtype*/ 0, /*needSchedule*/ 0, /*A*/ 16, /*BS*/ 8, /*K*/ 65, /*H*/ 4096), &compileInfo,
        "Ascend950", kAivNum64, kArch35Ub248K);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// E3｜A = max_out_shape[0] 越界(>1024) -> arch35 报错（spec MAX_SESSION_NUM=1024）。
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_session_out_of_range_fail)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        "FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        MakeArch35Attrs(/*tokenDtype*/ 0, /*needSchedule*/ 0, /*A*/ 1025, /*BS*/ 1, /*K*/ 9, /*H*/ 4096), &compileInfo,
        "Ascend950", kAivNum64, kArch35Ub248K);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// E4｜max_out_shape 长度 != 4 -> arch35 报错（spec: 长度必须为 4）。
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_max_out_shape_len_fail)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        "FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        {{"expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"max_out_shape", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({16, 8, 9})}, // len=3
         {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"layer_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
        &compileInfo, "Ascend950", kAivNum64, kArch35Ub248K);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// E5｜token_dtype 越界(>2) -> arch35 报错（spec 取值范围 [0,2]）。
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_token_dtype_out_of_range_fail)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        "FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        MakeArch35Attrs(/*tokenDtype*/ 3, /*needSchedule*/ 0, /*A*/ 16, /*BS*/ 8, /*K*/ 9, /*H*/ 4096), &compileInfo,
        "Ascend950", kAivNum64, kArch35Ub248K);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// E6｜need_schedule 越界(>1) -> arch35 报错（spec 取值范围 [0,1]）。
TEST_F(FfnWorkerBatchingTilingTest, ffn_wb_tiling_arch35_need_schedule_out_of_range_fail)
{
    gert::StorageShape scShape = {{1024}, {1024}};
    FfnWorkerBatchingCompileInfo compileInfo = {};
    gert::TilingContextPara para(
        "FfnWorkerBatching", {{scShape, ge::DT_INT8, ge::FORMAT_ND}}, MakeA5OutputDesc(),
        MakeArch35Attrs(/*tokenDtype*/ 0, /*needSchedule*/ 2, /*A*/ 16, /*BS*/ 8, /*K*/ 9, /*H*/ 4096), &compileInfo,
        "Ascend950", kAivNum64, kArch35Ub248K);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
