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
#include "mc2_tiling_case_executor.h"

using namespace std;

namespace MoeDistributeCombineUT {

static constexpr uint64_t kSkipTilingKeyCheck = UINT64_MAX;

class MoeDistributeCombineArch35TilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeDistributeCombineArch35TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeDistributeCombineArch35TilingTest TearDown" << std::endl;
    }
};

// Basic success: epWorldSize=8, tpWorldSize=1, shared expert (epRankId < sharedExpertRankNum)
TEST_F(MoeDistributeCombineArch35TilingTest, BasicSharedExpert)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{7 * 8}, {7 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, 32UL);
}

// Success: tpWorldSize=1, 校验 tp 属性参数
TEST_F(MoeDistributeCombineArch35TilingTest, TpWorldSize1)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    int64_t epWorldSize = 288;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    int64_t sharedExpertRankNum = 32;
    int64_t moeExpertNum = 256;
    int64_t globalBs = 0;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{576, 7160}, {576, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32 * 8}, {32 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{32, 7160}, {32, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epRankId)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertRankNum)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moeExpertNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// tpWorldSize>=2 校验失败
TEST_F(MoeDistributeCombineArch35TilingTest, Int8CommQuant)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    int64_t epWorldSize = 8;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 2;
    int64_t groupListType = 0;
    int64_t sharedExpertRankNum = 1;
    int64_t moeExpertNum = 7;
    int64_t globalBs = 0;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{7 * 8}, {7 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epRankId)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertRankNum)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moeExpertNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// Success: epWorldSize=4, covers A5 CheckEpWorldSize with 256%4==0
TEST_F(MoeDistributeCombineArch35TilingTest, EpWorldSize4)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    int64_t epWorldSize = 4;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    int64_t sharedExpertRankNum = 2;
    int64_t moeExpertNum = 2;
    int64_t globalBs = 0;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{8 * 2, 7168}, {8 * 2, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 2}, {8, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 2}, {8 * 2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 2}, {8, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epRankId)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertRankNum)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moeExpertNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// Success: epWorldSize=144, covers A5 CheckEpWorldSize with 144%144==0
TEST_F(MoeDistributeCombineArch35TilingTest, EpWorldSize144)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    int64_t epWorldSize = 144;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    int64_t sharedExpertRankNum = 16;
    int64_t moeExpertNum = 128;
    int64_t globalBs = 0;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{288, 7168}, {288, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 8}, {8 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{144}, {144}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epRankId)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertRankNum)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moeExpertNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// Success: epWorldSize=2, covers smallest valid epWorldSize (even, 256%2==0)
TEST_F(MoeDistributeCombineArch35TilingTest, EpWorldSize2)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    int64_t epWorldSize = 2;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    int64_t sharedExpertRankNum = 1;
    int64_t moeExpertNum = 1;
    int64_t globalBs = 0;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 1}, {8, 1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 1}, {8, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epRankId)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertRankNum)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moeExpertNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// Success: moe expert rank (epRankId >= sharedExpertRankNum), covers isShared=false path
TEST_F(MoeDistributeCombineArch35TilingTest, MoeExpertRank)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    int64_t epWorldSize = 288;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 32;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    int64_t sharedExpertRankNum = 32;
    int64_t moeExpertNum = 256;
    int64_t globalBs = 0;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{576, 7160}, {576, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32 * 8}, {32 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{32, 7160}, {32, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epRankId)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertRankNum)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moeExpertNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// Success: commQuantMode=2 (INT8_COMM_QUANT), tpWorldSize=1, covers quantMode=TILINGKEY_INT8_QUANT branch
TEST_F(MoeDistributeCombineArch35TilingTest, Int8CommQuantTp1)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    int64_t epWorldSize = 8;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 2;
    int64_t groupListType = 0;
    int64_t sharedExpertRankNum = 1;
    int64_t moeExpertNum = 7;
    int64_t globalBs = 0;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{7 * 8}, {7 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epRankId)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moeExpertNum)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpWorldSize)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertRankNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, kSkipTilingKeyCheck);
}

// Error: epWorldSize=7 (odd), covers CheckEpWorldSize error branch: epWorldSize%2!=0
TEST_F(MoeDistributeCombineArch35TilingTest, EpWorldSizeOdd)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    int64_t epWorldSize = 7;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    int64_t sharedExpertRankNum = 1;
    int64_t moeExpertNum = 6;
    int64_t globalBs = 0;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{7 * 8}, {7 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{7}, {7}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epRankId)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertRankNum)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moeExpertNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// Error: epWorldSize=6, covers CheckEpWorldSize error: 256%6!=0 && 6%144!=0
TEST_F(MoeDistributeCombineArch35TilingTest, EpWorldSizeInvalid)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    int64_t epWorldSize = 6;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    int64_t sharedExpertRankNum = 1;
    int64_t moeExpertNum = 5;
    int64_t globalBs = 0;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{7 * 8}, {7 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epRankId)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertRankNum)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moeExpertNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}

// Success: globalBs non-zero, covers CheckBatchAttrs globalBs!=0 branch
TEST_F(MoeDistributeCombineArch35TilingTest, GlobalBsNonZero)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    int64_t epWorldSize = 8;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    int64_t sharedExpertRankNum = 1;
    int64_t moeExpertNum = 7;
    int64_t globalBs = 64;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{7 * 8}, {7 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epRankId)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moeExpertNum)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpWorldSize)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertRankNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, kSkipTilingKeyCheck);
}

// Success: epWorldSize=288, sharedExpertRankNum=0 (all MOE experts), sharedExpertRankNum=0
TEST_F(MoeDistributeCombineArch35TilingTest, AllMoeExperts)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    int64_t epWorldSize = 8;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    int64_t sharedExpertRankNum = 0;
    int64_t moeExpertNum = 8;
    int64_t globalBs = 512;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{512, 7168}, {512, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 8}, {8 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epRankId)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moeExpertNum)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpWorldSize)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertRankNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, kSkipTilingKeyCheck);
}

// Success: MOE expert rank (epRankId=1 >= sharedExpertRankNum=1), expandX dim1=7168, covers isShared=false
TEST_F(MoeDistributeCombineArch35TilingTest, MoeExpertRankSuccess)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    int64_t epWorldSize = 8;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 1;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    int64_t sharedExpertRankNum = 1;
    int64_t moeExpertNum = 7;
    int64_t globalBs = 0;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{7 * 8}, {7 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epRankId)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moeExpertNum)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpWorldSize)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertRankNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, kSkipTilingKeyCheck);
}

// tpWorldSize=1 with valid shapes (expandX dim1=7168)
TEST_F(MoeDistributeCombineArch35TilingTest, TpWorldSize1Success)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    int64_t epWorldSize = 8;
    int64_t tpWorldSize = 1;
    int64_t epRankId = 0;
    int64_t tpRankId = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    int64_t sharedExpertRankNum = 1;
    int64_t moeExpertNum = 7;
    int64_t globalBs = 0;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{128, 7168}, {128, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{7 * 8}, {7 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epWorldSize)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(epRankId)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(moeExpertNum)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpWorldSize)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tpRankId)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(expertShardType)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertNum)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sharedExpertRankNum)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(globalBs)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(outDtype)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(commQuantMode)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(groupListType)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, kSkipTilingKeyCheck);
}

// Error: tpRankId != 0, covers CheckCommAttrValuesValid tpRankId branch
TEST_F(MoeDistributeCombineArch35TilingTest, InvalidTpRankId)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{7 * 8}, {7 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// Error: epWorldSize not divisible by sharedExpertRankNum
TEST_F(MoeDistributeCombineArch35TilingTest, EpWorldSizeNotDivisibleBySharedRank)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{7 * 8}, {7 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// Error: moeExpertNum not divisible by (epWorldSize - sharedExpertRankNum)
TEST_F(MoeDistributeCombineArch35TilingTest, MoeExpertNumIndivisible)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{7 * 8}, {7 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// Error: K > moeExpertNum
TEST_F(MoeDistributeCombineArch35TilingTest, KGreaterThanMoeExpertNum)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8 * 8}, {8 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// Error: globalBs not divisible by epWorldSize
TEST_F(MoeDistributeCombineArch35TilingTest, InvalidGlobalBs)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{7 * 8}, {7 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// Error: expandX dim0 too small
TEST_F(MoeDistributeCombineArch35TilingTest, InvalidExpandXDim0)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{1, 7168}, {1, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{7 * 8}, {7 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// Error: expandIdx dim0 != bs * k
TEST_F(MoeDistributeCombineArch35TilingTest, InvalidExpandIdxDim0)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{7 * 8 - 1}, {7 * 8 - 1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// Error: expertScales dim0 != bs
TEST_F(MoeDistributeCombineArch35TilingTest, InvalidExpertScalesDim0)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{7 * 8}, {7 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{7, 7}, {7, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

// Error: output x dim0 != bs
TEST_F(MoeDistributeCombineArch35TilingTest, InvalidOutputXDim0)
{
    struct MoeDistributeCombineCompileInfo {};
    MoeDistributeCombineCompileInfo compileInfo;
    std::string epGroup("epGroup");
    std::string tpGroup("tpGroup");
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "MoeDistributeCombine",
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{7 * 8}, {7 * 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{7, 7168}, {7, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>(epGroup)},
         {"epWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"epRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moeExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7)},
         {"group_tp", Ops::Transformer::AnyValue::CreateFrom<std::string>(tpGroup)},
         {"tpWorldSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"tpRankId", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"expertShardType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"sharedExpertNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"sharedExpertRankNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"globalBs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"outDtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"commQuantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"groupListType", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend950", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

} // namespace MoeDistributeCombineUT
