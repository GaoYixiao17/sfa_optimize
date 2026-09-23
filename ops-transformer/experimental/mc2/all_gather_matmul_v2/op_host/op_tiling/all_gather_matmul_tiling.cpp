/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file all_gather_matmul_tiling.cpp
 * \brief
 */
#include <queue>
#include <dlfcn.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <vector>

#include "op_host/tiling_templates_registry.h"
#include "mc2_hcom_topo_info.h"
#include "op_host/op_tiling/matmul_formulaic_tiling.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "mc2_log.h"
#include "op_host/op_tiling/new_mc2_tiling_utils.h"
#include "all_gather_matmul_tiling.h"

using namespace Mc2Log;
using namespace AscendC;
using namespace Mc2Tiling;

namespace optiling {
ge::graphStatus AllGatherMatmulTilingFunc(gert::TilingContext *context);
ge::graphStatus TilingParseForAllGatherMatmul(gert::TilingParseContext *context);

bool AllGatherMatmulTiling::IsCapable()
{
    OP_LOGI(opName_, "Start with AllGatherMatmulTiling tiling.");
    return true;
}

ge::graphStatus AllGatherMatmulTiling::SetRawTilingData()
{
    auto rawTilingData = context_->GetRawTilingData();
    allGatherMatmulTilingData_ = context_->GetTilingData<AllGatherMatmulTilingData>();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulTiling::DoOpTiling()
{
    MC2_CHECK_LOG_RET(opName_, SetRawTilingData());
    OP_TILING_CHECK(SetMc2Hcomm(MutableRCSTilingData()) != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILING(opName_, "Fail to set Mc2Hcomm."), return ge::GRAPH_FAILED);
    SetRcsTilingData(MutableRCSTilingData());
    DoSplitMTiling(MutableRCSTilingData());
    MC2_CHECK_LOG_RET(opName_, AdjustHCCLLimit(MutableRCSTilingData(), mc2tiling::Mc2QuantMode::DEFAULT));
    MC2_CHECK_LOG_RET(opName_, DoVersion2Tiling());
    DoAllGatherTiling(MutableRCSTilingData(), MutableMC2MatmulV3TileTilingData().tCubeTiling,
                      MutableMC2MatmulV3TailTilingData().tCubeTiling, allGatherMatmulTilingData_->dataType);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulTiling::PostTiling()
{
    OP_LOGD(opName_, "Final tiling data size=%zu and context capacity size=%zu.", sizeof(AllGatherMatmulTilingData),
            context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(sizeof(AllGatherMatmulTilingData));

    context_->SetBlockDim(args_.aicCoreNum);
    // 独占全核，设置以后会让所有核空闲以后才启动，有多核同步指令需要设置避免出现网络挂死
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulTiling::DoMatmulV3Tiling(Mc2MatmulHelper::Mc2MatmulTilingCfg &tilingCfg,
                                                        Mc2MMRegisterCfg &registerCfg,
                                                        Mc2MatMulV3TilingData &tilingData)
{
    tilingCfg.SetRankDim(args_.rankDim - 1);
    tilingCfg.SetMatMulV3TilingData(tilingData);
    if (args_.nValue != 0) {
        if (Mc2MMTilingRegistry::GetInstance().DoTilingImpl(context_, tilingCfg, registerCfg) != ge::GRAPH_SUCCESS) {
            OP_LOGE(opName_, "DoMatmulV3Tiling failed.");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulTiling::DoVersion2Tiling()
{
    // 获取芯片平台信息
    auto platformInfo = context_->GetPlatformInfo();
    // 获取 compileInfo
    if (mc2_matmul_v3_advanced::InitCompileInfo(platformInfo, &compileInfo_) != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName_, "Fail to Init CompileInfo!");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatForm = platform_ascendc::PlatformAscendC(platformInfo);

    // 根据芯片型号获取策略模板
    platform_ascendc::SocVersion socVersion = ascendcPlatForm.GetSocVersion();
    NpuArch npuArch = ascendcPlatForm.GetCurNpuArch();

    std::vector<int32_t> priorities;
    MC2_CHECK_LOG_RET(opName_, mc2tiling::NewGetMatmulV3PriorityPolicy(npuArch, priorities, opName_));

    Mc2MMRegisterCfg registerCfg{"Mc2MatMulV3", npuArch, priorities};

    mc2tiling::NewUpdateMatmulV3Args(mmV3Args_, args_, opName_);

    // 计算 local 块 tiling
    Mc2MatmulHelper::Mc2MatmulTilingCfg localTilingCfg(reinterpret_cast<const void *>(&compileInfo_),
                                                       reinterpret_cast<const void *>(&mmV3Args_));
    MC2_CHECK_LOG_RET(opName_, DoMatmulV3Tiling(localTilingCfg, registerCfg, MutableMC2MatmulV3LocalTilingData()));

    // 计算 tile 块 tiling
    mmV3Args_.mValue = tileMValue_ * (args_.rankDim - 1) * (MutableRCSTilingData().tileCnt);
    Mc2MatmulHelper::Mc2MatmulTilingCfg tileTilingCfg(reinterpret_cast<const void *>(&compileInfo_),
                                                      reinterpret_cast<const void *>(&mmV3Args_), tileMValue_);
    tileTilingCfg.SetCommCnt(MutableRCSTilingData().tileCnt);
    MC2_CHECK_LOG_RET(opName_, DoMatmulV3Tiling(tileTilingCfg, registerCfg, MutableMC2MatmulV3TileTilingData()));
    MutableMC2MatmulV3TileTilingData().tCubeTiling.M = (tileMValue_ * (args_.rankDim - 1));

    if (tailMValue_ > 0) {
        // 计算 tail 块 tiling
        mmV3Args_.mValue = tailMValue_ * (args_.rankDim - 1) * (MutableRCSTilingData().tailCnt);
        Mc2MatmulHelper::Mc2MatmulTilingCfg tailTilingCfg(reinterpret_cast<const void *>(&compileInfo_),
                                                          reinterpret_cast<const void *>(&mmV3Args_), tailMValue_);
        tailTilingCfg.SetCommCnt(MutableRCSTilingData().tailCnt);
        MC2_CHECK_LOG_RET(opName_, DoMatmulV3Tiling(tailTilingCfg, registerCfg, MutableMC2MatmulV3TailTilingData()));
        MutableMC2MatmulV3TailTilingData().tCubeTiling.M = (tailMValue_ * (args_.rankDim - 1));
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulTiling::SetMc2Hcomm(Mc2Tiling::RCSTiling &rcsCfg)
{
    int index = 0;
    auto group = context_->GetAttrs()->GetAttrPointer<char>(index++);
    std::string algConfig = "AllGather=level0:fullmesh";
    Mc2CcTilingConfig mc2CcTilingConfig(
        group, static_cast<uint32_t>(mc2tiling::AicpuComType::HCCL_CMD_ALLGATHER), algConfig, 0,
        static_cast<uint32_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, args_.geAType)),
        static_cast<uint32_t>(mc2tiling::ConvertGeTypeToHcclType(opName_, args_.geAType)));
    uint8_t skipBufferWindowCopy = (allGatherMatmulTilingData_->param.gatherLen == 0) ?
                                       static_cast<uint8_t>(mc2tiling::MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_DEFAULT) :
                                       static_cast<uint8_t>(mc2tiling::MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_OUTPUT);
    mc2CcTilingConfig.SetSkipBufferWindowCopy(skipBufferWindowCopy);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(allGatherMatmulTilingData_->mc2InitTiling) != 0,
                    OP_LOGE(opName_, "mc2CcTilingConfig mc2tiling GetTiling mc2InitTiling failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(allGatherMatmulTilingData_->mc2CcTiling) != 0,
                    OP_LOGE(opName_, "mc2CcTilingConfig mc2tiling GetTiling mc2CcTiling failed"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

AllGatherMatmulTiling::AllGatherMatmulTiling(gert::TilingContext *context)
    : AllGatherMatmulTilingBase(context), allGatherMatmulTilingData_(&allGatherMatmulTilingDataSelf_)
{
}
// 注册Tiling类
REGISTER_TILING_TEMPLATE_WITH_ARCH(AllGatherMatmulV2, AllGatherMatmulTiling, static_cast<int32_t>(NpuArch::DAV_3510),
                                   0);

ge::graphStatus AllGatherMatmulTilingFunc(gert::TilingContext *context)
{
    return Ops::Transformer::OpTiling::TilingRegistryArch::GetInstance().DoTilingImpl(context);
}

struct AllGatherMatmulCompileInfo {};

ge::graphStatus TilingParseForAllGatherMatmul(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AllGatherMatmulV2)
    .Tiling(AllGatherMatmulTilingFunc)
    .TilingParse<AllGatherMatmulCompileInfo>(TilingParseForAllGatherMatmul);
} // namespace optiling
