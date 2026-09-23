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
 * \file all_gather_matmul_tiling.h
 * \brief
 */

#ifndef __ALL_GATHER_MATMUL_TILING__
#define __ALL_GATHER_MATMUL_TILING__

#pragma once
#include "all_gather_matmul_tiling_base.h"
#include "register/tilingdata_base.h"
#include "mc2_matmul_tiling_cfg.h"
#include "../../op_kernel/all_gather_matmul_tiling.h"

namespace optiling {

using namespace mc2_matmul_v3_advanced;

class AllGatherMatmulTiling : public AllGatherMatmulTilingBase {
public:
    explicit AllGatherMatmulTiling(gert::TilingContext *context);
    ~AllGatherMatmulTiling() override = default;
    ge::graphStatus DoMatmulV3Tiling(Mc2MatmulHelper::Mc2MatmulTilingCfg &tilingCfg, Mc2MMRegisterCfg &registerCfg,
                                     Mc2MatMulV3TilingData &tilingData);

protected:
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;

    ge::graphStatus PostTiling() override;

    Mc2Tiling::RCSTiling &MutableRCSTilingData()
    {
        return allGatherMatmulTilingData_->param;
    }

    Mc2MatMulV3TilingData &MutableMC2MatmulV3LocalTilingData()
    {
        return allGatherMatmulTilingData_->mc2MmV3LocalTilingData;
    }
    Mc2MatMulV3TilingData &MutableMC2MatmulV3TileTilingData()
    {
        return allGatherMatmulTilingData_->mc2MmV3TileTilingData;
    }
    Mc2MatMulV3TilingData &MutableMC2MatmulV3TailTilingData()
    {
        return allGatherMatmulTilingData_->mc2MmV3TailTilingData;
    }

    ge::graphStatus DoVersion2Tiling();
    ge::graphStatus SetMc2Hcomm(Mc2Tiling::RCSTiling &rcsCfg);
    ge::graphStatus SetRawTilingData();

private:
    Mc2Tiling::AllGatherMatmulTilingData allGatherMatmulTilingDataSelf_;
    Mc2Tiling::AllGatherMatmulTilingData *allGatherMatmulTilingData_;
    uint64_t myWorkSpaceSize_{0U};
    Mc2MatMulV3Args mmV3Args_;
    Mc2MatmulV3CompileInfo compileInfo_;
};
} // namespace optiling
#endif
