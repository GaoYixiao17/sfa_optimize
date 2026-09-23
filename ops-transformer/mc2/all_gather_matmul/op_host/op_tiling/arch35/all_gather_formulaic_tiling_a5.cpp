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
 * \file all_gather_formulaic_tiling_a5.cpp
 * \brief
 */

#include "all_gather_formulaic_tiling_a5.h"

void AllGatherPlusMMA5::SetCommTimeFactor()
{
    commPerf_.ChangeCommTimeFactorByDivision(ALLGATHERMM_COMMTIME_FACTOR); // 2x time of factor
}

bool AllGatherPlusMMA5::GetAllowMoreCuts(bool smallMFlag)
{
    bool allowMoreCuts = (!tilingM_.cutRes.shortTileAtBack && smallMFlag);

    return allowMoreCuts;
}
