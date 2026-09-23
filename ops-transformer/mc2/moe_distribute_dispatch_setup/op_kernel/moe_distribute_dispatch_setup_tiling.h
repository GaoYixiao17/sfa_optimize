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
 * \file moe_distribute_dispatch_setup_tiling.h
 * \brief tiling_data定义
 */
#ifndef ASCENDC_MOE_DISTRIBUTE_DISPATCH_SETUP_TILING_H
#define ASCENDC_MOE_DISTRIBUTE_DISPATCH_SETUP_TILING_H

struct MoeDistributeDispatchSetupInfo {
    uint32_t epWorldSize;
    uint32_t epRankId;
    uint32_t expertShardType;
    uint32_t sharedExpertNum;
    uint32_t sharedExpertRankNum;
    uint32_t moeExpertNum;
    uint32_t moeExpertPerRankNum;
    uint32_t quantMode;
    uint32_t globalBs;
    uint32_t bs;
    uint32_t k;
    uint32_t h;
    uint32_t aivNum;
    uint32_t sdmaUsedStreamPerCore;
    bool isQuant;
    bool isActiveMask;
    bool reserved2;
    bool reserved3;
    uint64_t totalUbSize;
    uint64_t totalWinSize;
    uint64_t scalesCount;
};

struct MoeDistributeDispatchSetupTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    MoeDistributeDispatchSetupInfo moeDistributeDispatchSetupInfo;
};

#endif
