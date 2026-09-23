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
 * \file bandwidth_test_tiling.h
 * \brief
 */
#ifndef BANDWIDTH_TEST_TILING_H
#define BANDWIDTH_TEST_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct BandwidthTestTilingData {
    // ============ HCCL通信相关Tiling ============
    int32_t dataSize;      // hidden size，即特征维度大小
    int32_t worldSize;     // EP大小，即参与通信的rank总数
    int32_t maxBs;         // batch size，即token数量
    int32_t mode;          // 执行模式: 0=完整流程(发送+同步+接收), 1=仅发送
    int32_t aivNum;        // AIV核总数，用于任务分发
    uint64_t totalUbSize;  // 总UB大小，用于内存规划
    uint64_t totalWinSize; // HCCL通信窗口总大小
};

#endif // BANDWIDTH_TEST_TILING_H
