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
 * \file common_utils.h
 */
#ifndef _ASCENDC_COMMON_UTILS_H_
#define _ASCENDC_COMMON_UTILS_H_

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t SYNC_PING_AIV_AIC_FLAG = 1;
constexpr uint32_t SYNC_PONG_AIV_AIC_FLAG = 2;
constexpr uint32_t SYNC_PING_AIC_AIV_FLAG = 3;
constexpr uint32_t SYNC_PONG_AIC_AIV_FLAG = 4;
constexpr uint32_t SYNC_CLEAR_AIV_AIC_FLAG = 11;
constexpr uint32_t SYNC_CLEAR_AIC_AIV_FLAG = 12;
constexpr uint32_t SYNC_CLEAR_AIC_FLAG = 13;
constexpr uint32_t SYNC_CLEAR_AIV_FLAG = 14;
constexpr uint32_t GROUP_MAX = 384;
constexpr uint32_t TRANS_BASEN = 1536;   // 192*1024/2/32/sizeof(half)
constexpr uint32_t REROUTE_BASEN = 8192; // 192*1024/4/(sizeof(half)+sizeof(float))
constexpr uint32_t SWIGLU_BASEN = 8192;  // 192*1024/2/3/sizeof(float)
constexpr uint32_t REROUTE_BASEN_MIN = 256;
constexpr uint32_t MY_BUFFER_NUM = 8;
constexpr float SPLITK_THRES = 1.12;

#endif