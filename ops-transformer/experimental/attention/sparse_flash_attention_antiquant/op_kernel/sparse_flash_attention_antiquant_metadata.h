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
 * \file sparse_flash_attention_antiquant_metadata.h
 * \brief
 */

#ifndef SPARSE_FLASH_ATTENTION_ANTIQUANT_METADATA_H
#define SPARSE_FLASH_ATTENTION_ANTIQUANT_METADATA_H

#include <cstdint>

namespace optiling {
const uint32_t CORE_NUM = 24;
constexpr uint32_t SFA_META_SIZE = 1024;
using SFA_METADATA_T = int32_t;

namespace detail {
    // 分核功能模块输出：FD信息，包含需要归约的数据索引及其分核信息
    struct FlashDecodeResult {
        // 1、归约任务的索引信息
        uint32_t bN2IdxOfFdHead[CORE_NUM];                   // 每个归约任务的BN2索引，脚标为归约任务的序号，最大为核数-1
        uint32_t gS1IdxOfFdHead[CORE_NUM];                   // 每个归约任务的GS1索引，脚标为归约任务的序号
        uint32_t s2SplitNumOfFdHead[CORE_NUM];               // 每个归约任务的S2核间切分份数，脚标为归约任务的序号
        // 2、FD负载均衡阶段，归约任务的分核（vec）信息
        uint32_t gS1SplitNumOfFdHead[CORE_NUM];              // 每个归约任务m轴切分份数，脚标为归约任务的序号
        uint32_t gS1LastPartSizeOfFdHead[CORE_NUM];          // 每个归约任务m轴切分的最后一份的大小，脚标为归约任务的序号
        uint32_t gS1IdxEndOfFdHead[CORE_NUM * 2];              // FD负载均衡阶段，每个vector的一级索引，脚标为vector ID，值为归约任务的ID
        uint32_t gS1IdxEndOfFdHeadSplit[CORE_NUM * 2];         // FD负载均衡阶段，每个vector的二级索引，脚标为vector ID，值为归约任务的m轴切分ID
        // 3、每个core处理的第1个归约任务的数据应存放的workspace位置
        uint32_t s2SplitStartIdxOfCore[CORE_NUM];
    };

    struct SfaMetaData { // __attribute__((aligned(8))) 
        uint32_t bN2End[CORE_NUM];               // 每个核处理数据的BN2结束点
        uint32_t gS1End[CORE_NUM];               // 每个核处理数据的GS1结束点
        uint32_t s2End[CORE_NUM];                // 每个核处理数据的S2结束点
        uint32_t usedCoreNum = 0U;            // 使用的核数量
        uint32_t numOfFdHead = 0U;            // 归约任务数量
        uint32_t usedVecNumOfFd = 0U;         // 归约过程使用的vector数量
        uint32_t mBaseSize = 0U;
        uint32_t s2BaseSize = 0U;
        uint32_t gS1BaseSizeOfFd = 0U;
        struct FlashDecodeResult fdRes;                //  FD信息
    };
};
static_assert(SFA_META_SIZE * sizeof(SFA_METADATA_T) >= sizeof(detail::SfaMetaData));
};

#endif
