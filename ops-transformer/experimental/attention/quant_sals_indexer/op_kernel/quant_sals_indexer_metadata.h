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
 * \file quant_sals_indexer_metadata.h
 * \brief
 */

#ifndef QUANT_SALS_INDEXER_METADATA_H
#define QUANT_SALS_INDEXER_METADATA_H

#include <cstdint>

namespace optiling {
constexpr uint32_t CORE_NUM = 24;       // 样例代码，当前仅支持核数24
constexpr uint32_t QSI_META_SIZE = 1024;
using QSI_METADATA_T = int32_t;

namespace detail {
    struct QsiMetaData { // __attribute__((aligned(8))) 
        uint32_t bN2End[CORE_NUM];               // 每个核处理数据的BN2结束点
        uint32_t gS1End[CORE_NUM];               // 每个核处理数据的GS1结束点
        uint32_t s2End[CORE_NUM];                // 每个核处理数据的S2结束点
        uint32_t usedCoreNum = 0U;            // 使用的核数量
    };
};

static_assert(QSI_META_SIZE * sizeof(QSI_METADATA_T) >= sizeof(detail::QsiMetaData));
};

#endif
