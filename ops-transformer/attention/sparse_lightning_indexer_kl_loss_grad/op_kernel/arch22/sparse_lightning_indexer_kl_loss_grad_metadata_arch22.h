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
 * \file sparse_lightning_indexer_kl_loss_grad_metadata_arch22.h
 * \brief Runtime metadata consumed by sparse_lightning_indexer_kl_loss_grad.
 */
#ifndef SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_ARCH22_H
#define SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_ARCH22_H

#include <cstdint>

namespace optiling {

using SLI_METADATA_T = int32_t;

constexpr uint32_t SLI_METADATA_MAX_CORE_NUM = 25;
constexpr uint32_t SLI_METADATA_HEADER_SIZE = 8;
constexpr uint32_t SLI_METADATA_SIZE = 64;

constexpr uint32_t SLI_META_CORE_NUM_INDEX = 0;
constexpr uint32_t SLI_META_TOTAL_SIZE_INDEX = 1;
constexpr uint32_t SLI_META_SPLIT_FACTOR_SIZE_INDEX = 2;
constexpr uint32_t SLI_META_BS1_INDEX_BASE = SLI_METADATA_HEADER_SIZE;

namespace detail {
struct SliGradKLLossMetaData {
    int32_t coreNum;
    int32_t totalSize;
    int32_t splitFactorSize;
    int32_t reserved[SLI_METADATA_HEADER_SIZE - 3];
    int32_t bS1Index[SLI_METADATA_MAX_CORE_NUM];
};
} // namespace detail

static_assert(SLI_METADATA_SIZE * sizeof(SLI_METADATA_T) >= sizeof(detail::SliGradKLLossMetaData));

#ifdef __CCE_AICORE__
__aicore__ inline uint32_t GetSliMetaBS1IndexAttr(uint32_t coreIdx)
{
    return SLI_META_BS1_INDEX_BASE + coreIdx;
}
#endif

} // namespace optiling

#endif // SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_ARCH22_H
