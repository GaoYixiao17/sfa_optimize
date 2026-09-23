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
 * \file kv_quant_sparse_flash_attention_v2_tiling.h
 * \brief 复用 kv_quant_sparse_flash_attention 的 tiling 定义, 仅注册 v2 算子名到共享 TilingData
 */
#ifndef KV_QUANT_SPARSE_FLASH_ATTENTION_V2_TILING_H
#define KV_QUANT_SPARSE_FLASH_ATTENTION_V2_TILING_H

#include "register/tilingdata_base.h"
#include "../../kv_quant_sparse_flash_attention/op_host/kv_quant_sparse_flash_attention_tiling.h"

namespace optiling {
REGISTER_TILING_DATA_CLASS(KvQuantSparseFlashAttentionV2, KvQuantSparseFlashAttentionTilingDataMla)
} // namespace optiling

#endif // KV_QUANT_SPARSE_FLASH_ATTENTION_V2_TILING_H
