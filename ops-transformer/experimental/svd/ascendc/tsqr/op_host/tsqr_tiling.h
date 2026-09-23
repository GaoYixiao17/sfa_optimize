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
 * \file tsqr_tiling.h
 * \brief
 */
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "tiling/matrix/matmul_tiling_base.h"
#include <iostream>

namespace optiling {

BEGIN_TILING_DATA_DEF(TsqrTilingData)
    TILING_DATA_FIELD_DEF(int32_t, batchSize);
    TILING_DATA_FIELD_DEF(int32_t, m);
    TILING_DATA_FIELD_DEF(int32_t, n);
    TILING_DATA_FIELD_DEF(int32_t, blockSize);
    TILING_DATA_FIELD_DEF(int32_t, numLevels);
    TILING_DATA_FIELD_DEF(int32_t, numBlocks);
    TILING_DATA_FIELD_DEF(int64_t, tmpQSize);
    TILING_DATA_FIELD_DEF(int64_t, tmpRSize);
    TILING_DATA_FIELD_DEF(int64_t, bufferQSize);
    TILING_DATA_FIELD_DEF(int64_t, ubSize);
    TILING_DATA_FIELD_DEF(int64_t, maxQrWorkspace);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingData);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingDataF);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Tsqr, TsqrTilingData)
} // namespace optiling
