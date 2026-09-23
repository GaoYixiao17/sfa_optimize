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
#ifndef __ALL_GATHER_MATMUL_TILING_H_
#define __ALL_GATHER_MATMUL_TILING_H_

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_dependency/mc2_tiling_struct.h"

namespace Mc2Tiling {

struct AllGatherMatmulTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    RCSTiling param;
    uint32_t dataType;
    Mc2MatMulV3TilingData mc2MmV3LocalTilingData;
    Mc2MatMulV3TilingData mc2MmV3TileTilingData;
    Mc2MatMulV3TilingData mc2MmV3TailTilingData;
};
} // namespace Mc2Tiling
#endif
