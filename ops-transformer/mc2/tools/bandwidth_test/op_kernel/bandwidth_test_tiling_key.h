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
 * \file bandwidth_test_tiling_key.h
 * \brief
 */

#ifndef BANDWIDTH_TEST_TILING_KEY_H
#define BANDWIDTH_TEST_TILING_KEY_H
#include "ascendc/host_api/tiling/template_argument.h"

namespace BandwidthTestTiling {
#define TILINGKEY_TPL_A5 2

ASCENDC_TPL_ARGS_DECL(BandwidthTest,
                      ASCENDC_TPL_UINT_DECL(ARCH_TAG, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_A5), );
ASCENDC_TPL_SEL(
    // A5
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(ARCH_TAG, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_A5),
                         ASCENDC_TPL_TILING_STRUCT_SEL(BandwidthTestTilingData)), );
} // namespace BandwidthTestTiling
#endif
