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
 * \file moe_distribute_combine_teardown_tiling_key.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_COMBINE_TEARDOWN_TILING_KEY_H
#define MOE_DISTRIBUTE_COMBINE_TEARDOWN_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

// 模板参数
ASCENDC_TPL_ARGS_DECL(MoeDistributeCombineTeardown, ASCENDC_TPL_BOOL_DECL(TILINGKEY_TP, 0, 1), );

// 模板参数组合
// 用于调用GET_TPL_TILING_KEY获取TilingKey时，接口内部校验TilingKey是否合法
ASCENDC_TPL_SEL(ASCENDC_TPL_BOOL_SEL(TILINGKEY_TP, 0), );

#endif // MOE_DISTRIBUTE_COMBINE_TEARDOWN_TILING_KEY_H
