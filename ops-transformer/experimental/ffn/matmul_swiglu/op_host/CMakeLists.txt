# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# open 构建下将 def 注册进最终 aclnn 主库(与 ffn/其它 experimental 算子一致)
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        matmul_swiglu_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME MatmulSwiglu
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
)

# 自动扫描 def/tiling/infershape/op_api 并注册; aclnn 接口自动生成(无 op_api 目录时生成 stub)
add_modules_sources(HOSTNAME ${OPHOST_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR} OPTYPE matmul_swiglu ACLNNTYPE aclnn)
