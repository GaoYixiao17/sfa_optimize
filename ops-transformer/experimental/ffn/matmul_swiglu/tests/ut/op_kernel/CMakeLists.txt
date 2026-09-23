# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# AddOpTestCase(算子kernel名, soc版本, 自定义编译选项, tiling源文件)
# 第4参传 host 端 tiling.cpp, 用于构建真 tiling.so 供 kernel UT 使用。
if ((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    AddOpTestCase(matmul_swiglu "ascend910B1" ""
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/matmul_swiglu_tiling.cpp")
endif()
