# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if ((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    if (NOT TARGET swiglu_gated_mlp_ut_common_copy)
        add_custom_target(swiglu_gated_mlp_ut_common_copy
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/tbe/ascendc/common/act
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${OPS_NN_DIR}/common/act ${CMAKE_BINARY_DIR}/tbe/ascendc/common/act
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/tbe/ascendc/common/matmul_act
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_kernel/common/matmul_act ${CMAKE_BINARY_DIR}/tbe/ascendc/common/matmul_act
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/tbe/ascendc/inc
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${OPS_NN_DIR}/common/inc/op_kernel ${CMAKE_BINARY_DIR}/tbe/ascendc/inc
            VERBATIM
        )
    endif()

    AddOpTestCaseV2(swiglu_gated_mlp swiglu_gated_mlp "ascend910B1" "")

    if (TARGET swiglu_gated_mlp_Ascend910B1_cases_obj)
        target_include_directories(swiglu_gated_mlp_Ascend910B1_cases_obj PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_kernel
            ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include
        )
        add_dependencies(swiglu_gated_mlp_Ascend910B1_cases_obj swiglu_gated_mlp_ut_common_copy)
    endif()
endif()
