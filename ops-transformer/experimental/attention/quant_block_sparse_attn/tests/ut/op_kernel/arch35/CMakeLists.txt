# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# NOTE: quant_block_sparse_attn uses AIC+AIV coordination (KERNEL_TYPE_MIX_AIC_1_2) and RegBase
# patterns, which are NOT supported by tikicpulib CPU simulation. This CMakeLists.txt creates a
# placeholder test object that compiles and links into the UT runner without compiling the kernel
# source. Full kernel tests require NPU hardware execution.

if (UT_TEST_ALL OR OP_KERNEL_UT)
    set(socVersion "Ascend950PR_9599")
    set(opName "quant_block_sparse_attn")

    file(GLOB OPKERNEL_CASES_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test_${opName}*.cpp)

    add_library(${opName}_${socVersion}_cases_obj OBJECT ${OPKERNEL_CASES_SRC})
    target_include_directories(
        ${opName}_${socVersion}_cases_obj PRIVATE
        ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include
        ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc
        ${PROJECT_SOURCE_DIR}/tests/ut/framework_normal/op_kernel
        ${PROJECT_SOURCE_DIR}/tests/ut/framework_normal/common
    )
    target_link_libraries(
        ${opName}_${socVersion}_cases_obj PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17> gtest
    )

    list(FIND fastOpTestSocVersions "${socVersion}" index)
    if(index EQUAL -1)
        set(fastOpTestSocVersions
            ${fastOpTestSocVersions} ${socVersion}
            CACHE STRING "fastOp Test SocVersions" FORCE
        )
    endif()

    if(NOT TARGET ${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj)
        add_library(
            ${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj OBJECT
            $<TARGET_OBJECTS:${opName}_${socVersion}_cases_obj>
        )
    else()
        target_sources(
            ${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj PRIVATE
            $<TARGET_OBJECTS:${opName}_${socVersion}_cases_obj>
        )
    endif()
    add_dependencies(
        ${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj ${opName}_${socVersion}_cases_obj
    )
    target_link_libraries(
        ${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
        $<TARGET_OBJECTS:${opName}_${socVersion}_cases_obj>
    )
endif()
