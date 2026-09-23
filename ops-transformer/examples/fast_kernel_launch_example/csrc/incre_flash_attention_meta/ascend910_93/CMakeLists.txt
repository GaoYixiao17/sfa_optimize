# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

message(STATUS "BUILD_TORCH_OPS ON in incre_flash_attention_metadata")

# AICPU
set_source_files_properties(
    op_kernel/incre_flash_attention_metadata.cpp
    PROPERTIES
    LANGUAGE AICPU
)

add_library(incre_flash_attention_meta_kernel_objects STATIC
    op_kernel/incre_flash_attention_metadata.cpp
)

target_include_directories(incre_flash_attention_meta_kernel_objects PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ASCEND_HOME}/${SYSTEM_PREFIX}/include
    ${ASCEND_HOME}/${SYSTEM_PREFIX}/pkg_inc
    ${ASCEND_HOME}/${SYSTEM_PREFIX}/pkg_inc/base
    ${ASCEND_HOME}/${SYSTEM_PREFIX}/asc/include/aicpu_api
    ${ASCEND_HOME}/${SYSTEM_PREFIX}/pkg_inc/aicpu_common/context/utils
)

target_link_libraries(incre_flash_attention_meta_kernel_objects PRIVATE pthread)

# Torch
# Mark .cpp files with special properties
set_source_files_properties(
    npu_fused_infer_attention_score_metadata.cpp PROPERTIES
    LANGUAGE ASC
)

# Create object library
add_library(incre_flash_attention_meta_objects OBJECT npu_fused_infer_attention_score_metadata.cpp)

target_compile_options(incre_flash_attention_meta_objects PRIVATE ${COMPILE_OPTIONS} $<$<COMPILE_LANGUAGE:ASC>: --npu-arch=dav-2201>)
target_include_directories(incre_flash_attention_meta_objects PRIVATE ${INCLUDE_DIRECTORIES})
target_link_libraries(incre_flash_attention_meta_objects PRIVATE incre_flash_attention_meta_kernel_objects)

# add target obj to OBJECTS_LIST
set(TEMP_LIST ${OBJECTS_LIST})
list(APPEND TEMP_LIST $<TARGET_OBJECTS:incre_flash_attention_meta_objects>)
list(APPEND TEMP_LIST $<TARGET_OBJECTS:incre_flash_attention_meta_kernel_objects>)
set(OBJECTS_LIST ${TEMP_LIST} CACHE INTERNAL "List of objects" FORCE)
return()
