message(STATUS "BUILD_TORCH_OPS ON in moe_layernorm_add_add")
# MOE_LAYERNORM_ADD_ADD operation sources
file(GLOB MOE_LAYERNORM_ADD_ADD_NPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

set(MOE_LAYERNORM_ADD_ADD_SOURCES ${MOE_LAYERNORM_ADD_ADD_NPU_SOURCES})
# Mark .cpp files with special properties
set_source_files_properties(
    ${MOE_LAYERNORM_ADD_ADD_NPU_SOURCES} PROPERTIES
    LANGUAGE CXX
    COMPILE_FLAGS "--cce-soc-version=Ascend910B1 --cce-soc-core-type=VecCore --cce-auto-sync -xcce"
)

# Create object library
add_library(moe_layernorm_add_add_objects OBJECT ${MOE_LAYERNORM_ADD_ADD_SOURCES})

target_compile_options(moe_layernorm_add_add_objects PRIVATE ${COMMON_COMPILE_OPTIONS})
target_include_directories(moe_layernorm_add_add_objects PRIVATE ${COMMON_INCLUDE_DIRS})
return()