message(STATUS "BUILD_TORCH_OPS ON in layernorm_add_add_quant")
# LAYERNORM_ADD_ADD_QUANT operation sources
file(GLOB LAYERNORM_ADD_ADD_QUANT_NPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

set(LAYERNORM_ADD_ADD_QUANT_SOURCES ${LAYERNORM_ADD_ADD_QUANT_NPU_SOURCES})
# Mark .cpp files with special properties
set_source_files_properties(
    ${LAYERNORM_ADD_ADD_QUANT_NPU_SOURCES} PROPERTIES
    LANGUAGE CXX
    COMPILE_FLAGS "--cce-soc-version=Ascend910B1 --cce-soc-core-type=VecCore --cce-auto-sync -xcce"
)

# Create object library
add_library(layernorm_add_add_quant_objects OBJECT ${LAYERNORM_ADD_ADD_QUANT_SOURCES})

target_compile_options(layernorm_add_add_quant_objects PRIVATE ${COMMON_COMPILE_OPTIONS})
target_include_directories(layernorm_add_add_quant_objects PRIVATE ${COMMON_INCLUDE_DIRS})
return()