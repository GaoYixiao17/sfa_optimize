message(STATUS "BUILD_TORCH_OPS ON in moegategeluclamp")
# MOE_GATE_GELU_CLAMP operation sources
file(GLOB MOE_GATE_GELU_CLAMP_NPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

set(MOE_GATE_GELU_CLAMP_SOURCES ${MOE_GATE_GELU_CLAMP_NPU_SOURCES})
# Mark .cpp files with special properties
set_source_files_properties(
    ${MOE_GATE_GELU_CLAMP_NPU_SOURCES} PROPERTIES
    LANGUAGE CXX
    COMPILE_FLAGS "--cce-soc-version=Ascend910B1 --cce-soc-core-type=VecCore --cce-auto-sync -xcce"
)

# Create object library
add_library(moe_gate_gelu_clamp_objects OBJECT ${MOE_GATE_GELU_CLAMP_SOURCES})

target_compile_options(moe_gate_gelu_clamp_objects PRIVATE ${COMMON_COMPILE_OPTIONS})
target_include_directories(moe_gate_gelu_clamp_objects PRIVATE ${COMMON_INCLUDE_DIRS})
return()
