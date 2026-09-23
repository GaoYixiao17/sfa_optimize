message(STATUS "BUILD_TORCH_OPS ON in fusedexpert")
# FUSED_EXPERT operation sources
file(GLOB FUSED_EXPERT_NPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

set(FUSED_EXPERT_SOURCES ${FUSED_EXPERT_NPU_SOURCES})
# Mark .cpp files with special properties
set_source_files_properties(
    ${FUSED_EXPERT_NPU_SOURCES} PROPERTIES
    LANGUAGE CXX
    COMPILE_FLAGS "--cce-soc-version=Ascend910B1 --cce-soc-core-type=VecCore --cce-auto-sync -xcce"
)

# Create object library
add_library(fused_expert_objects OBJECT ${FUSED_EXPERT_SOURCES})

target_compile_options(fused_expert_objects PRIVATE ${COMMON_COMPILE_OPTIONS})
target_include_directories(fused_expert_objects PRIVATE ${COMMON_INCLUDE_DIRS})
return()
