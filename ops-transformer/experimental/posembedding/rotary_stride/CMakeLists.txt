message(STATUS "BUILD_TORCH_OPS ON in rotary_stride")
# ROTARY_STRIDE operation sources
file(GLOB ROTARY_STRIDE_NPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

set(ROTARY_STRIDE_SOURCES ${ROTARY_STRIDE_NPU_SOURCES})
# Mark .cpp files with special properties
set_source_files_properties(
    ${ROTARY_STRIDE_NPU_SOURCES} PROPERTIES
    LANGUAGE CXX
    COMPILE_FLAGS "--cce-soc-version=Ascend910B1 --cce-soc-core-type=VecCore --cce-auto-sync -xcce"
)

# Create object library
add_library(rotary_stride_objects OBJECT ${ROTARY_STRIDE_SOURCES})

target_compile_options(rotary_stride_objects PRIVATE ${COMMON_COMPILE_OPTIONS})
target_include_directories(rotary_stride_objects PRIVATE ${COMMON_INCLUDE_DIRS})
return()