message(STATUS "BUILD_TORCH_OPS ON in gategelu_quant")
# GATEGELU_QUANT operation sources
file(GLOB GATEGELU_QUANT_NPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

set(GATEGELU_QUANT_SOURCES ${GATEGELU_QUANT_NPU_SOURCES})
# Mark .cpp files with special properties
set_source_files_properties(
    ${GATEGELU_QUANT_NPU_SOURCES} PROPERTIES
    LANGUAGE CXX
    COMPILE_FLAGS "--cce-soc-version=Ascend910B1 --cce-soc-core-type=VecCore --cce-auto-sync -xcce"
)

# Create object library
add_library(gategelu_quant_objects OBJECT ${GATEGELU_QUANT_SOURCES})

target_compile_options(gategelu_quant_objects PRIVATE ${COMMON_COMPILE_OPTIONS})
target_include_directories(gategelu_quant_objects PRIVATE ${COMMON_INCLUDE_DIRS})
return()