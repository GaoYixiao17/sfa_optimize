message(STATUS "BUILD_TORCH_OPS ON in init_routing")
# INIT_ROUTING operation sources
file(GLOB INIT_ROUTING_NPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

set(INIT_ROUTING_SOURCES ${INIT_ROUTING_NPU_SOURCES})
# Mark .cpp files with special properties
set_source_files_properties(
    ${INIT_ROUTING_NPU_SOURCES} PROPERTIES
    LANGUAGE CXX
    COMPILE_FLAGS "--cce-soc-version=Ascend910B1 --cce-soc-core-type=VecCore --cce-auto-sync -xcce"
)

# Create object library
add_library(init_routing_objects OBJECT ${INIT_ROUTING_SOURCES})

target_compile_options(init_routing_objects PRIVATE ${COMMON_COMPILE_OPTIONS})
target_include_directories(init_routing_objects PRIVATE ${COMMON_INCLUDE_DIRS})
return()