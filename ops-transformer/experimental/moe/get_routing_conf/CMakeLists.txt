message(STATUS "BUILD_TORCH_OPS ON in get_routing_conf")
# GET_ROUTING_CONF operation sources
file(GLOB GET_ROUTING_CONF_NPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

set(GET_ROUTING_CONF_SOURCES ${GET_ROUTING_CONF_NPU_SOURCES})
# Mark .cpp files with special properties
set_source_files_properties(
    ${GET_ROUTING_CONF_NPU_SOURCES} PROPERTIES
    LANGUAGE CXX
    COMPILE_FLAGS "--cce-soc-version=Ascend910B1 --cce-soc-core-type=VecCore --cce-auto-sync -xcce"
)

# Create object library
add_library(get_routing_conf_objects OBJECT ${GET_ROUTING_CONF_SOURCES})

target_compile_options(get_routing_conf_objects PRIVATE ${COMMON_COMPILE_OPTIONS})
target_include_directories(get_routing_conf_objects PRIVATE ${COMMON_INCLUDE_DIRS})
return()