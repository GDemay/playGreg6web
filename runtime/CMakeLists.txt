cmake_minimum_required(VERSION 3.24)
project(playgta6_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(playgta6_core STATIC
    os/os_shim.cpp
    gpu/cmd_encoder.cpp
)
target_include_directories(playgta6_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(EMSCRIPTEN)
    add_executable(runtime main.cpp)
    target_link_libraries(runtime PRIVATE playgta6_core)
    set_target_properties(runtime PROPERTIES
        OUTPUT_NAME "runtime"
        SUFFIX ".js"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../web/dist"
    )
    target_compile_options(runtime PRIVATE -pthread -sMEMORY64=1)
    target_link_options(runtime PRIVATE
        -pthread
        -sMEMORY64=1
        -sPTHREAD_POOL_SIZE=4
        -sMODULARIZE=1
        -sEXPORT_NAME=createRuntimeModule
        -sALLOW_MEMORY_GROWTH=1
        -sENVIRONMENT=worker
        -sERROR_ON_UNDEFINED_SYMBOLS=0
    )
else()
    enable_testing()
    add_executable(runtime_tests ../tests/test_runtime_abi.cpp)
    target_link_libraries(runtime_tests PRIVATE playgta6_core)
    add_test(NAME runtime_abi_tests COMMAND runtime_tests)
endif()
