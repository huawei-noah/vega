# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(classify_net)

# Compile options
add_compile_options(-std=c++11)

# Specify target generation path
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../../outputs")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  "../../outputs")
set(CMAKE_INSTALL_PREFIX "../../../run")
set(CMAKE_OUTPUT_DIR "out")


set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

# Header path
include_directories(
../
$ENV{DDK_PATH}/include/inc/
$ENV{DDK_PATH}/include/third_party/protobuf/include
$ENV{DDK_PATH}/include/third_party/cereal/include
$ENV{DDK_PATH}/include/libc_sec/include
)

if(type STREQUAL host)
    if(target STREQUAL "RC")
        add_compile_options(-DIS_RC)
    endif()

    # add host lib path
    link_directories($ENV{NPU_HOST_LIB})

    add_executable(main classify_net_main.cpp data_recv.cpp classify_net_host.cpp util.cpp  sample_data.cpp)


    if(target STREQUAL "RC")
        target_link_libraries(main matrixdaemon pthread c_sec dl rt)
    else()
        target_link_libraries(main matrix pthread c_sec dl rt)
    endif()

    install(TARGETS main DESTINATION ${CMAKE_OUTPUT_DIR})
else()
    add_compile_options(-DCPU_ONLY)

    # add device lib path
    link_directories($ENV{NPU_DEV_LIB})

    add_library(ai_engine SHARED classify_net_ai_engine.cpp sample_data.cpp)

    if(target STREQUAL "RC")
        target_link_libraries(ai_engine idedaemon hiai_common c_sec)
    else()
        target_link_libraries(ai_engine idedaemon c_sec)
    endif()

    install(TARGETS ai_engine DESTINATION ${CMAKE_OUTPUT_DIR})
endif()

