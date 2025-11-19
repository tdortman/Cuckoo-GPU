cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(bulk_tcf VERSION 1.0 LANGUAGES CXX CUDA)


find_package(CUDAToolkit REQUIRED)

add_library(bulk_tcf INTERFACE)

target_include_directories(bulk_tcf INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
