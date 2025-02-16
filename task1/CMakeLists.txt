cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(SinusProject VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED TYPE)
    set(TYPE "float" CACHE STRING "Choose float or double")
endif()

add_executable(${PROJECT_NAME} main.cpp)

if(TYPE STREQUAL "double")
    message(STATUS "Compiling with double precision")
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_DOUBLE)
else()
    message(STATUS "Compiling with float precision")
endif()