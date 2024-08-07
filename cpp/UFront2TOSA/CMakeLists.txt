cmake_minimum_required(VERSION 3.13.4)
set(CMAKE_C_COMPILER clang-16)
set(CMAKE_CXX_COMPILER clang++-16)

add_compile_options("-Wno-ambiguous-reversed-operator")
add_compile_options("-fexceptions")
# add_compile_options("-g")
# add_compile_options("-O0")


add_link_options("-fuse-ld=lld")


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")

project(ufront LANGUAGES CXX C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)

find_package(MLIR REQUIRED CONFIG)

if(NOT USEOMP)
    set(USEOMP ON)
endif()

if (USEOMP)
    find_package(OpenMP REQUIRED)
    add_compile_options("-fopenmp")
    add_link_options("-fopenmp")
endif()

option(USEMKL "use mkl to generate random numbers" OFF)
if (USEMKL)
    add_link_options(-lmkl_intel_lp64 -lmkl_core -lmkl_sequential)
endif()

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/third-party/cnpy)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
link_directories(/usr/local/lib)
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(third-party/cnpy)
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
