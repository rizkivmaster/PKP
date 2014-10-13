cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)
 
# Pass options to NVCC
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};)
 
# Specify include directories
#include_directories(
#/usr/local/cuda-6.5/include
#)
 
# Specify library paths
#link_directories(
#/usr/local/cuda-6.5/lib
#)
 
# For compilation ...
cuda_add_executable(
task1
task1.cu
)

cuda_add_executable(
task2
task2.cu
)
