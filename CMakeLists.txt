cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)
 
# Pass options to NVCC
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};)
 
# Specify include directories
include_directories(
/usr/local/cuda-5.0/include
)
 
# Specify library paths
link_directories(
/usr/local/cuda-5.0/lib
)
 
# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
main
main.cu
)
 
# For linking ...
# Specify target & libraries to link it with
#target_link_libraries(
#hellocuda
#-lcudaa
#)