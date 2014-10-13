#include <stdio.h>
#include <string.h>
#include <cuda.h>

#define GRID_ROW_SIZE 1
#define GRID_COL_SIZE 1
#define BLOCK_ROW_SIZE 20
#define BLOCK_COL_SIZE 1

void checkCudaError(cudaError_t errorCode)
{
    if (errorCode != cudaSuccess)
        fprintf(stderr, "Error %d\n", errorCode);
}

__global__ void kernel(float *a)
{
    int blockRowOffset = blockIdx.x*gridDim.y*blockDim.x*blockDim.y;
    int blockColOffset = blockIdx.y*blockDim.x*blockDim.y;
    int threadRowOffset = threadIdx.x*blockDim.y;
    int threadColOffset = threadIdx.y;
    int idx = blockRowOffset + blockColOffset + threadRowOffset + threadColOffset;
    a[idx] = threadIdx.x;
}

int main(void)
{
    float *ha;          // host data
    float *da;          // device data
    int N = GRID_ROW_SIZE * GRID_COL_SIZE * BLOCK_ROW_SIZE * BLOCK_COL_SIZE;
    int nbytes, i;

    nbytes = N * sizeof(float);
    ha = (float *) malloc(nbytes);
    checkCudaError(cudaMalloc((void **) &da, nbytes));

    memset(ha, 0, nbytes);

    checkCudaError(cudaMemcpy(da, ha, nbytes, cudaMemcpyHostToDevice));

    dim3 grid(GRID_ROW_SIZE, GRID_COL_SIZE);
    dim3 block(BLOCK_ROW_SIZE, BLOCK_COL_SIZE);
    kernel<<<grid, block>>>(da);

    checkCudaError(cudaMemcpy(ha, da, nbytes, cudaMemcpyDeviceToHost));

    for (i = 0; i < N; i++)
        printf("ha[%d] = %f\n", i, ha[i]);

    return 0;
}
