#include <stdio.h>
#include <string.h>
#include <cuda.h>

#define GRID_ROW_SIZE 1
#define GRID_COL_SIZE 1
#define GRID_DEP_SIZE 1
#define BLOCK_ROW_SIZE 2
#define BLOCK_COL_SIZE 3
#define BLOCK_DEP_SIZE 4

void checkCudaError(cudaError_t errorCode)
{
    if (errorCode != cudaSuccess)
        fprintf(stderr, "Error %d\n", errorCode);
}

__global__ void kernel(float *a, float *b, float *c)
{
    int numBlockThread = blockDim.x*blockDim.y*blockDim.z;

    int blockRowOffset = blockIdx.x*gridDim.y*gridDim.z*numBlockThread;
    int blockColOffset = blockIdx.y*gridDim.z*numBlockThread;
    int blockDepOffset = blockIdx.z*numBlockThread;
    int blockPos = blockRowOffset + blockColOffset + blockDepOffset;

    int threadRowOffset = threadIdx.x*blockDim.y*blockDim.z;
    int threadColOffset = threadIdx.y*blockDim.z;
    int threadDepOffset = threadIdx.z;
    int threadPos = threadRowOffset + threadColOffset + threadDepOffset;
    int idx = blockPos + threadPos;

    a[idx] = threadIdx.x;
    b[idx] = threadIdx.y;
    c[idx] = threadIdx.z;
}

__global__ void kernelWow(float *a)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    a[idx] = threadIdx.x;
}

int main(void)
{
    float *ha, *hb, *hc;          // host data
    float *da, *db, *dc;          // device data
    int N = GRID_ROW_SIZE * GRID_COL_SIZE * GRID_DEP_SIZE
          * BLOCK_ROW_SIZE * BLOCK_COL_SIZE * BLOCK_DEP_SIZE;
    int nbytes, i;

    nbytes = N * sizeof(float);
    ha = (float *) malloc(nbytes);
    hb = (float *) malloc(nbytes);
    hc = (float *) malloc(nbytes);
    checkCudaError(cudaMalloc((void **) &da, nbytes));
    checkCudaError(cudaMalloc((void **) &db, nbytes));
    checkCudaError(cudaMalloc((void **) &dc, nbytes));

    memset(ha, 0, nbytes);
    memset(hb, 0, nbytes);
    memset(hc, 0, nbytes);

    checkCudaError(cudaMemcpy(da, ha, nbytes, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(db, hb, nbytes, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(dc, hc, nbytes, cudaMemcpyHostToDevice));

    dim3 grid(GRID_ROW_SIZE, GRID_COL_SIZE, GRID_DEP_SIZE);
    dim3 block(BLOCK_ROW_SIZE, BLOCK_COL_SIZE, BLOCK_DEP_SIZE);
    kernel<<<grid, block>>>(da, db, dc);
    //kernelWow<<<grid, block>>>(da);

    checkCudaError(cudaMemcpy(ha, da, nbytes, cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(hb, db, nbytes, cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(hc, dc, nbytes, cudaMemcpyDeviceToHost));

    for (i = 0; i < N; i++)
        printf("[%d] = (%f, %f, %f)\n", i, ha[i], hb[i], hc[i]);

    return 0;
}
