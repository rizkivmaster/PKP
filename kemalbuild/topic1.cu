#include <stdio.h>
#include <string.h>
#include <cuda.h>

#define GRID_ROW_SIZE 65535
#define GRID_COL_SIZE 65535
#define GRID_DEP_SIZE 65535
#define BLOCK_ROW_SIZE 1
#define BLOCK_COL_SIZE 10
#define BLOCK_DEP_SIZE 10

void checkCudaError(cudaError_t errorCode)
{
    if (errorCode != cudaSuccess)
        fprintf(stderr, "Error %d\n", errorCode);
}

__global__ void kernel(float *a, float *b, float *c, int size)
{
    long long numBlockThread = blockDim.x*blockDim.y*blockDim.z;

    long long blockRowOffset = blockIdx.x*gridDim.y*gridDim.z*numBlockThread;
    long long blockColOffset = blockIdx.y*gridDim.z*numBlockThread;
    long long blockDepOffset = blockIdx.z*numBlockThread;
    long long blockPos = blockRowOffset + blockColOffset + blockDepOffset;

    long long threadRowOffset = threadIdx.x*blockDim.y*blockDim.z;
    long long threadColOffset = threadIdx.y*blockDim.z;
    long long threadDepOffset = threadIdx.z;
    long long threadPos = threadRowOffset + threadColOffset + threadDepOffset;
    long long idx = blockPos + threadPos;

    if (idx < size)
    {
        a[idx] = threadIdx.x;
        b[idx] = threadIdx.y;
        c[idx] = threadIdx.z;
    }
}

__global__ void kernelWow(float *a)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    a[idx] = blockIdx.x;
}

int main(void)
{
    float *ha, *hb, *hc;          // host data
    float *da, *db, *dc;          // device data
    int N = 10;
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
    kernel<<<grid, block>>>(da, db, dc, N);
    //kernelWow<<<grid, block>>>(da);

    checkCudaError(cudaMemcpy(ha, da, nbytes, cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(hb, db, nbytes, cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(hc, dc, nbytes, cudaMemcpyDeviceToHost));

    for (i = 0; i < 10; i++)
        printf("[%d] = (%f, %f, %f)\n", i, ha[i], hb[i], hc[i]);

    return 0;
}
