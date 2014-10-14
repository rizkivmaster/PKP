#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <cuda.h>

#define BLOCK_SIZE 4

void checkCudaError(cudaError_t errorCode)
{
    if (errorCode != cudaSuccess)
        fprintf(stderr, "Error %d\n", errorCode);
}

void incrementArrayOnHost(float *a, int size, int k)
{
    int i;
    for (i = 0; i < size; i++)
        a[i] += k;
}

__global__ void kernel(float *a, int size)
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

    if (idx < size)
        a[idx] += 1.0;
}

int main(void)
{
    float *ha, *hb;     // host data
    float *da;          // device data
    int N = 1000000;
    int nbytes, i;

    nbytes = N * sizeof(float);
    ha = (float *) malloc(nbytes);
    hb = (float *) malloc(nbytes);
    checkCudaError(cudaMalloc((void **) &da, nbytes));

    for (i = 0; i < N; i++)
        ha[i] = 100.0 + i;

    checkCudaError(cudaMemcpy(da, ha, nbytes, cudaMemcpyHostToDevice));

    incrementArrayOnHost(ha, N, 1.0);
    int nblocks = N/BLOCK_SIZE + (N%BLOCK_SIZE==0?0:1);
    dim3 grid(nblocks);
    dim3 block(BLOCK_SIZE);
    kernel<<<grid, block>>>(da, N);

    checkCudaError(cudaMemcpy(hb, da, nbytes, cudaMemcpyDeviceToHost));

    for (i = 0; i < N; i++)
        assert(ha[i] == hb[i]);

    for (i = 0; i < 10; i++)
        printf("%f %f\n", ha[i], hb[i]);

    return 0;
}
