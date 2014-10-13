#include <stdio.h>
#include <cuda.h>

#define GRID_ROW_SIZE 3
#define GRID_COL_SIZE 3
#define BLOCK_ROW_SIZE 2
#define BLOCK_COL_SIZE 2

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
    int blockRowOffset = blockIdx.x*gridDim.y*blockDim.x*blockDim.y;
    int blockColOffset = blockIdx.y*blockDim.x*blockDim.y;
    int threadRowOffset = threadIdx.x*blockDim.y;
    int threadColOffset = threadIdx.y;
    int idx = blockRowOffset + blockColOffset + threadRowOffset + threadColOffset;
    a[idx] = gridDim.x;
}

int main(void)
{
    float *ha, *hb;     // host data
    float *da;          // device data
    int N = GRID_ROW_SIZE * GRID_COL_SIZE * BLOCK_ROW_SIZE * BLOCK_COL_SIZE;
    int nbytes, i;

    nbytes = N * sizeof(float);
    ha = (float *) malloc(nbytes);
    hb = (float *) malloc(nbytes);
    checkCudaError(cudaMalloc((void **) &da, nbytes));

    for (i = 0; i < N; i++)
        ha[i] = 0.0;

    checkCudaError(cudaMemcpy(da, ha, nbytes, cudaMemcpyHostToDevice));

    // incrementArrayOnHost(ha, N, 1.0);
    dim3 grid(GRID_ROW_SIZE, GRID_COL_SIZE);
    dim3 block(BLOCK_ROW_SIZE, BLOCK_COL_SIZE);
    kernel<<<grid, block>>>(da, N);

    checkCudaError(cudaMemcpy(hb, da, nbytes, cudaMemcpyDeviceToHost));

    for (i = 0; i < N; i++)
        printf("hb[%d] = %f\n", i, hb[i]);

    return 0;
}
