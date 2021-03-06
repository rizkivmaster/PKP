#include <stdio.h>
#include <assert.h>
#include <cuda.h>

void incrementArrayOnHost(float *a, int N)
{
    int i;
    for (i = 0; i < N; i++)
        a[i] = a[i] + 1.f;
}

__global__ void incrementArrayOnDevice(float *a, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    //if (idx < N)
        a[idx] = 100.0;
}

int main(void)
{
    float *a_h, *b_h; // pointers to host memory
    float *a_d; // pointer to device memory
    int i, errcode, N = 10;
    size_t size = N * sizeof(float);

    // allocate arrays on host
    a_h = (float *) malloc(size);
    b_h = (float *) malloc(size);

    // allocate array on device
    fprintf(stdout, "Allocating array on device...\n");
    errcode = cudaMalloc((void **) &a_d, size);
    if (errcode != cudaSuccess)
    {
        fprintf(stderr, "Error invoking cudaMalloc (ERRCODE %d)\n", errcode);
        return 1;
    }
    else
    {
        fprintf(stdout, "cudaMalloc success\n");
    }

    // initialization of host data
    for (i = 0; i < N; i++)
        a_h[i] = (float) i;

    // copy data from host to device
    errcode = cudaMemcpy(a_d, a_h, sizeof(float) * N, cudaMemcpyHostToDevice);
    if (errcode != cudaSuccess)
    {
        fprintf(stderr, "Error invoking cudaMemcpy (ERRCODE %d)\n", errcode);
        return 1;
    }
    else
    {
        fprintf(stdout, "cudaMemcpy success\n");
    }

    // do calculation on host
    incrementArrayOnHost(a_h, N);

    // do calculation on device:
    // Part 1 of 2. Compute execution configuration
    dim3 grid(1, 1), block(1, N+1);
    // int blockSize = 4;
    // int nBlocks = N/blockSize + (N%blockSize == 0 ? 0 : 1);

    // Part 2 of 2. Call incrementArrayOnDevice kernel
    // incrementArrayOnDevice<<<nBlocks, blockSize>>>(a_d, N);
    fprintf(stdout, "Begin incrementing array on device\n");
    incrementArrayOnDevice<<<grid, block>>>(a_d, N);
    fprintf(stdout, "Finished incrementing array on device\n");

    // Retrieve result from device and store in b_h
    errcode = cudaMemcpy(b_h, a_d, sizeof(float) * N, cudaMemcpyDeviceToHost);
    if (errcode != cudaSuccess)
    {
        fprintf(stderr, "Error invoking cudaMemcpy (ERRCODE %d)\n", errcode);
        return 1;
    }
    else
    {
        fprintf(stdout, "cudaMemcpy success\n");
    }

    // check results
    for (i = 0; i < N; i++)
        // assert(a_h[i] == b_h[i]);
        printf("%f %f\n", a_h[i], b_h[i]);

    // cleanup
    free(a_h); free(b_h); cudaFree(a_d);
}

