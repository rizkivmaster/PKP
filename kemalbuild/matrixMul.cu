#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <cuda.h>

#define N 100
#define BLOCK_SIZE 1024

void checkCudaError(cudaError_t errorCode)
{
    if (errorCode != cudaSuccess)
    {
        fprintf(stderr, "Error %d\n", errorCode);
        exit(1);
    }
}

float** createSquareMatOnHost(int size)
{
    int i;
    float **mat;

    mat = (float **) malloc(size * sizeof(float *));
    if (!mat)
    {
        fprintf(stderr, "error allocating row memory");
        exit(1);
    }

    mat[0] = (float *) malloc(size * size * sizeof(float));
    if (!mat[0])
    {
        fprintf(stderr, "error allocating col memory");
        exit(1);
    }

    for (i = 1; i < size; i++)
        mat[i] = mat[i-1] + size;

    return mat;
}

void freeSquareMatOnHost(float **mat)
{
    free(mat[0]);
    free(mat);
}

void printSquareMat(float **mat, int size)
{
    int i, j;
    for (i = 0; i < size; i++, printf("\n"))
        for (j = 0; j < size; j++)
            printf(" %f", mat[i][j]);
}

void multiplySquareMatOnHost(float **C, float **A, float **B, int size)
{
    int i, j, k;
    memset(C[0], 0, size * size * sizeof(float));
    for (i = 0; i < size; i++)
        for (j = 0; j < size; j++)
            for (k = 0; k < size; k++)
                C[i][j] += A[i][k] * B[k][j];
}

__global__ void multiplySquareSerializedMatOnDevice(float *C, float *A, float *B, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size * size)
    {
        int i = idx / size;
        int j = idx % size;
        int k;
        for (k = 0; k < size; k++)
        {
            int idxa = i * size + k;
            int idxb = k * size + j;
            C[idx] += A[idxa] * B[idxb];
        }
    }
}

int main(void)
{
    float **ha, **hb, **hc, **hd;   // host data
    float *da, *db, *dc;            // device data
    int i, j;
    int nbytes = N * N * sizeof(float);

    // allocate memory in host
    ha = createSquareMatOnHost(N);
    hb = createSquareMatOnHost(N);
    hc = createSquareMatOnHost(N);
    hd = createSquareMatOnHost(N);

    // allocate memory in device
    checkCudaError(cudaMalloc((void **) &da, nbytes));
    checkCudaError(cudaMalloc((void **) &db, nbytes));
    checkCudaError(cudaMalloc((void **) &dc, nbytes));

    // initialize all values to zero
    memset(ha[0], 0, nbytes);
    memset(hb[0], 0, nbytes);
    memset(hc[0], 0, nbytes);
    memset(hd[0], 0, nbytes);

    // set values in ha randomly
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            ha[i][j] = rand() % 10;
    //printf("HA:\n");
    //printSquareMat(ha, N);

    // set values in hb randomly
    srand(time(NULL));
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            hb[i][j] = rand() % 10;
    //printf("HB:\n");
    //printSquareMat(hb, N);

    // copy from host to device
    checkCudaError(cudaMemcpy(da, ha[0], nbytes, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(db, hb[0], nbytes, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(dc, hc[0], nbytes, cudaMemcpyHostToDevice));

    // multiply matrix on host
    multiplySquareMatOnHost(hd, ha, hb, N);
    //printf("HD:\n");
    //printSquareMat(hd, N);

    // multiply matrix on device
    int gridSize = (N*N/BLOCK_SIZE) + ((N*N)%BLOCK_SIZE>0?1:0);
    dim3 grid(gridSize), block(BLOCK_SIZE);
    multiplySquareSerializedMatOnDevice<<<grid, block>>>(dc, da, db, N);

    // copy from device to host
    checkCudaError(cudaMemcpy(hc[0], dc, nbytes, cudaMemcpyDeviceToHost));
    //printf("CUDA result:\n");
    //printSquareMat(hc, N);

    // assertion
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            assert(hc[i][j] == hd[i][j]);

    freeSquareMatOnHost(ha);
    freeSquareMatOnHost(hb);
    freeSquareMatOnHost(hc);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
}
