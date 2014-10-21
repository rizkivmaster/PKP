#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <cuda.h>

#define N 1500
#define TILE_SIZE 4
#define NANO 1000000000

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
    for (i = 0; i < size; i++)
        for (j = 0; j < size; j++)
        {
            float sum = 0.0;
            for (k = 0; k < size; k++)
                sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
}

__global__ void multiplySquareSerializedMatOnDevice(float *C, float *A, float *B, int size)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if (i < size && j < size)
    {
        int k;
        float sum = 0.0;
        for (k = 0; k < size; k++)
            sum += A[i*size+k] * B[k*size+j];
        C[i*size+j] = sum;
    }
}

long long convertToNsec(struct timespec ts)
{
    long long tmp = (long long) ts.tv_sec*NANO + ts.tv_nsec;
    return tmp;
}

int main(void)
{
    float **ha, **hb, **hc, **hd;   // host data
    float *da, *db, *dc;            // device data
    int i, j;
    int nbytes = N * N * sizeof(float);
    long long elapsedTime;
    struct timespec ts_start, ts_end;

    // allocate memory in host
    ha = createSquareMatOnHost(N);
    hb = createSquareMatOnHost(N);
    hc = createSquareMatOnHost(N);
    hd = createSquareMatOnHost(N);

    // allocate memory in device
    checkCudaError(cudaMalloc((void **) &da, nbytes));
    checkCudaError(cudaMalloc((void **) &db, nbytes));
    checkCudaError(cudaMalloc((void **) &dc, nbytes));

    // set values in ha randomly
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            ha[i][j] = rand() % 10;

    // set values in hb randomly
    srand(time(NULL));
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            hb[i][j] = rand() % 10;

    // copy from host to device
    checkCudaError(cudaMemcpy(da, ha[0], nbytes, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(db, hb[0], nbytes, cudaMemcpyHostToDevice));

    // multiply matrix on host
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    multiplySquareMatOnHost(hd, ha, hb, N);
    clock_gettime(CLOCK_MONOTONIC, &ts_end);

    // compute elapsed time
    elapsedTime = convertToNsec(ts_end) - convertToNsec(ts_start);
    printf("CPU time: %f\n", (float) elapsedTime / NANO);

    // multiply matrix on device
    int gridSize = (N/TILE_SIZE) + (N%TILE_SIZE>0?1:0);
    dim3 grid(gridSize, gridSize), block(TILE_SIZE, TILE_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    multiplySquareSerializedMatOnDevice<<<grid, block>>>(dc, da, db, N);
    clock_gettime(CLOCK_MONOTONIC, &ts_end);

    // compute elapsed time
    elapsedTime = convertToNsec(ts_end) - convertToNsec(ts_start);
    printf("CUDA time: %f\n", (float) elapsedTime / NANO);

    // copy from device to host
    checkCudaError(cudaMemcpy(hc[0], dc, nbytes, cudaMemcpyDeviceToHost));

    // assertion
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            assert(hc[i][j] == hd[i][j]);

    // free memory
    freeSquareMatOnHost(ha);
    freeSquareMatOnHost(hb);
    freeSquareMatOnHost(hc);
    freeSquareMatOnHost(hd);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
}
