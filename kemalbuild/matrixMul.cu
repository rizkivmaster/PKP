#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define N 3

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

int main(void)
{
    float **ha, **hb, **hc;         // host data
    float *da, *db, *dc;         // device data
    int i, j;
    int nbytes = N * N * sizeof(float);

    // allocate memory in host
    ha = createSquareMatOnHost(N);
    hb = createSquareMatOnHost(N);
    hc = createSquareMatOnHost(N);

    // allocate memory in device
    checkCudaError(cudaMalloc((void **) &da, nbytes));
    checkCudaError(cudaMalloc((void **) &db, nbytes));
    checkCudaError(cudaMalloc((void **) &dc, nbytes));

    // initialize all values to zero
    memset(ha[0], 0, nbytes);
    memset(hb[0], 0, nbytes);
    memset(hc[0], 0, nbytes);

    // set ha as an identity matrix
    for (i = 0; i < N; i++)
        ha[i][i] = 1;

    // set values in hb randomly
    srand(time(NULL));
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            hb[i][j] = rand() % 10;

    // copy from host to device
    checkCudaError(cudaMemcpy(da, ha[0], nbytes, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(db, hb[0], nbytes, cudaMemcpyHostToDevice));

    // check
    checkCudaError(cudaMemcpy(hc[0], da, nbytes, cudaMemcpyDeviceToHost));
    printf("HA -> DA -> HC\n");
    for (i = 0; i < N; i++, printf("\n"))
        for (j = 0; j < N; j++)
            printf(" %f", hc[i][j]);

    checkCudaError(cudaMemcpy(hc[0], db, nbytes, cudaMemcpyDeviceToHost));
    printf("HB -> DB -> HC\n");
    for (i = 0; i < N; i++, printf("\n"))
        for (j = 0; j < N; j++)
            printf(" %f", hc[i][j]);

    freeSquareMatOnHost(ha);
    freeSquareMatOnHost(hb);
    freeSquareMatOnHost(hc);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
}
