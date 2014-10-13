#include <stdio.h>
#include <cuda.h>


void checkCudaError(cudaError_t errorCode)
{
    if (errorCode != cudaSuccess)
        fprintf(stderr, "Error %d\n", errorCode);
}

int main(void)
{
    float *ha, *hb;     // host data
    float *da, *db;     // device data
    int N = 10, nbytes, i;

    nbytes = N * sizeof(float);
    ha = (float *) malloc(nbytes);
    hb = (float *) malloc(nbytes);
    checkCudaError(cudaMalloc((void **) &da, nbytes));
    checkCudaError(cudaMalloc((void **) &db, nbytes));

    for (i = 0; i < N; i++)
        ha[i] = 100.0 + i;

    checkCudaError(cudaMemcpy(da, ha, nbytes, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(db, da, nbytes, cudaMemcpyDeviceToDevice));
    checkCudaError(cudaMemcpy(hb, db, nbytes, cudaMemcpyDeviceToHost));

    for (i = 0; i < N; i++)
        printf("%f %f\n", ha[i], hb[i]);

    free(ha);
    free(hb);
    cudaFree(da);
    cudaFree(db);

    return 0;
}
