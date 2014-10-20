#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#define TILE_WIDTH 4

__global__ void matmulNBlok(float* matA, float* matB, float* matC, int width){
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float pVal = 0;
    for(int i=0; i<width; ++i){
       float elementMatA = matA[row*width+i];
       float elementMatB = matB[i*width+col];
       pVal += elementMatA * elementMatB; 
    }
    matC[threadIdx.y*width+threadIdx.x] = pVal;
}

void matriksMulHost(float* M, float* N, float* P, int Width){
    for(int i=0; i<Width; i++){
        for(int j=0; j<Width; j++){
            float sum = 0;
            for(int k=0; k<Width; k++){
                float a = M[i*Width + k];
                float b = N[k*Width + j];
                sum += a * b;
            }
            P[i*Width+j] = sum;
        }
    }
}

void matriksMulNBlok(float* mA, float* mB, float* mC, int width){
    //Device pointer 
    float* a_d, *b_d, *c_d;
    //Matriks size
    int size = width * width *sizeof(float) ;

    //allocate dan copy matriks a
    int cudaError = cudaMalloc((void**)&a_d, size);
    if (cudaError != cudaSuccess)
    {
        fprintf(stderr, "Error invoking cudaMemcpy (ERRCODE %d)\n", cudaError);
    }
    fprintf(stderr, "cudaMemcpy (ERRCODE %d)\n", cudaError);
    cudaMemcpy(a_d, mA, size , cudaMemcpyHostToDevice );

    //allocate dan copy matriks b
    cudaMalloc((void**)&b_d, size);
    cudaMemcpy(b_d, mB, size , cudaMemcpyHostToDevice );

    //allocate memory to device c
    cudaMalloc((void**)&c_d, size);

   dim3 dimGrid(TILE_WIDTH/width, TILE_WIDTH/width);
   dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    
    matmulNBlok<<<dimGrid,dimBlock>>>(a_d,b_d,c_d,width);

    cudaMemcpy(mC,c_d,size, cudaMemcpyDeviceToHost );

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}
 
int main(void){
	void matriksMulNBlok(float *, float *, float *, int);

	const int width = 4;
	float* M, *N, *P;
	size_t size = width * width *sizeof(float);
    // allocate arrays on host
    M = (float *) malloc(size);
    N = (float *) malloc(size);
    P = (float *) malloc(size);

    // float M[width*width], N[width*width], P[width*width];
	for(int i = 0; i < (width*width) ; i++) {
        M[i] = i;
        N[i] = width*width - i;
        P[i] = 0.f;
        printf("%.3f %.3f %.3f\n", M[i], N[i], P[i]);
    }

    matriksMulNBlok(M, N, P, width);

    for(int i = 0; i < (width*width) ; i++) {
        if( i%width ==0){
             printf("\n");
        }
        printf("%.3f ", P[i]);
    }

    free(M); 
    free(N);
    free(P);
    return 0;
}
