#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cstdlib>
#include <ctime>
#include <iostream>

__global__ void matmul(float* matA, float* matB, float* matC, int width){
	float pVal = 0;
	for(int i=0; i<width; ++i){
		float elementMatA = matA[threadIdx.y*width+i];
		float elementMatB = matB[i*width+threadIdx.x];
		pVal += elementMatA * elementMatB; 
	}

	matC[threadIdx.y*width+threadIdx.x] = pVal;
}

void matriksMul(float* mA, float* mB, float* mC, int width){
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

    dim3 dimGrid(1, 1);
   	dim3 dimBlock(width, width);
    
    matmul<<<dimGrid,dimBlock>>>(a_d,b_d,c_d,width);

    cudaMemcpy(mC,c_d,size, cudaMemcpyDeviceToHost );

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}
 
int main(void){
	void matriksMul(float *, float *, float *, int);

	const int width = 10;
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
        // printf("%3f %3f %3f\n", M[i], N[i], P[i]);
    }

    matriksMul(M, N, P, width);

    for(int i = 0; i < (width*width) ; i++) {
        printf("%f", P[i]);
        if( i%width ==0){
        	 printf("\n");
        }
    }

    free(M); 
    free(N);
    free(P);
    return 0;
}