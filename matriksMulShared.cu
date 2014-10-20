#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#define WIDTH 4
#define TILE_WIDTH 2

//Naive one block
__global__ void matmulShared(float* matA, float* matB, float* matC){
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

    // calculate thread id
    unsigned int col = TILE_WIDTH*blockIdx.x + threadIdx.x ;
    unsigned int row = TILE_WIDTH*blockIdx.y + threadIdx.y ;
    float cVal = 0;
    for (int m = 0 ; m<WIDTH/TILE_WIDTH ; m++ ) // m indicate number of phase
    {
        sA[threadIdx.y][threadIdx.x] =  matA[row*WIDTH + (m*TILE_WIDTH + threadIdx.x)]  ;
        sB[threadIdx.y][threadIdx.x] =  matB[ ( m*TILE_WIDTH + threadIdx.y) * WIDTH + col] ;
        __syncthreads() ; // for syncronizeing the threads
        // Do for tile
        for ( int k = 0; k<WIDTH ; k++ )
            cVal += sA[threadIdx.x][k] * sB[k][threadIdx.y] ;
        __syncthreads() ; // for syncronizeing the threads
    } 
	matC[row*WIDTH + col] = cVal;    
} 

void matriksMulShared(float* mA, float* mB, float* mC){
    //Device pointer 
    float* a_d, *b_d, *c_d;
    //Matriks size
    int size = WIDTH * WIDTH *sizeof(float) ;

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

    dim3 dimGrid(WIDTH/TILE_WIDTH, WIDTH/TILE_WIDTH);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    
    matmulShared<<<dimGrid,dimBlock>>>(a_d,b_d,c_d);

    cudaMemcpy(mC,c_d,size, cudaMemcpyDeviceToHost );

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}
 
int main(void){
	void matriksMulShared(float *, float *, float *);

	float* M, *N, *P;
	size_t size = WIDTH * WIDTH *sizeof(float);
    // allocate arrays on host
    M = (float *) malloc(size);
    N = (float *) malloc(size);
    P = (float *) malloc(size);

    // float M[width*width], N[width*width], P[width*width];
	for(int i = 0; i < (WIDTH*WIDTH) ; i++) {
        M[i] = i;
        N[i] = (WIDTH*WIDTH -1)- i;
        P[i] = 0.f;
        printf("%.3f %.3f %.3f\n", M[i], N[i], P[i]);
    }

    matriksMulShared(M, N, P);

    for(int i = 0; i < (WIDTH*WIDTH) ; i++) {
        if( i%WIDTH ==0){
        	 printf("\n");
        }
        printf("%.3f ", P[i]);
    }

    free(M); 
    free(N);
    free(P);
    return 0;
}
