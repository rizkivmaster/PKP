#include <stdio.h>
#include <assert.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cstdlib>
#include <ctime>
#include <iostream>

__global__ void kernel(int * a)
{
	int idx = blockIdx.x*blockDim.x +threadIdx.x;
	a[idx] = 7;
}


__global__ void kernelByThreadId(int * a)
{
	int idx = blockIdx.x*blockDim.x +threadIdx.x;
	a[idx] = threadIdx.x;
}

__global__ void kernelByBlockId(int * a)
{
	int idx = blockIdx.x*blockDim.x +threadIdx.x;
	a[idx] = blockIdx.x;
}



int main(void)
{
	int N = 10;
	int *host_vector;
	int *dev_vector;


	dim3 blockSize(3,3,3);
	dim3 gridSize(1,1);

	//first experiment
	host_vector = (int*) malloc(N*sizeof(int));

	for(int ii = 0; ii < N ;ii++) host_vector[ii] = 0;

	cudaMalloc((void**)&dev_vector,N*sizeof(int));

	cudaMemcpy(dev_vector,host_vector,N*sizeof(int),cudaMemcpyHostToDevice);

	kernel<<<gridSize,blockSize>>>(dev_vector);

	cudaMemcpy(host_vector,dev_vector,N*sizeof(int),cudaMemcpyDeviceToHost);

	printf("first experiment\n");

	for(int ii = 0; ii < N ;ii++)	printf("%d \n",host_vector[ii]);
	
	free(host_vector);
	cudaFree(dev_vector);

	//second experiment
	host_vector = (int*) malloc(N*sizeof(int));

	for(int ii = 0; ii < N ;ii++) host_vector[ii] = 0;

	cudaMalloc((void**)&dev_vector,N*sizeof(int));

	cudaMemcpy(dev_vector,host_vector,N*sizeof(int),cudaMemcpyHostToDevice);

	kernelByBlockId<<<gridSize,blockSize>>>(dev_vector);

	cudaMemcpy(host_vector,dev_vector,N*sizeof(int),cudaMemcpyDeviceToHost);

	printf("second experiment\n");

	for(int ii = 0; ii < N ;ii++)	printf("%d \n",host_vector[ii]);
	
	free(host_vector);
	cudaFree(dev_vector);

	//third experiment
	host_vector = (int*) malloc(N*sizeof(int));

	for(int ii = 0; ii < N ;ii++) host_vector[ii] = 0;
	cudaMalloc((void**)&dev_vector,N*sizeof(int));

	cudaMemcpy(dev_vector,host_vector,N*sizeof(int),cudaMemcpyHostToDevice);

	kernelByThreadId<<<gridSize,blockSize>>>(dev_vector);

	cudaMemcpy(host_vector,dev_vector,N*sizeof(int),cudaMemcpyDeviceToHost);

	printf("third experiment\n");

	for(int ii = 0; ii < N ;ii++)	printf("%d \n",host_vector[ii]);
	
	free(host_vector);
	cudaFree(dev_vector);
}