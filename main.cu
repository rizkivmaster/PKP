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
	a[idx] = blockIdx.x;
}

int main(void)
{
	std::srand(std::time(0));
	int N = 100;
	int *avector;
	int *dev_a;
	avector = (int*) malloc(N*sizeof(int));
	//for(int ii =9 ; ii < 100;ii++)
	//{
	//	avector[ii] = std::rand();
	//}

	cudaMalloc((void**)&dev_a,N*sizeof(int));

	kernel<<<20,5>>>(dev_a);

	cudaMemcpy(avector,dev_a,N*sizeof(int),cudaMemcpyDeviceToHost);

	for(int ii = 0; ii < N ;ii++)
	{
		printf("%d \n",avector[ii]);
	}
	free(avector);
	cudaFree(dev_a);
}