#include <cuda_runtime.h>
#include <stdio.h>

__device__ int x = 0;

__global__ void K() {
	if(threadIdx.x == 0) {
		printf("Hello Block: %d, Thread: %d\n", blockIdx.x, threadIdx.x);
		atomicAdd(&x, 1);
	}
	
	while(atomicAdd(&x, 0) != 25);

	if(threadIdx.x == 0)
		printf("Escaped from spin wait! Block: %d\n", blockIdx.x);

	return ;
}

int main() {
	int num_blocks = 25;
	int num_threads = 1024;
	K<<<num_blocks, num_threads>>>();
	cudaDeviceSynchronize();

	return 0;
}

