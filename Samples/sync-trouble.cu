#include <stdio.h>
#include <cuda_runtime.h>

__global__ void K() {
	printf("Executing...\n");

	if(threadIdx.x == 0)
		__syncthreads();

	printf("Returning...\n");
}

int main() {
	K<<<1, 64>>>();
	cudaDeviceSynchronize();

	return 0;
}
