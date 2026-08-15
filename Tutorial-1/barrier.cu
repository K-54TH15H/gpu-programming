#include <stdio.h>
#include <cuda_runtime.h>

__device__ int count = 0;

__global__ void K1(int num_blocks) {
    if(threadIdx.x == 0)
	atomicAdd(&count, 1);
    
    __syncthreads();
    
    if(threadIdx.x == 0) 
	printf("Block %d Waiting...\n", blockIdx.x);

    // spin wait
    while(count < num_blocks);
    
    if(threadIdx.x == 0)
	printf("All Threads Synced\n");
}

int main() {
    // launch kernel
    K1<<<10, 32>>>(10);
    
    // synchronize kernel
    cudaDeviceSynchronize();

    return 0;
}
