#include <iostream>
#include <cuda_runtime.h>

int main() {
    int device_count = 0;
    cudaError_t cuda_error;
    cuda_error = cudaGetDeviceCount(&device_count);

    if(cuda_error != cudaSuccess) {
	std::cerr << "[ERROR]: " << cudaGetErrorString(cuda_error) << std::endl;
	return 0;
    }

    std::cout << "No of devices found: " << device_count << std::endl;
    std::cout << std::endl;

    for(int i = 0; i < device_count; i++) {
	cudaDeviceProp prop;

	cudaGetDeviceProperties(&prop, i);
	
	std::cout << "=== DEVICE " << i << ' ' << prop.name << " ===" << std::endl;
	std::cout << std::endl;
	
	std::cout << "Concurrent Kernels: " << prop.concurrentKernels << std::endl;
	std::cout << "L2 Cache Size: " << prop.l2CacheSize << " Bytes" << std::endl;
	std::cout << "Max Blocks Per Multi-Processor: " << prop.maxBlocksPerMultiProcessor << std::endl;
	std::cout << "Max Grid Size: " << prop.maxGridSize << std::endl;
	std::cout << "Max Threads Dimension: " << prop.maxThreadsDim << std::endl;
	std::cout << "Max Threads Per Block: " << prop.maxThreadsPerBlock << std::endl;
	std::cout << "Max Threads Per Multi-Processor: " << prop.maxThreadsPerMultiProcessor << std::endl;
	std::cout << "Multi-Processor Count: " << prop.multiProcessorCount << std::endl;
	std::cout << "Registers Per Block: " << prop.regsPerBlock << std::endl;
	std::cout << "Registers Per Multi-Processor: " << prop.regsPerMultiprocessor << std::endl;
	std::cout << "Reserved Shared Memory Per Block: " << prop.reservedSharedMemPerBlock << std::endl;
	std::cout << "Shared Memory Per Block: " << prop.sharedMemPerBlock << " Bytes" << std::endl;
	std::cout << "Shared Memory Per Multi-Processor: " << prop.sharedMemPerMultiprocessor << " Bytes" << std::endl;
	std::cout << "Total Constant Memory: " << prop.totalConstMem << " Bytes" << std::endl;
	std::cout << "Total Global Memory: " << prop.totalGlobalMem << " Bytes" << std::endl;
	std::cout << "Warp Size: " << prop.warpSize << std::endl;

	std::cout << "Compute Capability: " << prop.major << '.' << prop.minor << std::endl;

	std::cout << std::endl;
    }
}
