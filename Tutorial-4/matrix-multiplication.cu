#include <stdio.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <curand_kernel.h>

#define CUDA_CHECK(expr_to_check) do { \
    cudaError_t err = expr_to_check;   \
    if(err != cudaSuccess)             \
        fprintf(stderr, "CUDA Runtime Error: %s:%i:%d = %s\n", __FILE__, __LINE__, err, cudaGetErrorString(err)); \
} while(0) \

#define WARP_SIZE 32
#define MATRIX_DIM 64
#define TILE 16

/* 
 * uncomment the below line to enable logging 
 * which prints the randomly generated matrices
 */

// #define LOGGING

__global__ void init(half* A, half* B, float* C, curandState* states) {
    int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;

    curand_init(42, thread_id, 0, &states[thread_id]);

    // let A and B be generated as it's slot identified by the thread_id
    A[thread_id] = curand_uniform(&states[thread_id]) * 50;
    B[thread_id] = curand_uniform(&states[thread_id]) * 50;
    
    // initialize result matrix to zero
    C[thread_id] = 0;
}

__global__ void K(half* A, half* B, float* C) {
    int warp_xid = blockIdx.x;
    int warp_yid = blockIdx.y;
    
    // declare fragments
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, TILE, TILE, TILE, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TILE, TILE, TILE, half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TILE, TILE, TILE, float> c_frag;

    // initialize output to zero
    nvcuda::wmma::fill_fragment(c_frag, 0.0);
    
    for(int indicator = 0; indicator < MATRIX_DIM; indicator += TILE) { 
        int offset_a = (warp_yid * MATRIX_DIM * TILE) + indicator;
        int offset_b = (indicator * MATRIX_DIM) + (warp_xid * TILE);
        
        // relatively adjusted matrices for A and B w.r.t warp
        half* r_A = &A[offset_a]; 
        half* r_B = &B[offset_b];

        // load the matrices into fragment
        nvcuda::wmma::load_matrix_sync(a_frag, r_A, MATRIX_DIM);
        nvcuda::wmma::load_matrix_sync(b_frag, r_B, MATRIX_DIM);

        // perform the matrix multiplication
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    int offset_c = (warp_yid * MATRIX_DIM * TILE) + (warp_xid * TILE);
    
    // relatively adjusted matrices for C wrt warp
    float* r_C = &C[offset_c];

    // store the matrix output
    nvcuda::wmma::store_matrix_sync(r_C, c_frag, MATRIX_DIM, nvcuda::wmma::mem_row_major);
}

int main() {
    printf("Matrix-Multiplication with Tensor Cores: (%dx%d) x (%dx%d)\n", MATRIX_DIM, MATRIX_DIM, MATRIX_DIM, MATRIX_DIM);

    half* devA;
    half* devB;
    float* devC;
    
    half hostA[MATRIX_DIM * MATRIX_DIM];
    half hostB[MATRIX_DIM * MATRIX_DIM]; 
    float hostC[MATRIX_DIM * MATRIX_DIM];
    
    // launch config 
    dim3 grid(MATRIX_DIM / TILE, MATRIX_DIM / TILE); // each tile is mapped to a warp
    dim3 block(1, 32); // 1-warp per block

    // device states for random generator in each thread
    curandState* devStates;

    // allocate memory for matrices on device memory
    CUDA_CHECK(cudaMalloc(&devA, MATRIX_DIM * MATRIX_DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&devB, MATRIX_DIM * MATRIX_DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&devC, MATRIX_DIM * MATRIX_DIM * sizeof(float)));

    // allocate states for each thread 
    CUDA_CHECK(cudaMalloc(&devStates, sizeof(curandState) * MATRIX_DIM * MATRIX_DIM));

    // check before kernel-call
    CUDA_CHECK(cudaGetLastError());

    // initialize the matrices  
    init<<<MATRIX_DIM, MATRIX_DIM>>>(devA, devB, devC, devStates);
    cudaDeviceSynchronize();
    
    // check after kernel-call 
    CUDA_CHECK(cudaGetLastError());

    // check for init kernel failures
    CUDA_CHECK(cudaGetLastError()); 
    
    CUDA_CHECK(cudaMemcpy(hostA, devA, sizeof(half) * MATRIX_DIM * MATRIX_DIM, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hostB, devB, sizeof(half) * MATRIX_DIM * MATRIX_DIM, cudaMemcpyDeviceToHost));
    
    // check for errors on copy call
    CUDA_CHECK(cudaGetLastError());

#ifdef LOGGING 
    // print matrix A
    printf("\n--- Matrix A ---\n");
    for(int i = 0; i < MATRIX_DIM; i++) {
        for(int j = 0; j < MATRIX_DIM; j++)
            // type cast to convert half to native float type
            printf("%0.1f ", (float) hostA[i * MATRIX_DIM + j]);
        printf("\n");
    }

    // print matrix B
    printf("\n--- Matrix B ---\n");
    for(int i = 0; i < MATRIX_DIM; i++) {
        for(int j = 0; j < MATRIX_DIM; j++)
            // type cast to convert half to native float type
            printf("%0.1f ", (float) hostB[i * MATRIX_DIM + j]);
        printf("\n");
    }
#endif

    printf("\n--- Performing Matrix Multiplication ---\n");
    
    // check before kernel-call 
    CUDA_CHECK(cudaGetLastError());

    // perform computation
    K<<<grid, block>>>(devA, devB, devC);
    cudaDeviceSynchronize();
    
    // check after kernel-call 
    CUDA_CHECK(cudaGetLastError());

    // copy result to host
    CUDA_CHECK(cudaMemcpy(hostC, devC, sizeof(float) * MATRIX_DIM * MATRIX_DIM, cudaMemcpyDeviceToHost));

    // check for matrix-multiplication kernel/copy failures 
    CUDA_CHECK(cudaGetLastError());
    
#ifdef LOGGING
    // print matrix C
    printf("\n--- Matrix C ---\n");
    for(int i = 0; i < MATRIX_DIM; i++) {
        for(int j = 0; j < MATRIX_DIM; j++)
            // type cast to convert half to native float type
            printf("%0.1f ", (float) hostC[i * MATRIX_DIM + j]);
        printf("\n");
    }   
#endif

    bool valid = true;

    // verification of result
    for(int i = 0; i < MATRIX_DIM; i++) {
        for(int j = 0; j < MATRIX_DIM; j++) {
            float value = 0;
            float expected = (float) hostC[i * MATRIX_DIM + j];

            for(int k = 0; k < MATRIX_DIM; k++) {
                float x = (float) hostA[i * MATRIX_DIM + k];
                float y = (float) hostB[k * MATRIX_DIM + j];
                
                value += x * y;
            }
            
            if(abs(value - expected) > 1e-1) {
                printf("\n[ERROR] Output differs at C[%d,%d] | Expected: %f | Calculated: %f\n", i, j, value, expected);
                valid = false;
            }
        }
    } 
    
    // summary of result and verification
    if(valid)
        printf("\nMatrix Multiplication has been verified with host generated solution\n");
    else
        printf("\nMatrix Multiplication has been found to be incorrect wrt to the host generated solution\n");

    // free up allocated memory on device 
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    cudaFree(devStates);

    // verify success of cleanup and graceful exit
    CUDA_CHECK(cudaGetLastError());

    printf("Exiting...\n");
    return 0;
}
