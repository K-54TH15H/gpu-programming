// LOGGING to stdout uses a simple convention
// `---` - represents a normal expected execution
// `!!!` - represents a abnormal unexpected execution

#include <stdio.h>
#include <curand_kernel.h>
#include <string>

#define CUDA_CHECK(expr) do {                                            \
    cudaError_t result = expr;                                           \
    if(result != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA Runtime Error: %s:%i:%d = %s\n",           \
                __FILE__, __LINE__, result, cudaGetErrorString(result)); \
    }                                                                    \
} while(0)

#define MATRIX_DIM 1024
#define TILE 32

enum Mode {
    NAIVE,
    TILED
};

__global__ void init(int* A, int* B, int* C, curandState* states) {
    int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    // initialise curand state for that thread
    curand_init(42, thread_id, 0, &states[thread_id]);
    
    // generate random values for A and B between [0, 50]
    A[thread_id] = (int) (curand_uniform(&states[thread_id]) * 50);
    B[thread_id] = (int) (curand_uniform(&states[thread_id]) * 50);

    // initialize result matrix to zero
    C[thread_id] = 0;
}

// naive implementation of matrix-multiplication
__global__ void naive_matrix_multiplication(int* A, int* B, int* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // bounds check - not really required as TILE is a multiplicand of MATRIX_DIM by default
    // just in case if the macros are changed
    if(row < MATRIX_DIM && col < MATRIX_DIM) {
        int sum = 0;
        for(int i = 0; i < MATRIX_DIM; i++)
            sum += A[row * MATRIX_DIM + i] * B[i * MATRIX_DIM + col];

        // store the computed value
        C[(row * MATRIX_DIM) + col] = sum;
    }
}

// tiled implementation of matrix-multiplication using shared memory
__global__ void tiled_matrix_multiplication(int* A, int* B, int* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ int tile_A[TILE][TILE];
    __shared__ int tile_B[TILE][TILE];
    
    int num_tiles = (MATRIX_DIM + TILE - 1) / TILE;
    int sum = 0;

    for(int t = 0; t < num_tiles; t++) {
        int a_row = row;
        int a_col = t * TILE + threadIdx.x;

        // load A tile
        if(a_row < MATRIX_DIM && a_col < MATRIX_DIM)
            tile_A[threadIdx.y][threadIdx.x] = A[a_row * MATRIX_DIM + a_col];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0;

        int b_row = t * TILE + threadIdx.y;
        int b_col = col;

        // load B tile
        if(b_row < MATRIX_DIM && b_col < MATRIX_DIM)
            tile_B[threadIdx.y][threadIdx.x] = B[b_row * MATRIX_DIM + b_col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0;

        // sync to load the tile
        __syncthreads();
        
        // perform multiplication
        for(int k = 0; k < TILE; k++)
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];

        __syncthreads();
    }
    
    // store the computed result
    if(row < MATRIX_DIM && col < MATRIX_DIM)
        C[row * MATRIX_DIM + col] = sum;

}

// verifies the result of gpu matrix multiplication
bool verify(int* A, int* B, int* C) {
    for(int i = 0; i < MATRIX_DIM; i++) {
        for(int j = 0; j < MATRIX_DIM; j++) {
            int value = 0;
            for(int k = 0; k < MATRIX_DIM; k++)
                value += A[(i * MATRIX_DIM) + k] * B[(k * MATRIX_DIM) + j];
        
            // check error in computation
            if(value != C[(i * MATRIX_DIM) + j])
                return false;
        }
    }
    
    return true;
}

int main(int argc, const char** argv) {
    // simple help page
    if(argc != 2) {
        printf("-- HELP\n");
        printf("    CMD: ./matrix-multiplication <mode>\n");
        printf("\n--- MODES AVAILABLE\n");
        printf("    0: Naive matrix multiplication\n");
        printf("    1: Tiled matrix multiplication\n");
        printf("\n--- EXAMPLES\n");
        printf("    ./matrix-multiplication 0\n");
        printf("    ./matrix-multiplication 1\n");

        return 0;
    }
    
    // parse mode from args
    Mode mode;
    if(strcmp(argv[1], "0") == 0)
        mode = NAIVE;
    else if(strcmp(argv[1], "1") == 0)
        mode = TILED;
    else {
        printf("!!! Unknown mode: %s\n", argv[1]);
        printf("!!! Please look into the help page with ./matrix-multiplication");
        return 1;
    }

    printf("--- Matrix-Multiplication on GPU\n");
    int* d_A, *d_B, *d_C;

    int* h_A = new int[MATRIX_DIM * MATRIX_DIM];
    int* h_B = new int[MATRIX_DIM * MATRIX_DIM];
    int* h_C = new int[MATRIX_DIM * MATRIX_DIM];

    //device states for random generation
    curandState* d_states;

    // allocate matrices on device  
    CUDA_CHECK(cudaMalloc(&d_A, MATRIX_DIM * MATRIX_DIM * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_B, MATRIX_DIM * MATRIX_DIM * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_C, MATRIX_DIM * MATRIX_DIM * sizeof(int)));
    
    // allocate states on devices
    CUDA_CHECK(cudaMalloc(&d_states, sizeof(curandState) * MATRIX_DIM * MATRIX_DIM));   

    // initialize matrices
    init<<<MATRIX_DIM, MATRIX_DIM>>>(d_A, d_B, d_C, d_states);
    // check for init-kernel failures
    CUDA_CHECK(cudaGetLastError());
    
    // not really required to upper bound the config for TILE by default
    // as 32 (TILE) is a multiplicand of 1024 (MATRIX_DIM)
    dim3 block_cfg(TILE, TILE);
    dim3 grid_cfg((MATRIX_DIM + TILE - 1) / TILE, (MATRIX_DIM + TILE - 1) / TILE);
    
    if(mode == NAIVE)
        // multiply using naive implementation
        naive_matrix_multiplication<<<grid_cfg, block_cfg>>>(d_A, d_B, d_C);
    else
        // multiply using tiled implementation
        tiled_matrix_multiplication<<<grid_cfg, block_cfg>>>(d_A, d_B, d_C);

    // check for multiplication-kernel error
    CUDA_CHECK(cudaGetLastError()); 
    
    // copy back results    
    CUDA_CHECK(cudaMemcpy(h_A, d_A, sizeof(int) * MATRIX_DIM * MATRIX_DIM, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_B, d_B, sizeof(int) * MATRIX_DIM * MATRIX_DIM, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeof(int) * MATRIX_DIM * MATRIX_DIM, cudaMemcpyDeviceToHost));
    
    // verify the results on host
    if(verify(h_A, h_B, h_C))
        printf("--- Matrix Multiplication Verified Successfully\n");
    else
        printf("!!! Matrix Multiplication Error\n");

    printf("--- Exiting\n");
    
    // host clean-up    
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    // device clean-up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_states));
    
    return 0;
}
