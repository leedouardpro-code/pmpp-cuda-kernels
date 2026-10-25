#include <cuda_check.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "matmul.cuh"

__global__ void Matmul_register_tiling(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    float Pval[COARSE_FACTOR] = {0.0f};

    for (int ph = 0; ph < (K + TILE_WIDTH - 1) / TILE_WIDTH; ph++) {
        // Data transfert global memory to shared memory
        for(int i = 0; i < COARSE_FACTOR; i++) {
            if (row * COARSE_FACTOR + i < M && TILE_WIDTH * ph + tx < K){
                As[ty * COARSE_FACTOR + i][tx] = A[(row * COARSE_FACTOR + i) * K + TILE_WIDTH * ph + tx];
            }
            else As[ty * COARSE_FACTOR + i][tx] = 0.0f;

            if (ph * TILE_WIDTH + ty * COARSE_FACTOR + i < K && col < N)
                Bs[ty * COARSE_FACTOR + i][tx] = B[col + (ph * TILE_WIDTH + ty * COARSE_FACTOR + i) * N];
            else
                Bs[ty * COARSE_FACTOR + i][tx] = 0.0f;
        }

        // synchronizing read after write
        __syncthreads();

        // FLOPs sur les tuiles chargées en mémoires
        for (int c = 0; c < COARSE_FACTOR; c++) {
            for (int k = 0; k < TILE_WIDTH; k++) {
                Pval[c] += As[ty * COARSE_FACTOR + c][k] * Bs[k][tx];
            }
        }

        // synchronizing write after read
        __syncthreads();
    }
    for(int c = 0; c < COARSE_FACTOR; c++){
        if (row * COARSE_FACTOR + c < M && col < N) C[(row * COARSE_FACTOR + c)* N + col] = Pval[c];
    }
}

void matmul_gpu(const float* A_h, const float* B_h, float* C_h, int M, int N, int K) {
    const size_t sizeA = (size_t)M * K * sizeof(float);
    const size_t sizeB = (size_t)K * N * sizeof(float);
    const size_t sizeC = (size_t)M * N * sizeof(float);

    // Device global memory allocation
    float *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc((void**)&A_d, sizeA));
    CUDA_CHECK(cudaMalloc((void**)&B_d, sizeB));
    CUDA_CHECK(cudaMalloc((void**)&C_d, sizeC));

    // Host to Device data transfert
    CUDA_CHECK(cudaMemcpy(A_d, A_h, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, sizeB, cudaMemcpyHostToDevice));

    // kernel execution
    matmul_launch(A_d, B_d, C_d, M, N, K);

    // Error checking
    CUDA_CHECK(cudaGetLastError());

    // Device to Host data transfert
    CUDA_CHECK(cudaMemcpy(C_h, C_d, sizeC, cudaMemcpyDeviceToHost));

    // Free Device allocated memory
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
}

void matmul_launch(const float* A_d, const float* B_d, float* C_d, int M, int N, int K) {
    const dim3 blocksize(TILE_WIDTH, TILE_WIDTH/COARSE_FACTOR);
    const dim3 gridsize((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    Matmul_register_tiling<<<gridsize, blocksize>>>(A_d, B_d, C_d, M, N, K);
}