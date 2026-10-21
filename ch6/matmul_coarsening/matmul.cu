#include <cuda_check.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "matmul.cuh"

// natural coalescing + tiling + coarsening
__global__ void Matmul_coarsening(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];
    float Pval[COARSE_FACTOR];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = by * TILE_WIDTH + ty;
    int colStart = bx * TILE_WIDTH * COARSE_FACTOR + tx;

    for (int c = 0; c < COARSE_FACTOR; c++) {
        Pval[c] = 0.0f;
    }

    for (int ph = 0; ph < (K + TILE_WIDTH - 1) / TILE_WIDTH; ph++) {
        // Data transfert global memory to shared memory
        if (row < M && TILE_WIDTH * ph + tx < K)
            As[ty][tx] = A[row * K + TILE_WIDTH * ph + tx];
        else
            As[ty][tx] = 0.0f;

        for (int c = 0; c < COARSE_FACTOR; c++) {
            int col = colStart + c * TILE_WIDTH;
            if (ph * TILE_WIDTH + ty < K && col < N)
                Bs[ty][tx] = B[col + (ph * TILE_WIDTH + ty) * N];
            else
                Bs[ty][tx] = 0.0f;

            // synchronizing read after write
            __syncthreads();

            // FLOPs sur les tuiles chargées en mémoires
            for (int k = 0; k < TILE_WIDTH; k++) {
                Pval[c] += As[ty][k] * Bs[k][tx];
            }
            // synchronizing write after read
            __syncthreads();
        }
    }
    for (int c = 0; c < COARSE_FACTOR; c++) {
        int col = colStart + c * TILE_WIDTH;
        if (row < M && col < N) C[row * N + col] = Pval[c];
    }
}

void matmul_gpu(const float* A_h, const float* B_h, float* C_h, int M, int N, int K) {
    const size_t sizeA = (size_t)M * K * sizeof(float);
    const size_t sizeB = (size_t)K * N * sizeof(float);
    const size_t sizeC = (size_t)M * N * sizeof(float);

    const dim3 blocksize(TILE_WIDTH, TILE_WIDTH);
    const dim3 gridsize(
        ((((N + blocksize.x - 1) / blocksize.x) + COARSE_FACTOR - 1) / COARSE_FACTOR),
        (M + blocksize.y - 1) / blocksize.y);

    // Device global memory allocation
    float *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc((void**)&A_d, sizeA));
    CUDA_CHECK(cudaMalloc((void**)&B_d, sizeB));
    CUDA_CHECK(cudaMalloc((void**)&C_d, sizeC));

    // Host to Device data transfert
    CUDA_CHECK(cudaMemcpy(A_d, A_h, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, sizeB, cudaMemcpyHostToDevice));

    // kernel execution
    Matmul_coarsening<<<gridsize, blocksize>>>(A_d, B_d, C_d, M, N, K);

    // Error checking
    CUDA_CHECK(cudaGetLastError());

    // Device to Host data transfert
    CUDA_CHECK(cudaMemcpy(C_h, C_d, sizeC, cudaMemcpyDeviceToHost));

    // Free Device allocated memory
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
}
