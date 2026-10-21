#include <cuda_check.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "matmul.cuh"

__global__ void Matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float Lval = 0;
        for (int k = 0; k < K; k++) {
            Lval += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = Lval;
    }
}

void matmul_gpu(const float* A_h, const float* B_h, float* C_h, int M, int N, int K) {
    const size_t sizeA = (size_t)M * K * sizeof(float);
    const size_t sizeB = (size_t)K * N * sizeof(float);
    const size_t sizeC = (size_t)M * N * sizeof(float);


    const dim3 blocksize(32, 32);
    const dim3 gridsize((N + blocksize.x - 1) / blocksize.x,
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
    Matmul<<<gridsize, blocksize>>>(A_d, B_d, C_d, M, N, K);

    // Error checking
    CUDA_CHECK(cudaGetLastError());

    // Device to Host data transfert
    CUDA_CHECK(cudaMemcpy(C_h, C_d, sizeC, cudaMemcpyDeviceToHost));

    // Free Device allocated memory
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
}
