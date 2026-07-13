#include <cuda_check.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "vecadd.cuh"

__global__ void vecAddKernel(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

void vec_add_gpu(float* A_h, float* B_h, float* C_h, int N) {
    const size_t size = N * sizeof(float);
    const int blocksize = 256;
    const int gridsize = (N + blocksize - 1) / blocksize;

    // Device global memory allocation
    float *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc((void**)&A_d, size));
    CUDA_CHECK(cudaMalloc((void**)&B_d, size));
    CUDA_CHECK(cudaMalloc((void**)&C_d, size));

    // Host to Device data transfert
    CUDA_CHECK(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));

    // kernel execution
    vecAddKernel<<<gridsize, blocksize>>>(A_d, B_d, C_d, N);

    // Error checking
    CUDA_CHECK(cudaGetLastError());

    // Device to Host data transfert
    CUDA_CHECK(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost));

    // Free Device allocated memory
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
}

void vec_add_cpu(float* A_h, float* B_h, float* C_h, int N) {
    for (int i = 0; i < N; i++) {
        C_h[i] = A_h[i] + B_h[i];
    }
}