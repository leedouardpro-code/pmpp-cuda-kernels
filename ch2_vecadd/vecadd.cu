%%cuda
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_check.h>
#include <vecadd.cuh>

__global__
void vecAddKernel(float* A,float* B,float* C, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N){
        C[i] = A[i] + B[i];
    }
}

void vec_add(float* A_h, float* B_h, float* C_h, int N){
    int size = N*sizeof(float);
    int gridsize = (N + size - 1)/size;
    int blocksize = 256;

    // Device global memory allocation
    float* A_d;
    float* C_d;
    float* B_d;

    CUDA_CHECK(cudaMalloc((void **)&A_d, size));
    CUDA_CHECK(cudaMalloc((void **)&B_d, size));
    CUDA_CHECK(cudaMalloc((void **)&C_d, size));

    // Host to Device data transfert
    CUDA_CHECK(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));

    // kernel execution
    vecAddKernel<<<gridsize, blocksize>>>(A_d, B_d, C_d, N);

    // Device to Host data transfert
    CUDA_CHECK(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
}
