#include <cuda_check.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "matmul.cuh"
#define TILE_WIDTH 32

__global__ void Matmul_tiling(const float* A, const float* B, float* C, int width, int height) {
    __shared__ float M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float N[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    float Pval = 0;
    
    for (int ph = 0; ph < (width + TILE_WIDTH -1)/TILE_WIDTH ; ph ++){

        // Data transfert global memory to shared memory
        if (row < height && TILE_WIDTH * ph + tx < width) 
            M[ty][tx] = A[row * width + TILE_WIDTH * ph + tx];
        else M[ty][tx] = 0.0f;

        if (ph * TILE_WIDTH + ty < height && col < width)
            N[ty][tx] = B[col + (ph * TILE_WIDTH + ty) * width];
        else n[ty][tx] = 0.0f;
        
        // synchronizing read after write
        __syncthreads();

        // FLOPs sur les tuiles chargées en mémoires
        for(int k = 0; k < TILE_WIDTH; k++){
            Pval += M[ty][k] * N[k][tx];
        }

        // synchronizing write after read
        __syncthreads();
    }
    C[row * width + col] = Pval;
}


void matmul_gpu(float* A_h, float* B_h, float* C_h, int width, int height) {
    const size_t size = (size_t)width * height * sizeof(float);
    const dim3 blocksize(32, 32);
    const dim3 gridsize((width + blocksize.x - 1) / blocksize.x,
                        (height + blocksize.y - 1) / blocksize.y);

    // Device global memory allocation
    float *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc((void**)&A_d, size));
    CUDA_CHECK(cudaMalloc((void**)&B_d, size));
    CUDA_CHECK(cudaMalloc((void**)&C_d, size));

    // Host to Device data transfert
    CUDA_CHECK(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));

    // kernel execution
    Matmul_tiling<<<gridsize, blocksize>>>(A_d, B_d, C_d, width, height);

    // Error checking
    CUDA_CHECK(cudaGetLastError());

    // Device to Host data transfert
    CUDA_CHECK(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost));

    // Free Device allocated memory
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
}

void matmul_cpu(float* A_h, float* B_h, float* C_h, int width, int height) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            float Lval = 0;
            for (int k = 0; k < width; k++) {
                Lval += A_h[row * width + k] * B_h[k * width + col];
            }
            C_h[row * width + col] = Lval;
        }
    }
}