#include <cuda_check.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "color2gray.cuh"

__global__ void col2gray_kernel(float* Pin, float* Pout, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        int grayOffset = row * width + col;
        int colOffset = grayOffset * 3;
        float r = Pin[colOffset];
        float g = Pin[colOffset + 1];
        float b = Pin[colOffset + 2];
        Pout[grayOffset] = 0.21 * r + 0.72 * g + 0.07 * b;
    }
}

void color2grayscale_gpu(float* Pin_h, float* Pout_h, int width, int height) {
    const size_t size = height * width * sizeof(float);
    const dim3 blocksize(32, 32);
    const dim3 gridsize((width + 32 - 1) / 32, (height + 32 - 1) / 32);

    // Device global memory allocation
    float *Pin_d, *Pout_d;
    CUDA_CHECK(cudaMalloc((void**)&Pin_d, size * 3));
    CUDA_CHECK(cudaMalloc((void**)&Pout_d, size));

    // Host to Device data transfert
    CUDA_CHECK(cudaMemcpy(Pin_d, Pin_h, size * 3, cudaMemcpyHostToDevice));

    // kernel execution
    col2gray_kernel<<<gridsize, blocksize>>>(Pin_d, Pout_d, width, height);

    // Error checking
    CUDA_CHECK(cudaGetLastError());

    // Device to Host data transfert
    CUDA_CHECK(cudaMemcpy(Pout_h, Pout_d, size, cudaMemcpyDeviceToHost));

    // Free Device allocated memory
    CUDA_CHECK(cudaFree(Pin_d));
    CUDA_CHECK(cudaFree(Pout_d));
}

void color2grayscale_cpu(float* Pin_h, float* Pout_h, int width, int height) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            int grayOffset = j * width + i;
            int colOffset = grayOffset * 3;
            float r = Pin_h[colOffset];
            float g = Pin_h[colOffset + 1];
            float b = Pin_h[colOffset + 2];
            Pout_h[grayOffset] = 0.21 * r + 0.72 * g + 0.07 * b;
        }
    }
}