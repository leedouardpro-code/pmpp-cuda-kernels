#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "color2gray.cuh"
#include "cuda_check.h"

static int performance() {
    int width = 2 << 12;
    int height = 2 << 11;
    int warmup = 2;
    const size_t size = sizeof(float) * width * height;
    const dim3 blocksize(32, 32);
    const dim3 gridsize((width + 32 - 1) / 32, (height + 32 - 1) / 32);

    // Host memory allocation
    float *Pin_h, *Pout_h_gpu, *Pout_h_cpu;
    Pin_h = (float*)malloc(size * 3);
    Pout_h_gpu = (float*)malloc(size);
    Pout_h_cpu = (float*)malloc(size);

    // Initialisation
    for (int i = 0; i < width * height; i++) {
        int j = i * 3;
        Pin_h[j] = (float)j;
        Pin_h[j + 1] = (float)(j + 1);
        Pin_h[j + 2] = (float)(j + 2);
    }

    // Device global memory allocation
    float *Pin_d, *Pout_d;
    CUDA_CHECK(cudaMalloc((void**)&Pin_d, size * 3));
    CUDA_CHECK(cudaMalloc((void**)&Pout_d, size));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms_h2d, ms_d2h, ms_k;

    // Warmup
    for (int i = 0; i < warmup; i++) {
        col2gray_kernel<<<gridsize, blocksize>>>(Pin_d, Pout_d, width, height);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Host to Device data transfert
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(Pin_d, Pin_h, size * 3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms_h2d, start, stop));

    // kernel execution
    CUDA_CHECK(cudaEventRecord(start));
    col2gray_kernel<<<gridsize, blocksize>>>(Pin_d, Pout_d, width, height);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms_k, start, stop));

    // Error checking
    CUDA_CHECK(cudaGetLastError());

    // Device to Host data transfert
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(Pout_h_gpu, Pout_d, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms_d2h, start, stop));

    // Free Device allocated memory
    CUDA_CHECK(cudaFree(Pin_d));
    CUDA_CHECK(cudaFree(Pout_d));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float total = ms_h2d + ms_k + ms_d2h;
    printf("\n===== Performance =====\n");
    printf("width = %d, height = %d,(%zu bytes)\n", width, height, size * 3);
    printf("H2D (A+B) : %8.3f ms\n", ms_h2d);
    printf("Kernel    : %8.3f ms\n", ms_k);
    printf("D2H (C)   : %8.3f ms\n", ms_d2h);
    printf("Transfers : %5.1f %% of total\n", 100.0f * (ms_h2d + ms_d2h) / total);

    free(Pin_h);
    free(Pout_h_gpu);
    free(Pout_h_cpu);

    return 0;
}

int main(void) { performance(); }