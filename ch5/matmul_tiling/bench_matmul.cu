#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "matmul.cuh"
#include "cuda_check.h"

static int performance() {
    int width = 2 << 9;
    int height = 2 << 9;
    int warmup = 2;
    const size_t size =  (size_t)width * height * sizeof(float);
    const dim3 blocksize(32, 32);
    const dim3 gridsize((width + blocksize.x - 1) / blocksize.x, (height + blocksize.y - 1) / blocksize.y);

    // Host memory allocation
    float *A_h, *B_h, *C_h_gpu;
    A_h = (float*)malloc(size);
    B_h = (float*)malloc(size);
    C_h_gpu = (float*)malloc(size);

    // Initialisation
    for (int i = 0; i < width * height; i++) {
        A_h[i] = rand() / (float)RAND_MAX;
        B_h[i] = rand() / (float)RAND_MAX;
    }

    // Device memory allocation
    float *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc((void**)&A_d, size));
    CUDA_CHECK(cudaMalloc((void**)&B_d, size));
    CUDA_CHECK(cudaMalloc((void**)&C_d, size));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms_h2d, ms_d2h, ms_k;

    // Warmup
    for (int i = 0; i < warmup; i++) {
        Matmul_tiling<<<gridsize, blocksize>>>(A_d, B_d, C_d, width, height);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Host to Device data transfert
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms_h2d, start, stop));

    // Kernel execution
    CUDA_CHECK(cudaEventRecord(start));
    Matmul_tiling<<<gridsize, blocksize>>>(A_d, B_d, C_d, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms_k, start, stop));

    // Device to Host data transfert
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(C_h_gpu, C_d, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms_d2h, start, stop));

    float total = ms_h2d + ms_k + ms_d2h;
    int N = width;
    printf("\n===== Performance =====\n");
    printf("width = %d, height = %d,(%zu bytes)\n", width, height, (size * 3));
    printf("H2D (A+B) : %8.3f ms\n", ms_h2d);
    printf("Kernel    : %8.3f ms\n", ms_k);
    printf("D2H (C)   : %8.3f ms\n", ms_d2h);
    printf("Transfers : %5.1f %% of total\n", 100.0f * (ms_h2d + ms_d2h) / total);
    printf("Throughput: %8.3f\n", ((2*N-1)*N*N)*1e-9/ms_k);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
    free(A_h);
    free(B_h);
    free(C_h_gpu);

    return 0;
}
int main(void) { performance(); }