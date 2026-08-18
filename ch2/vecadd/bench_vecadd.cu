#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_check.h"
#include "vecadd.cuh"

static int performance() {
    const int N = 1 << 20;
    const size_t size = sizeof(float) * N;
    const int blocksize = 256;
    const int gridsize = (N + blocksize - 1) / blocksize;
    const int warmup = 2;

    // Host memory allocation
    float *A_h, *B_h, *C_h_gpu;
    A_h = (float*)malloc(size);
    B_h = (float*)malloc(size);
    C_h_gpu = (float*)malloc(size);

    // Init
    for (int i = 0; i < N; i++) {
        A_h[i] = (float)i;
        B_h[i] = (float)i;
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
        vecAddKernel<<<gridsize, blocksize>>>(A_d, B_d, C_d, N);
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
    vecAddKernel<<<gridsize, blocksize>>>(A_d, B_d, C_d, N);
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
    printf("\n===== Performance =====\n");
    printf("N = %d ,(%zu bytes per array)\n", N, size);
    printf("H2D (A+B) : %8.3f ms\n", ms_h2d);
    printf("Kernel    : %8.3f ms\n", ms_k);
    printf("D2H (C)   : %8.3f ms\n", ms_d2h);
    printf("Transfers : %5.1f %% of total\n", 100.0f * (ms_h2d + ms_d2h) / total);

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