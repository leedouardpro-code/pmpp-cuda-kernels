// bench_matmul.cu
// Kernel-only timing of matmul_launch(): C = A * B, square M = N = K = n.
// Same file for ch6/matmul, ch5/matmul_tiling and ch6/matmul_coarsening:
// each chapter's matmul.cuh provides MATMUL_NAME, matmul_launch() and,
// when relevant, TILE_WIDTH / COARSE_FACTOR.
//
// Usage: ./bench_matmul [n] [reps]      (defaults: n = 4096, reps = 30)
// Last line of output is machine-readable (prefix "CSV,"), see CSV_HEADER.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "bench.h"
#include "cuda_check.h"
#include "matmul.cuh"

#ifdef TILE_WIDTH
#define BENCH_TILE TILE_WIDTH
#else
#define BENCH_TILE 0  // naive kernel: no shared-memory tile
#endif
#ifdef COARSE_FACTOR
#define BENCH_COARSE COARSE_FACTOR
#else
#define BENCH_COARSE 1
#endif

#define CSV_HEADER "CSV_HEADER,kernel,tile,coarse,M,N,K,reps,median_ms,min_ms,max_ms,stddev_ms,gflops"

int main(int argc, char** argv) {
    const int n = (argc > 1) ? atoi(argv[1]) : 4096;
    const int reps = (argc > 2) ? atoi(argv[2]) : 30;
    if (n <= 0 || reps <= 0) {
        fprintf(stderr, "usage: %s [n > 0] [reps > 0]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int M = n, N = n, K = n;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // ---- Inputs: fixed seed, same data for every kernel compared ----
    const size_t sizeA = (size_t)M * K * sizeof(float);
    const size_t sizeB = (size_t)K * N * sizeof(float);
    const size_t sizeC = (size_t)M * N * sizeof(float);
    std::vector<float> A_h((size_t)M * K), B_h((size_t)K * N);
    srand(42);
    for (float& x : A_h) x = rand() / (float)RAND_MAX;
    for (float& x : B_h) x = rand() / (float)RAND_MAX;

    float *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc((void**)&A_d, sizeA));
    CUDA_CHECK(cudaMalloc((void**)&B_d, sizeB));
    CUDA_CHECK(cudaMalloc((void**)&C_d, sizeC));

    // ---- H2D transfer: context only (pageable host memory), not part of kernel time ----
    const BenchStats h2d = bench_kernel(
        [&] {
            CUDA_CHECK(cudaMemcpy(A_d, A_h.data(), sizeA, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(B_d, B_h.data(), sizeB, cudaMemcpyHostToDevice));
        },
        10, 0.0);

    // ---- Kernel ----
    const BenchStats k = bench_kernel([&] { matmul_launch(A_d, B_d, C_d, M, N, K); }, reps);

    // Standard GEMM convention (same as cuBLAS): 2*M*N*K FLOP (one FMA = 2 FLOP).
    const double flop = 2.0 * M * N * K;
    const double gflops = flop / (k.median_ms * 1e-3) / 1e9;

    printf("GPU        : %s (%d SMs)\n", prop.name, prop.multiProcessorCount);
    printf("Kernel     : %s (tile %d, coarse %d)\n", MATMUL_NAME, BENCH_TILE, BENCH_COARSE);
    printf("Size       : M = N = K = %d\n", n);
    printf("Kernel time: median %.3f ms | min %.3f | max %.3f | stddev %.3f (%.1f %%) | %d reps\n",
           k.median_ms, k.min_ms, k.max_ms, k.stddev_ms, 100.0 * k.stddev_ms / k.median_ms,
           k.reps);
    printf("Throughput : %.1f GFLOP/s (2*M*N*K / median time)\n", gflops);
    printf("H2D (A+B)  : median %.3f ms (pageable host memory, excluded from kernel time)\n",
           h2d.median_ms);
    printf("%s\n", CSV_HEADER);
    printf("CSV,%s,%d,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.1f\n", MATMUL_NAME, BENCH_TILE,
           BENCH_COARSE, M, N, K, k.reps, k.median_ms, k.min_ms, k.max_ms, k.stddev_ms, gflops);

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
    return 0;
}
