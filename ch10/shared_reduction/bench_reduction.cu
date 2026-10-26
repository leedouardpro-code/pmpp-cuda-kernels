// bench_reduction.cu
// Kernel-only timing of reduction_launch(): sum of n floats.
// A reduction reads each element once and does one addition per 4 bytes: it is
// memory-bound (PMPP 5.1), so the metric is effective bandwidth = n * 4 bytes / time.
// Default n = 2^26 floats (256 MiB), far above the 32 MiB L2 cache.
//
// Usage: ./bench_reduction [log2_n] [reps]     (defaults: 26, 30)
// Last line of output is machine-readable (prefix "CSV,").
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "bench.h"
#include "cuda_check.h"
#include "reduction.cuh"

#define PEAK_GBPS 384.0  // RTX 5060 Laptop: 128-bit bus x 24 Gbps GDDR7
#define CSV_HEADER \
    "CSV_HEADER,kernel,block_dim,coarse,n,reps,median_ms,min_ms,max_ms,stddev_ms,gbps"

int main(int argc, char** argv) {
    const int log2n = (argc > 1) ? atoi(argv[1]) : 26;
    const int reps = (argc > 2) ? atoi(argv[2]) : 30;
    if (log2n < 10 || log2n > 28 || reps <= 0) {
        fprintf(stderr, "usage: %s [log2_n in 10..28] [reps > 0]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int n = 1 << log2n;
    const size_t size = (size_t)n * sizeof(float);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::vector<float> h(n);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    for (float& x : h) x = u01(rng);

    float *input_d, *scratch_d, *output_d;
    CUDA_CHECK(cudaMalloc((void**)&input_d, size));
    CUDA_CHECK(cudaMalloc((void**)&scratch_d, size));
    CUDA_CHECK(cudaMalloc((void**)&output_d, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(input_d, h.data(), size, cudaMemcpyHostToDevice));

    const BenchStats k =
        bench_kernel([&] { reduction_launch(input_d, scratch_d, n); }, reps);

    const double gbps = size / (k.median_ms * 1e-3) / 1e9;

    printf("GPU        : %s (%d SMs)\n", prop.name, prop.multiProcessorCount);
    printf("Kernel     : %s (BLOCK_DIM %d, COARSE_FACTOR %d)\n", REDUCTION_NAME, BLOCK_DIM,
           COARSE_FACTOR);
    printf("Size       : n = 2^%d = %d floats (%.0f MiB)\n", log2n, n, size / (1024.0 * 1024.0));
    printf("Kernel time: median %.4f ms | min %.4f | max %.4f | stddev %.4f (%.1f %%) | %d reps\n",
           k.median_ms, k.min_ms, k.max_ms, k.stddev_ms, 100.0 * k.stddev_ms / k.median_ms,
           k.reps);
    printf("Effective bandwidth: %.1f GB/s (n*4 bytes / median time), %.1f %% of %.0f GB/s\n",
           gbps, 100.0 * gbps / PEAK_GBPS, PEAK_GBPS);
    printf("%s\n", CSV_HEADER);
    printf("CSV,%s,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.1f\n", REDUCTION_NAME, BLOCK_DIM,
           COARSE_FACTOR, n, k.reps, k.median_ms, k.min_ms, k.max_ms, k.stddev_ms, gbps);

    CUDA_CHECK(cudaFree(input_d));
    CUDA_CHECK(cudaFree(scratch_d));
    CUDA_CHECK(cudaFree(output_d));
    return 0;
}
