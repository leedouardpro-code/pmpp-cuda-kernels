// bench_vecadd.cu
// Kernel-only timing of vecAddKernel: C = A + B on N floats.
// Memory-bound kernel: the relevant metric is effective bandwidth,
//   BW = (bytes read + bytes written) / time = 3 * N * 4 bytes / time.
// N must be much larger than the L2 cache (32 MiB on the RTX 5060 Laptop),
// otherwise we measure L2 bandwidth, not DRAM. Default N = 2^26 (256 MiB per array).
//
// Usage: ./bench_vecadd [log2_N] [reps]   (defaults: 26, 30)

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "bench.h"
#include "cuda_check.h"
#include "vecadd.cuh"

#define CSV_HEADER "CSV_HEADER,kernel,N,reps,median_ms,min_ms,max_ms,stddev_ms,gbps"

int main(int argc, char** argv) {
    const int log2n = (argc > 1) ? atoi(argv[1]) : 26;
    const int reps = (argc > 2) ? atoi(argv[2]) : 30;
    if (log2n < 0 || log2n > 28 || reps <= 0) {
        fprintf(stderr, "usage: %s [log2_N in 0..28] [reps > 0]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int N = 1 << log2n;
    const size_t size = (size_t)N * sizeof(float);
    const int blocksize = 256;
    const int gridsize = (N + blocksize - 1) / blocksize;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::vector<float> A_h(N), B_h(N);
    for (int i = 0; i < N; i++) {
        A_h[i] = (float)i;
        B_h[i] = (float)(2 * i);
    }

    float *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc((void**)&A_d, size));
    CUDA_CHECK(cudaMalloc((void**)&B_d, size));
    CUDA_CHECK(cudaMalloc((void**)&C_d, size));
    CUDA_CHECK(cudaMemcpy(A_d, A_h.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h.data(), size, cudaMemcpyHostToDevice));

    const BenchStats k =
        bench_kernel([&] { vecAddKernel<<<gridsize, blocksize>>>(A_d, B_d, C_d, N); }, reps);

    const double bytes = 3.0 * size;  // 2 reads + 1 write
    const double gbps = bytes / (k.median_ms * 1e-3) / 1e9;

    printf("GPU        : %s\n", prop.name);
    printf("Kernel     : vecadd, N = 2^%d = %d floats (%.0f MiB per array)\n", log2n, N,
           size / (1024.0 * 1024.0));
    printf("Kernel time: median %.4f ms | min %.4f | max %.4f | stddev %.4f (%.1f %%) | %d reps\n",
           k.median_ms, k.min_ms, k.max_ms, k.stddev_ms, 100.0 * k.stddev_ms / k.median_ms,
           k.reps);
    printf("Effective bandwidth: %.1f GB/s (3*N*4 bytes / median time)\n", gbps);
    printf("%s\n", CSV_HEADER);
    printf("CSV,vecadd,%d,%d,%.4f,%.4f,%.4f,%.4f,%.1f\n", N, k.reps, k.median_ms, k.min_ms,
           k.max_ms, k.stddev_ms, gbps);

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
    return 0;
}
