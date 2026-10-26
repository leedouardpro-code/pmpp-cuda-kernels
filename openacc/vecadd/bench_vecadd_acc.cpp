// bench_vecadd_acc.cpp
// Timing of the OpenACC vector addition, same protocol and same CSV format as
// ch2/vecadd/bench_vecadd.cu: data copied to the GPU once, outside the timing;
// warmup, then the median of `reps` runs (utils/bench.h).
//
// Timing note: the OpenACC compute region is synchronous (the host waits for its
// end), and it may run on a CUDA stream chosen by the OpenACC runtime. The CUDA
// events of bench.h, recorded before and after the call, therefore bracket the
// whole region: the result is an upper bound on the kernel time (it includes the
// launch overhead, a few microseconds against ~2 ms at N = 2^26). Check it once
// against the kernel duration reported by Nsight Compute.
//
// Usage: ./bench_vecadd_acc [log2_N] [reps]   (defaults: 26, 30)
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "bench.h"
#include "cuda_check.h"
#include "vecadd_acc.h"

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

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::vector<float> A_h(N), B_h(N), C_h(N);
    for (int i = 0; i < N; i++) {
        A_h[i] = (float)i;
        B_h[i] = (float)(2 * i);
    }
    float* A = A_h.data();
    float* B = B_h.data();
    float* C = C_h.data();

    BenchStats k;
    // Data region: A and B copied to the GPU once, C allocated on the GPU only.
    // Inside, vec_add_acc_compute() finds the arrays already present.
#pragma acc data copyin(A[0:N], B[0:N]) create(C[0:N])
    {
        k = bench_kernel([&] { vec_add_acc_compute(A, B, C, N); }, reps);
    }

    const double bytes = 3.0 * size;  // 2 reads + 1 write
    const double gbps = bytes / (k.median_ms * 1e-3) / 1e9;

    printf("GPU        : %s\n", prop.name);
    printf("Kernel     : vecadd (OpenACC), N = 2^%d = %d floats (%.0f MiB per array)\n", log2n, N,
           size / (1024.0 * 1024.0));
    printf("Kernel time: median %.4f ms | min %.4f | max %.4f | stddev %.4f (%.1f %%) | %d reps\n",
           k.median_ms, k.min_ms, k.max_ms, k.stddev_ms, 100.0 * k.stddev_ms / k.median_ms,
           k.reps);
    printf("Effective bandwidth: %.1f GB/s (3*N*4 bytes / median time)\n", gbps);
    printf("%s\n", CSV_HEADER);
    printf("CSV,vecadd_openacc,%d,%d,%.4f,%.4f,%.4f,%.4f,%.1f\n", N, k.reps, k.median_ms, k.min_ms,
           k.max_ms, k.stddev_ms, gbps);
    return 0;
}
