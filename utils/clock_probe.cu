// utils/clock_probe.cu
// Measures, under a full FP32 FMA load on every SM:
//   1. the REAL SM clock, independently of nvidia-smi:
//      each block reads the SM cycle counter (clock64) and the GPU nanosecond
//      timer (%globaltimer) before and after its work -> f = cycles / ns;
//   2. the achieved FP32 throughput of a pure FMA loop (a "practical" compute peak),
//      compared with the theoretical peak at the measured clock.
//
// Usage: ./clock_probe [iters] [reps]   (defaults: 2000000, 10)

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "bench.h"
#include "cuda_check.h"

// FP32 lanes per SM: 128 on Ampere (8.6), Ada (8.9) and consumer Blackwell (12.x).
// Not valid for Turing / Volta / A100 (64). One FMA = 2 FLOP.
constexpr int FP32_LANES_PER_SM = 128;
constexpr int CHAINS = 8;  // independent FMA chains per thread (ILP hides FMA latency)

__device__ __forceinline__ unsigned long long globaltimer_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__global__ void fma_probe(float* out, int iters, unsigned long long* cycles,
                          unsigned long long* nanosecs) {
    // Values stay bounded: x <- 0.9999 * x + 0.0001 converges towards 1.
    const float b = 0.9999f, c = 0.0001f;
    float x[CHAINS];
#pragma unroll
    for (int j = 0; j < CHAINS; j++) x[j] = threadIdx.x * 1e-3f + j;

    __syncthreads();
    const unsigned long long c0 = clock64();
    const unsigned long long t0 = globaltimer_ns();

    for (int i = 0; i < iters; i++) {
#pragma unroll
        for (int j = 0; j < CHAINS; j++) x[j] = fmaf(x[j], b, c);
    }

    __syncthreads();
    const unsigned long long c1 = clock64();
    const unsigned long long t1 = globaltimer_ns();

    if (threadIdx.x == 0) {
        cycles[blockIdx.x] = c1 - c0;
        nanosecs[blockIdx.x] = t1 - t0;
    }
    // Write the result so the compiler cannot remove the loop.
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < CHAINS; j++) s += x[j];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

int main(int argc, char** argv) {
    const int iters = (argc > 1) ? atoi(argv[1]) : 2000000;
    const int reps = (argc > 2) ? atoi(argv[2]) : 10;
    if (iters <= 0 || reps <= 0) {
        fprintf(stderr, "usage: %s [iters > 0] [reps > 0]\n", argv[0]);
        return EXIT_FAILURE;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const int sms = prop.multiProcessorCount;
    const int threads = 256;
    // Fill every SM to its thread limit (1536 / 256 = 6 blocks per SM here).
    const int blocks = sms * (prop.maxThreadsPerMultiProcessor / threads);

    float* out_d;
    unsigned long long *cyc_d, *ns_d;
    CUDA_CHECK(cudaMalloc((void**)&out_d, (size_t)blocks * threads * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&cyc_d, blocks * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc((void**)&ns_d, blocks * sizeof(unsigned long long)));

    const BenchStats k = bench_kernel(
        [&] { fma_probe<<<blocks, threads>>>(out_d, iters, cyc_d, ns_d); }, reps, 1000.0);

    // Clock measured during the LAST timed launch (GPU warm and under load).
    std::vector<unsigned long long> cyc(blocks), ns(blocks);
    CUDA_CHECK(cudaMemcpy(cyc.data(), cyc_d, blocks * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ns.data(), ns_d, blocks * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    std::vector<double> mhz(blocks);
    for (int i = 0; i < blocks; i++) mhz[i] = (ns[i] > 0) ? 1e3 * (double)cyc[i] / ns[i] : 0.0;
    std::sort(mhz.begin(), mhz.end());
    const double f_mhz = mhz[blocks / 2];

    const double flop = 2.0 * CHAINS * (double)iters * blocks * threads;
    const double gflops = flop / (k.median_ms * 1e-3) / 1e9;
    const double peak_gflops = 2.0 * FP32_LANES_PER_SM * sms * f_mhz * 1e-3;

    printf("GPU               : %s (%d SMs)\n", prop.name, sms);
    printf("Launch            : %d blocks x %d threads, %d iters x %d FMA chains\n", blocks,
           threads, iters, CHAINS);
    printf("Kernel time       : median %.3f ms | min %.3f | max %.3f | stddev %.3f (%.1f %%)\n",
           k.median_ms, k.min_ms, k.max_ms, k.stddev_ms, 100.0 * k.stddev_ms / k.median_ms);
    printf("SM clock (measured): median %.0f MHz over blocks (min %.0f, max %.0f)\n", f_mhz,
           mhz.front(), mhz.back());
    printf("FMA throughput    : %.1f GFLOP/s\n", gflops);
    printf("Theoretical peak  : %.1f GFLOP/s at the measured clock (%d lanes x %d SMs x 2 x f)\n",
           peak_gflops, FP32_LANES_PER_SM, sms);
    printf("Efficiency        : %.1f %% of theoretical peak\n", 100.0 * gflops / peak_gflops);

    CUDA_CHECK(cudaFree(out_d));
    CUDA_CHECK(cudaFree(cyc_d));
    CUDA_CHECK(cudaFree(ns_d));
    return 0;
}
