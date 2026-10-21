// Statistical timing harness for GPU work, shared by every bench_*.cu.
//
// Method (see README "Method"):
//   1. Warmup for a minimum wall-clock time: a laptop GPU idles in a low
//      P-state, so the first launches run at low clocks. Warming up for a few
//      hundred ms lets the clocks ramp up (and loads the module, fills caches).
//   2. `reps` timed runs, each bracketed by its own pair of CUDA events.
//   3. Report median (main value, robust to outliers), min, max and stddev.
#ifndef BENCH_H
#define BENCH_H

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

#include "cuda_check.h"

struct BenchStats {
    float median_ms;
    float min_ms;
    float max_ms;
    float stddev_ms;  // sample standard deviation
    int reps;
};

// `launch` is any callable that enqueues the work to time on the default stream,
// e.g. [&] { my_kernel<<<grid, block>>>(args); }
template <typename Launch>
BenchStats bench_kernel(Launch launch, int reps = 30, double warmup_ms = 300.0) {
    // ---- Warmup: at least 3 launches AND at least warmup_ms of wall time ----
    const auto t0 = std::chrono::steady_clock::now();
    for (int n = 1;; ++n) {
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        const double elapsed =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                .count();
        if (n >= 3 && elapsed >= warmup_ms) break;
    }

    // ---- Timed runs ----
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> t(reps);
    for (int i = 0; i < reps; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));  // wait until the GPU reached `stop`
        CUDA_CHECK(cudaGetLastError());          // catch launch errors of this run
        CUDA_CHECK(cudaEventElapsedTime(&t[i], start, stop));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // ---- Statistics ----
    std::vector<float> s = t;
    std::sort(s.begin(), s.end());
    const float median = (reps % 2) ? s[reps / 2] : 0.5f * (s[reps / 2 - 1] + s[reps / 2]);

    double mean = 0.0;
    for (float x : t) mean += x;
    mean /= reps;
    double var = 0.0;
    for (float x : t) var += (x - mean) * (x - mean);
    var = (reps > 1) ? var / (reps - 1) : 0.0;

    return {median, s.front(), s.back(), (float)std::sqrt(var), reps};
}

#endif  // BENCH_H