// test_reduction.cu
// Correctness test for reduction_gpu(): sum of n floats.
//
// Test 1, EXACT. Inputs are 0 or 1. Every partial sum is then an integer below 2^24,
//   which FP32 represents exactly, so the GPU result must equal the CPU count EXACTLY,
//   whatever the order of the additions. One missing, duplicated or out-of-bounds
//   element is detected. Sizes: 1, around powers of two, and sizes leaving a partial
//   last segment, up to 2^26 + 3.
// Test 2, ACCURACY. Uniform floats in [0, 1) on 2^26 elements, compared with a
//   double-precision CPU sum. Floating-point addition is not associative (PMPP 10.2),
//   so CPU and GPU results differ slightly; the relative error is printed and checked
//   against a loose bound (REL_TOL), Test 1 being the one that catches indexing bugs.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "reduction.cuh"

// Worst-case FP32 error of a sum of positive terms is about L*u times the sum, where
// L is the longest chain of additions an input goes through and u = 2^-24. With one
// atomicAdd per block, L can reach the number of blocks (~2^26 / 2048 = 32768, i.e.
// ~2e-3), while rounding errors largely cancel in practice (~sqrt(L)*u). 1e-4 sits
// between the two.
static const double REL_TOL = 1e-4;

static int check_exact(int n, std::mt19937& rng) {
    // Probability of a 1 such that the expected total is <= 2^22: every partial sum
    // stays far below 2^24 and is exact in FP32.
    const double p = std::min(0.5, (double)(1 << 22) / n);
    std::bernoulli_distribution bit(p);
    std::vector<float> h(n);
    long long expected = 0;
    for (float& x : h) {
        x = bit(rng) ? 1.0f : 0.0f;
        expected += (long long)x;
    }
    const float got = reduction_gpu(h.data(), n);
    if (got != (float)expected) {
        printf("[FAIL] exact  n = %9d: gpu = %.1f, expected %lld\n", n, got, expected);
        return 1;
    }
    printf("[PASS] exact  n = %9d: %lld\n", n, expected);
    return 0;
}

static int check_random(int n, std::mt19937& rng) {
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    std::vector<float> h(n);
    double ref = 0.0;
    for (float& x : h) {
        x = u01(rng);
        ref += x;
    }
    const float got = reduction_gpu(h.data(), n);
    const double rel = std::fabs((double)got - ref) / ref;
    if (!(rel <= REL_TOL)) {  // written this way so that NaN also fails
        printf("[FAIL] random n = %9d: gpu = %.7e, ref = %.7e, rel. err = %.2e (tol %.0e)\n", n,
               got, ref, rel, REL_TOL);
        return 1;
    }
    printf("[PASS] random n = %9d: rel. err = %.2e (tol %.0e)\n", n, rel, REL_TOL);
    return 0;
}

int main(void) {
    std::mt19937 rng(42);  // fixed seed: reproducible inputs
    printf("%s, BLOCK_DIM = %d, COARSE_FACTOR = %d, segment = %d elements\n", REDUCTION_NAME,
           BLOCK_DIM, COARSE_FACTOR, SEGMENT_SIZE);

    const int sizes[] = {1,    2,    3,    31,   32,   33,    1023,          1024,    1025,
                         2047, 2048, 2049, 4097, 1000, 12345, (1 << 20) + 3, 1 << 24, (1 << 26) + 3};
    int failures = 0;
    for (int n : sizes) failures += check_exact(n, rng);
    failures += check_random(1 << 26, rng);

    printf("%s: %d failure(s)\n", failures ? "FAILED" : "ALL PASSED", failures);
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
