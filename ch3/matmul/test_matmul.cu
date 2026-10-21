// test_matmul.cu
// Correctness test for matmul_gpu(): C = A * B, row-major,
// A: M x K, B: K x N, C: M x N.
// Same file for ch3/matmul, ch5/matmul_tiling and ch6/matmul_coarsening.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#include "matmul.cuh"

// CPU reference accumulated in double: its own rounding error is negligible
// compared with FP32, so any discrepancy is attributable to the GPU result.
// Loop order i-k-j keeps the inner loop contiguous in B and C (cache friendly).
static void matmul_ref(const std::vector<float>& A, const std::vector<float>& B,
                       std::vector<double>& C, int M, int N, int K) {
    std::fill(C.begin(), C.end(), 0.0);
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            const double a = A[(size_t)i * K + k];
            for (int j = 0; j < N; j++) {
                C[(size_t)i * N + j] += a * B[(size_t)k * N + j];
            }
        }
    }
}

// Tolerance: inputs are in [0, 1], so every product a*b is >= 0 and the classic
// dot-product error bound applies: |fl(sum) - sum| <= gamma_K * sum, with
// gamma_K ~= K * u and u = FLT_EPSILON / 2 (Higham, "Accuracy and Stability of
// Numerical Algorithms", ch. 3). We use K * FLT_EPSILON (2x margin).
// It is a worst-case bound: observed errors are usually far smaller, so the
// max relative error is printed as well.
static int check(int M, int N, int K) {
    std::vector<float> A((size_t)M * K), B((size_t)K * N), C((size_t)M * N);
    std::vector<double> ref((size_t)M * N);

    for (float& x : A) x = rand() / (float)RAND_MAX;
    for (float& x : B) x = rand() / (float)RAND_MAX;
    // Poison the output: an element the kernel forgets to write is detected.
    std::fill(C.begin(), C.end(), NAN);

    matmul_gpu(A.data(), B.data(), C.data(), M, N, K);
    matmul_ref(A, B, ref, M, N, K);

    const double rel_tol = (double)K * FLT_EPSILON;
    double max_rel = 0.0;
    for (size_t idx = 0; idx < C.size(); idx++) {
        const double err = std::fabs((double)C[idx] - ref[idx]);
        const double bound = rel_tol * std::fabs(ref[idx]);
        if (!(err <= bound)) {  // written this way so that NaN also fails
            printf("[FAIL] M=%4d N=%4d K=%4d at (row %zu, col %zu): gpu=%.8g ref=%.8g "
                   "abs.err=%.3g bound=%.3g\n",
                   M, N, K, idx / N, idx % N, C[idx], ref[idx], err, bound);
            return 1;
        }
        if (ref[idx] != 0.0) max_rel = std::max(max_rel, err / std::fabs(ref[idx]));
    }
    printf("[PASS] M=%4d N=%4d K=%4d  max rel. err = %.2e  (tol %.2e)\n", M, N, K, max_rel,
           rel_tol);
    return 0;
}

int main(void) {
    srand(42);  // fixed seed: reproducible inputs

    // {M, N, K}: degenerate, non-multiples of the tile size, thin shapes,
    // K smaller and larger than M/N, and a large square case.
    const int cases[][3] = {
        {1, 1, 1},       {33, 65, 17},   {17, 33, 65},  {64, 1, 64},     {1, 64, 64},
        {64, 64, 1},     {31, 32, 33},   {257, 129, 511}, {1024, 1024, 1024},
    };

    int failures = 0;
    for (const auto& c : cases) failures += check(c[0], c[1], c[2]);

    printf("%s: %d failure(s)\n", failures ? "FAILED" : "ALL PASSED", failures);
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
