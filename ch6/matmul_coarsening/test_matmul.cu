#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "matmul.cuh"

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

    const int cases[][3] = {
        {1, 1, 1},       {33, 65, 17},   {17, 33, 65},  {64, 1, 64},     {1, 64, 64},
        {64, 64, 1},     {31, 32, 33},   {257, 129, 511}, {1024, 1024, 1024},
    };

    int failures = 0;
    for (const auto& c : cases) failures += check(c[0], c[1], c[2]);

    printf("%s: %d failure(s)\n", failures ? "FAILED" : "ALL PASSED", failures);
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
