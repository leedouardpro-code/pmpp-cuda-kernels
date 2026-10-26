// test_vecadd_acc.cpp
// Correctness test for vec_add_acc(): C = A + B on N floats.
// Inputs are small integers, so every sum is exact in FP32 and the GPU result
// must equal the CPU result exactly. C is pre-filled with NaN, so an element the
// GPU forgets to write fails the test. Sizes include non-multiples of any
// plausible block size.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "vecadd_acc.h"

static int check(int N) {
    std::vector<float> A(N), B(N), C_gpu(N, NAN), C_cpu(N);
    for (int i = 0; i < N; i++) {
        A[i] = (float)(i % 1000);        // integers < 2^24: sums are exact in FP32
        B[i] = (float)(2 * (i % 777));
    }
    vec_add_acc(A.data(), B.data(), C_gpu.data(), N);
    vec_add_cpu(A.data(), B.data(), C_cpu.data(), N);

    for (int i = 0; i < N; i++) {
        if (!(C_gpu[i] == C_cpu[i])) {  // written this way so that NaN also fails
            printf("[FAIL] N = %9d: index %d, gpu = %f, cpu = %f\n", N, i, C_gpu[i], C_cpu[i]);
            return 1;
        }
    }
    printf("[PASS] N = %9d\n", N);
    return 0;
}

int main(void) {
    const int sizes[] = {1, 31, 32, 33, 255, 256, 257, 1000003, 1 << 24, (1 << 26) + 3};
    int failures = 0;
    for (int N : sizes) failures += check(N);
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
