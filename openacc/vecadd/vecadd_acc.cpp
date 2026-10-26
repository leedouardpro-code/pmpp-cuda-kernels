#include "vecadd_acc.h"

void vec_add_acc_compute(const float* A, const float* B, float* C, int N) {
    // parallel loop : run the loop on the GPU; the programmer asserts that its
    //                 iterations are independent (the compiler does not check).
    // present(...)  : A, B and C are already in GPU memory, no transfer here.
#pragma acc parallel loop present(A[0:N], B[0:N], C[0:N])
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

void vec_add_acc(const float* A, const float* B, float* C, int N) {
    // Data region: A and B copied host -> GPU at its start, C allocated on the GPU
    // and copied GPU -> host at its end (copyout: C is never read on the GPU).
#pragma acc data copyin(A[0:N], B[0:N]) copyout(C[0:N])
    {
        vec_add_acc_compute(A, B, C, N);
    }
}

void vec_add_cpu(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}
