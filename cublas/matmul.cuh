#ifndef MATMUL_CUH
#define MATMUL_CUH

// Reference implementation: cuBLAS SGEMM, behind the same interface as the
// hand-written kernels, so test_matmul.cu and bench_matmul.cu are reused unchanged.

// Short label printed by bench_matmul
#define MATMUL_NAME "cublas"

// C = A * B, row-major. A: M x K, B: K x N, C: M x N. Host pointers.
void matmul_gpu(const float* A_h, const float* B_h, float* C_h, int M, int N, int K);

// Same computation on DEVICE pointers, asynchronous (default stream).
// The cuBLAS handle is created on the first call, i.e. during the bench warmup,
// never inside a timed run.
void matmul_launch(const float* A_d, const float* B_d, float* C_d, int M, int N, int K);

#endif  // MATMUL_CUH
