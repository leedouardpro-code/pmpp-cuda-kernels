#ifndef MATMUL_CUH
#define MATMUL_CUH

#define MATMUL_NAME "naive"

// C = A * B, row-major. A: M x K, B: K x N, C: M x N.
__global__ void Matmul(const float* A, const float* B, float* C, int M, int N, int K);
// Host wrapper: allocates device memory, copies, launches, copies back.
void matmul_gpu(const float* A_h, const float* B_h, float* C_h, int M, int N, int K);

// Launches the kernel on DEVICE pointers with this chapter's launch configuration.
// Asynchronous (no synchronization). Used by both matmul_gpu() and the bench,
// so tests and benchmarks always run exactly the same configuration.
void matmul_launch(const float* A_d, const float* B_d, float* C_d, int M, int N, int K);
#endif