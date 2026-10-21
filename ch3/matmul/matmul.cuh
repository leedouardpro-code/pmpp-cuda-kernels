#ifndef MATMUL_CUH
#define MATMUL_CUH

// C = A * B, row-major. A: M x K, B: K x N, C: M x N.
__global__ void Matmul(const float* A, const float* B, float* C, int M, int N, int K);
// Host wrapper: allocates device memory, copies, launches, copies back.
void matmul_gpu(const float* A_h, const float* B_h, float* C_h, int M, int N, int K);

#endif  // MATMUL_CUH