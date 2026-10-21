#ifndef MATMUL_CUH
#define MATMUL_CUH

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif
#ifndef COARSE_FACTOR
#define COARSE_FACTOR 2
#endif

__global__ void Matmul_coarsening(const float* A, const float* B, float* C, int M, int N, int K);
void matmul_gpu(const float* A_h, const float* B_h, float* C_h, int M, int N, int K);
void matmul_cpu(const float* A_h, const float* B_h, float* C_h, int M, int N, int K);
int coarse_factor();

#endif