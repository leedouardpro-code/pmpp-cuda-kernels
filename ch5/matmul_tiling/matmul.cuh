#ifndef MATMUL_CUH
#define MATMUL_CUH

#ifndef TILE_WIDTH
#define TILE_WIDTH 32

#endif

__global__ void Matmul_tiling(const float* A, const float* B, float* C, int M, int N, int K);
void matmul_gpu(const float* A_h, const float* B_h, float* C_h, int M, int N, int K);

#endif