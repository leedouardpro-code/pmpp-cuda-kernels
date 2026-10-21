#ifndef MATMUL_CUH
#define MATMUL_CUH

#ifndef TILE_WIDTH
#define TILE_WIDTH 32

#define MATMUL_NAME "tiled"

#endif

__global__ void Matmul_tiling(const float* A, const float* B, float* C, int M, int N, int K);
void matmul_gpu(const float* A_h, const float* B_h, float* C_h, int M, int N, int K);
void matmul_launch(const float* A_d, const float* B_d, float* C_d, int M, int N, int K);
#endif