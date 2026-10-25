#ifndef MATMUL_CUH
#define MATMUL_CUH

#ifndef TILE_WIDTH
#define TILE_WIDTH 64
#endif
#ifndef COARSE_FACTOR
#define COARSE_FACTOR 4
#endif
#if TILE_WIDTH % COARSE_FACTOR != 0 
#error "TILE_WIDTH must be a multiple of COARSE_FACTOR"
#endif

#define MATMUL_NAME "register tiled 2D"

__global__ void Matmul_register_tiling(const float* A, const float* B, float* C, int M, int N, int K);
void matmul_gpu(const float* A_h, const float* B_h, float* C_h, int M, int N, int K);
void matmul_launch(const float* A_d, const float* B_d, float* C_d, int M, int N, int K);
#endif