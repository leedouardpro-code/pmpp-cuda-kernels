#ifndef VECADD_CUH
#define VECADD_CUH

__global__ void Matmul_tiling(const float* A, const float* B, float* C, int width, int height);
void matmul_gpu(float* A_h, float* B_h, float* C_h, int width, int height);
void matmul_cpu(float* A_h, float* B_h, float* C_h, int width, int height);

#endif  // VECADD_CUH