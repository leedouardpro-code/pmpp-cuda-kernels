#ifndef VECADD_CUH
#define VECADD_CUH

__global__ void vecAddKernel(float* A, float* B, float* C, int N);
void vec_add(float* A_h, float* B_h, float* C_h, int N);

#endif // VECADD_CUH