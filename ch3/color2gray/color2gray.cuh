#ifndef COL2GRAY_CUH
#define COL2GRAY_CUH

__global__ void col2gray_kernel(float* Pin, float* Pout, int width, int height);
void color2grayscale_gpu(float* Pin_h, float* Pout_h, int width, int height);
void color2grayscale_cpu(float* Pin_h, float* Pout_h, int width, int height);

#endif  // COL2GRAY_CUH