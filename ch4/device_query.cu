#include <cuda_check.h>
#include <cuda_runtime.h>
#include <stdio.h>

int main (void){
    cudaDeviceProp devProp;
    int devCount;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    for (unsigned int i = 0; i < devCount; i++) {
        CUDA_CHECK(cudaGetDeviceProperties(&devProp, i));
    }
}