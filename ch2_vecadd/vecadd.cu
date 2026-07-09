#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <chrono>

#define CUDA_CHECK(call)
    do {
        cudaError_t err = (call);
        if(err != cudaSuccess){
            fprintf(stderr, "CUDA error %s at %s:%d\n",
                cudaGetErrorString(err),__FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
    } while(0)

__global__
void vecAddKernel(float* A,float* B,float* C, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N){
        C[i] = A[i] + B[i];
    }
}

void vec_add(float* A_h, float* B_h, float* C_h, int N){
    int size = N*sizeof(float);

    // Device global memory allocation
    float* A_d;
    float* C_d;
    float* B_d;

    cudaMalloc((void **)&A_d, size);
    cudaMalloc((void **)&B_d, size);
    cudaMalloc((void **)&C_d, size);

    // Host to Device data transfert
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // kernel execution
    vecAddKernel<<<ceil(N/256.0), 256>>>(A_d, B_d, C_d);

    // Device to Host data transfert
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main(void){
    N = 1000;

    float* A_h, *B_h, *C_h;
    A_h = (float*)malloc(sizeof(float)*N);
    B_h = (float*)malloc(sizeof(float)*N);
    C_h = (float*)malloc(sizeof(float)*N);

    for(int i=0; i<N; i++){
        A_h[i]=1;
        B_h[i]=1;
    }

    vec_add(A_h, B_h, C_h, N);

    for(int i=0; i<N; i++){
        if(C_h[i]!=2){
            printf("error : unexpected result");
            break;
        }
    }
}