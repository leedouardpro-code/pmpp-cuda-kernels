// cuBLAS SGEMM reference for C = A * B with row-major arrays.
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "cuda_check.h"
#include "matmul.cuh"

#define CUBLAS_CHECK(call)                                                       \
    do {                                                                         \
        cublasStatus_t st = (call);                                              \
        if (st != CUBLAS_STATUS_SUCCESS) {                                       \
            fprintf(stderr, "cuBLAS error %s at %s:%d\n", cublasGetStatusString(st), \
                    __FILE__, __LINE__);                                         \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

// One handle for the whole process (creating it is expensive: never time it).
static cublasHandle_t get_handle() {
    static cublasHandle_t handle = [] {
        cublasHandle_t h;
        CUBLAS_CHECK(cublasCreate(&h));
        // Explicit default math mode: plain FP32 SGEMM. TF32 Tensor Core math is
        // only used if requested (CUBLAS_TF32_TENSOR_OP_MATH), so the comparison
        // with the FP32 hand-written kernels is fair.
        CUBLAS_CHECK(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));
        return h;
    }();
    return handle;
}

void matmul_launch(const float* A_d, const float* B_d, float* C_d, int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    // cuBLAS is column-major. A row-major X (r x c) is the column-major X^T (c x r).
    // So we ask cuBLAS for C^T = B^T * A^T in column-major, which is exactly
    // C = A * B in row-major: pass B first, then A, with m = N, n = M, k = K.
    CUBLAS_CHECK(cublasSgemm(get_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K, &alpha,
                             B_d, N,    // B^T: N x K column-major, leading dim N
                             A_d, K,    // A^T: K x M column-major, leading dim K
                             &beta,
                             C_d, N));  // C^T: N x M column-major, leading dim N
}

void matmul_gpu(const float* A_h, const float* B_h, float* C_h, int M, int N, int K) {
    const size_t sizeA = (size_t)M * K * sizeof(float);
    const size_t sizeB = (size_t)K * N * sizeof(float);
    const size_t sizeC = (size_t)M * N * sizeof(float);

    float *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc((void**)&A_d, sizeA));
    CUDA_CHECK(cudaMalloc((void**)&B_d, sizeB));
    CUDA_CHECK(cudaMalloc((void**)&C_d, sizeC));

    CUDA_CHECK(cudaMemcpy(A_d, A_h, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, sizeB, cudaMemcpyHostToDevice));

    matmul_launch(A_d, B_d, C_d, M, N, K);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(C_h, C_d, sizeC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
}
