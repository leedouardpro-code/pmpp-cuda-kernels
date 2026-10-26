// Reference reduction: CUB DeviceReduce::Sum.
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include "cuda_check.h"
#include "reduction.cuh"

void reduction_launch(const float* input_d, float* scratch_d, float* output_d, int n) {
    (void)scratch_d;
    // CUB needs a temporary buffer whose size depends on n. A first call with a null
    // pointer only returns the required size. The buffer is (re)allocated only when a
    // larger one is needed, so allocation never happens inside a timed run.
    static void* temp_d = nullptr;
    static size_t temp_capacity = 0;

    size_t temp_bytes = 0;
    CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, temp_bytes, input_d, output_d, n));
    if (temp_bytes > temp_capacity) {
        if (temp_d) CUDA_CHECK(cudaFree(temp_d));
        CUDA_CHECK(cudaMalloc(&temp_d, temp_bytes));
        temp_capacity = temp_bytes;
    }
    // Writes the sum to *output_d (no need to zero it first).
    CUDA_CHECK(cub::DeviceReduce::Sum(temp_d, temp_bytes, input_d, output_d, n));
}

float reduction_gpu(const float* input_h, int n) {
    const size_t size = (size_t)n * sizeof(float);
    float *input_d, *output_d;
    CUDA_CHECK(cudaMalloc((void**)&input_d, size));
    CUDA_CHECK(cudaMalloc((void**)&output_d, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(input_d, input_h, size, cudaMemcpyHostToDevice));

    reduction_launch(input_d, nullptr, output_d, n);
    CUDA_CHECK(cudaGetLastError());

    float output_h;
    CUDA_CHECK(cudaMemcpy(&output_h, output_d, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(input_d));
    CUDA_CHECK(cudaFree(output_d));
    return output_h;
}
