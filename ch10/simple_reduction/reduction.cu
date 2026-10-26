#include <cuda_check.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "reduction.cuh"

// Block b reduces the segment input[b * SEGMENT_SIZE .. (b+1) * SEGMENT_SIZE - 1]
// (elements beyond n count as 0), then one thread adds the partial sum to *output
// with atomicAdd (PMPP 10.7).
__global__ void Reduction(const float* input, float* scratch, float* output, int n) {
    unsigned int i = threadIdx.x * 2 + blockIdx.x * SEGMENT_SIZE;
    if (i < n) scratch[i] = input[i];
    else scratch[i] = 0.0f;
    if (i + 1 < n) scratch[i+1] = input[i+1];
    else scratch[i+1] = 0.0f;

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 0) {
            scratch[i] += scratch[i + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x % blockDim.x == 0) atomicAdd(output, scratch[i]);
}

void reduction_launch(const float* input_d, float* scratch_d, float* output_d, int n) {
    const int blocksize = BLOCK_DIM;
    const int gridsize = (n + SEGMENT_SIZE - 1) / SEGMENT_SIZE;  // one block per segment
    // The blocks accumulate into *output_d with atomicAdd: it must start at 0.
    CUDA_CHECK(cudaMemsetAsync(output_d, 0, sizeof(float)));
    Reduction<<<gridsize, blocksize>>>(input_d, scratch_d, output_d, n);
}

float reduction_gpu(const float* input_h, int n) {
    const size_t size = (size_t)n * sizeof(float);

    // Device global memory allocation
    float *input_d, *scratch_d, *output_d;
    CUDA_CHECK(cudaMalloc((void**)&input_d, size));
    CUDA_CHECK(cudaMalloc((void**)&scratch_d, SCRATCH_SIZE(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&output_d, sizeof(float)));

    // Host to Device data transfert
    CUDA_CHECK(cudaMemcpy(input_d, input_h, size, cudaMemcpyHostToDevice));

    // kernel execution
    reduction_launch(input_d, scratch_d, output_d, n);

    // Error checking
    CUDA_CHECK(cudaGetLastError());

    // Device to Host data transfert
    float output_h;
    CUDA_CHECK(cudaMemcpy(&output_h, output_d, sizeof(float), cudaMemcpyDeviceToHost));

    // Free Device allocated memory
    CUDA_CHECK(cudaFree(input_d));
    CUDA_CHECK(cudaFree(scratch_d));
    CUDA_CHECK(cudaFree(output_d));
    return output_h;
}
