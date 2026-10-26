#include <cuda_check.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "reduction.cuh"

// Block b reduces the segment input[b * SEGMENT_SIZE .. (b+1) * SEGMENT_SIZE - 1]
// (elements beyond n count as 0), then one thread adds the partial sum to *output
// with atomicAdd (PMPP 10.7).
__global__ void Reduction(const float* input, float* output, int n) {
    unsigned int i = threadIdx.x + blockIdx.x * SEGMENT_SIZE;
    unsigned int t = threadIdx.x;
    __shared__ float input_s [BLOCK_DIM];
    
    if (i + BLOCK_DIM < n) input_s[t] = input[i] + input[i + BLOCK_DIM];
    else if (i + BLOCK_DIM >= n && i < n) input_s[t] = input[i];
    else input_s[t] = 0.0f;

    for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (threadIdx.x < stride) {
            input_s[t] += input_s[t + stride];
        }
    }
    if (t % blockDim.x == 0) atomicAdd(output, input_s[t]);
}

void reduction_launch(const float* input_d, float* output_d, int n) {
    const int blocksize = BLOCK_DIM;
    const int gridsize = (n + SEGMENT_SIZE - 1) / SEGMENT_SIZE;  // one block per segment
    // The blocks accumulate into *output_d with atomicAdd: it must start at 0.
    CUDA_CHECK(cudaMemsetAsync(output_d, 0, sizeof(float)));
    Reduction<<<gridsize, blocksize>>>(input_d, output_d, n);
}

float reduction_gpu(const float* input_h, int n) {
    const size_t size = (size_t)n * sizeof(float);

    // Device global memory allocation
    float *input_d, *scratch_d, *output_d;
    CUDA_CHECK(cudaMalloc((void**)&input_d, size));
    CUDA_CHECK(cudaMalloc((void**)&output_d, sizeof(float)));

    // Host to Device data transfert
    CUDA_CHECK(cudaMemcpy(input_d, input_h, size, cudaMemcpyHostToDevice));

    // kernel execution
    reduction_launch(input_d, output_d, n);

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
