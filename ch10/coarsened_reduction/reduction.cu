#include <cuda_check.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "reduction.cuh"

// Thread-coarsened sum reduction (PMPP fig. 10.15), extended to an arbitrary n
// (PMPP exercise 10.5).
// Block b reduces the segment input[b * SEGMENT_SIZE .. (b+1) * SEGMENT_SIZE - 1],
// with SEGMENT_SIZE = COARSE_FACTOR * 2 * BLOCK_DIM:
//   1. each thread adds its 2 * COARSE_FACTOR elements in a register, independently:
//      all threads are active, no barrier, no shared-memory access (PMPP 10.8);
//   2. the BLOCK_DIM partial sums are reduced in shared memory by the convergent
//      tree of PMPP 10.4 / 10.6;
//   3. thread 0 adds the block's sum to *output with atomicAdd (PMPP 10.7).
// scratch is not used: the input is read once from global memory and never modified.
__global__ void Reduction(const float* input, float* scratch, float* output, int n) {
    __shared__ float input_s[BLOCK_DIM];

    const unsigned int segment = blockIdx.x * SEGMENT_SIZE;
    const unsigned int t = threadIdx.x;
    const unsigned int i = segment + t;

    // 1. Coarsening loop. At each iteration, consecutive threads read consecutive
    //    addresses (coalesced, PMPP 6.1). Elements beyond n count as 0, the identity
    //    value of the sum (PMPP 10.1).
    float sum = 0.0f;
    for (unsigned int tile = 0; tile < 2 * COARSE_FACTOR; tile++) {
        const unsigned int idx = i + tile * BLOCK_DIM;
        if (idx < n) sum += input[idx];
    }
    input_s[t] = sum;

    // 2. Convergent reduction tree in shared memory
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) input_s[t] += input_s[t + stride];
    }

    // 3. One atomic addition per block
    if (t == 0) atomicAdd(output, input_s[0]);
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

    // Device global memory allocation (no scratch: this kernel does not need it)
    float *input_d, *output_d;
    CUDA_CHECK(cudaMalloc((void**)&input_d, size));
    CUDA_CHECK(cudaMalloc((void**)&output_d, sizeof(float)));

    // Host to Device data transfert
    CUDA_CHECK(cudaMemcpy(input_d, input_h, size, cudaMemcpyHostToDevice));

    // kernel execution
    reduction_launch(input_d, nullptr, output_d, n);

    // Error checking
    CUDA_CHECK(cudaGetLastError());

    // Device to Host data transfert
    float output_h;
    CUDA_CHECK(cudaMemcpy(&output_h, output_d, sizeof(float), cudaMemcpyDeviceToHost));

    // Free Device allocated memory
    CUDA_CHECK(cudaFree(input_d));
    CUDA_CHECK(cudaFree(output_d));
    return output_h;
}
