#ifndef REDUCTION_CUH
#define REDUCTION_CUH

#ifndef BLOCK_DIM
#define BLOCK_DIM 1024
#endif
#ifndef COARSE_FACTOR
#define COARSE_FACTOR 1
#endif

#define SEGMENT_SIZE (2 * BLOCK_DIM * COARSE_FACTOR)
#define SCRATCH_SIZE(n) ((size_t)(((n) + SEGMENT_SIZE - 1) / SEGMENT_SIZE) * SEGMENT_SIZE)

#define REDUCTION_NAME "simple"

// Kernel. Launched with BLOCK_DIM threads per block, one block per segment.
//   input   : n floats, read-only (the benchmark reuses them between runs)
//   scratch : n floats of device memory, free to use (contents undefined);
//             for the in-place kernels of PMPP 10.3-10.5, which overwrite their input
//   output  : 1 float, set to 0 by reduction_launch() before the kernel
__global__ void Reduction(const float* input, float* scratch, float* output, int n);

// Host wrapper on HOST data: allocates, copies, launches, returns the sum.
float reduction_gpu(const float* input_h, int n);

// Same on DEVICE pointers, asynchronous (default stream). Zeroes *output_d, then launches.
// Used by both reduction_gpu() and the benchmark.
void reduction_launch(const float* input_d, float* scratch_d, float* output_d, int n);

#endif  // REDUCTION_CUH
