#ifndef REDUCTION_CUH
#define REDUCTION_CUH

// Sum reduction (PMPP chapter 10): *output = input[0] + ... + input[n-1].
// 1D input, 1D blocks, hierarchical (segmented multiblock) reduction as in PMPP 10.7:
// each block reduces one segment of the input and adds its partial sum to *output
// with atomicAdd.
//
// Single source of truth for the configuration; override at compile time,
// e.g. make EXTRA="-DBLOCK_DIM=512 -DCOARSE_FACTOR=4".
#ifndef BLOCK_DIM
#define BLOCK_DIM 1024     // threads per block (1D)
#endif
#ifndef COARSE_FACTOR
#define COARSE_FACTOR 1    // PMPP 10.8: elements per thread = 2 * COARSE_FACTOR
#endif

// Elements reduced by one block (PMPP: 2 * blockDim.x, times COARSE_FACTOR in 10.8)
#define SEGMENT_SIZE (2 * BLOCK_DIM * COARSE_FACTOR)

// Short label printed by bench_reduction (set it for each version you write)
#define REDUCTION_NAME "reduction"

// Kernel. Launched with BLOCK_DIM threads per block, one block per segment.
//   input   : n floats, read-only (the benchmark reuses them between runs)
//   scratch : n floats of device memory, free to use (contents undefined);
//             for the in-place kernels of PMPP 10.3-10.5, which overwrite their input
//   output  : 1 float, set to 0 by reduction_launch() before the kernel
__global__ void Reduction(const float* input, float* output, int n);

// Host wrapper on HOST data: allocates, copies, launches, returns the sum.
float reduction_gpu(const float* input_h, int n);

// Same on DEVICE pointers, asynchronous (default stream). Zeroes *output_d, then launches.
// Used by both reduction_gpu() and the benchmark.
void reduction_launch(const float* input_d, float* output_d, int n);

#endif  // REDUCTION_CUH
