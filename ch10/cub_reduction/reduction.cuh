#ifndef REDUCTION_CUH
#define REDUCTION_CUH

// Reference reduction: CUB cub::DeviceReduce::Sum (bundled with the CUDA toolkit),
// behind the same interface as ch10/reduction, so test_reduction.cu and
// bench_reduction.cu are reused unchanged.

// Not used by CUB (it picks its own configuration); defined for the shared test/bench.
#define BLOCK_DIM 0
#define COARSE_FACTOR 0
#define SEGMENT_SIZE 0
#define SCRATCH_SIZE(n) ((size_t)(n))

#define REDUCTION_NAME "cub"

float reduction_gpu(const float* input_h, int n);

// scratch_d is not used: CUB manages its own temporary storage (allocated on the
// first call, i.e. during the benchmark warmup, never in a timed run).
void reduction_launch(const float* input_d, float* scratch_d, float* output_d, int n);

#endif  // REDUCTION_CUH
