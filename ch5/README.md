# Chapter 5 — Memory architecture and data locality


## Exercises

### 1.

Consider matrix addition. Can one use shared memory to reduce the
global memory bandwidth consumption? Hint: Analyze the elements that
are accessed by each thread and see whether there is any commonality
between threads.

Since input data with the same index are not reused between threads, it is useless to transfer input data into shared memory.

### 2.

Draw the equivalent of Fig. 5.7 for a 8x8 matrix multiplication with 2x2
tiling and 4x4 tiling. Verify that the reduction in global memory bandwidth
is indeed proportional to the dimension size of the tiles.

In this example 2x2 (respectively 4x4), each input data is used twice (respectively four times). We then reduce the memory bandwidth twice (respectively four times).

### 3.

What type of incorrect execution behavior can happen if one forgot to use one or both `__syncthreads()` in the kernel of Fig. 5.9?

Without the first `__syncthreads()`, some threads might start the matrix multiplication without having all the input data in shared memory which will result in a false result.
Without the second, some threads might start loading new data in the shared memory while other threads are still doing the previous matrix multiplication.

### 4.

Assuming that capacity is not an issue for registers or shared memory, give one important reason why it would be valuable to use shared memory instead of registers to hold values fetched from global memory? Explain your answer.

If some calculation of the output data reuse the same input data, it is valuable to put the input data in the shared memory. Registers are private to each thread, so a value in a register cannot be shared. With shared memory, one thread loads a value from global memory once and all threads of the block can reuse it : the global load is done a single time for the whole block.

### 5.

For our tiled matrix-matrix multiplication kernel, if we use a 32x32 tile, what is the reduction of memory bandwidth usage for input matrices M and N?

The reduction of memory bandwidth is 32 times, equal to the tile dimension.

### 6.

Assume that a CUDA kernel is launched with 1000 thread blocks, each of which has 512 threads. If a variable is declared as a local variable in the kernel, how many versions of the variable will be created through the lifetime of the execution of the kernel?

Since local variables are private in each thread, we would have 512 000 versions.

### 7.

In the previous question, if a variable is declared as a shared memory variable, how many versions of the variable will be created through the lifetime of the execution of the kernel?

Since shared variables are private in each block, we would have 1000 versions.

### 8.

Consider performing a matrix multiplication of two input matrices with dimensions NxN. How many times is each element in the input matrices requested from global memory when:

a. There is no tiling?

each element is requested N times

b. Tiles of size TxT are used?

each element is requested N/T times

### 9.

A kernel performs 36 floating-point operations and seven 32-bit global memory accesses per thread. For each of the following device properties, indicate whether this kernel is compute-bound or memory bound.

a. Peak FLOPS=200 GFLOPS, peak memory bandwidth=100 GB/second

Ridge point : a = Peak FLOPS/peak memory bandwidth = 2 FLOP/B
Kernel arithmetic intensity : a = 36/((32/8)x7) = 1.28 FLOP/B

Since the kernel intensity is lower than the ridge point, the kernel is memory bound.

b. Peak FLOPS=300 GFLOPS, peak memory bandwidth=250 GB/second

Ridge point : a = Peak FLOPS/peak memory bandwidth = 1.2 FLOP/B

The kernel intensity (1.28) is higher than the new ridge point (1.2), so the kernel is compute bound.

### 10.

To manipulate tiles, a new CUDA programmer has written a device kernel that will transpose each tile in a matrix. The tiles are of size BLOCK_WIDTH by BLOCK_WIDTH, and each of the dimensions of matrix A is known to be a multiple of BLOCK_WIDTH. The kernel invocation and code are shown below. BLOCK_WIDTH is known at compile time and could be set anywhere from 1 to 20.
```cpp
dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH);
dim3 gridDim(A_width / blockDim.x, A_height / blockDim.y);
BlockTranspose<<<gridDim, blockDim>>>(A, A_width, A_height);

__global__ void
BlockTranspose(float* A_elements, int A_width, int A_height)
{
    __shared__ float blockA[BLOCK_WIDTH][BLOCK_WIDTH];

    int baseIdx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    baseIdx += (blockIdx.y * BLOCK_SIZE + threadIdx.y) * A_width;

    blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];

    A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y];
}
```
a. Out of the possible range of values for BLOCK_SIZE, for what values of BLOCK_SIZE will this kernel function execute correctly on the device?

There is no synchronization between the write into `blockA` and the transposed read. A thread reads `blockA[tx][ty]`, a value written by another thread, so the code is only correct when the whole block fits in a single warp and executes in lock-step. That means BLOCK_WIDTH² ≤ 32, so BLOCK_WIDTH from 1 to 5.
(Strictly, since Volta's independent thread scheduling, even a single warp is no longer guaranteed to be in lock-step without `__syncwarp()`.)

b. If the code does not execute correctly for all BLOCK_SIZE values, what is the root cause of this incorrect execution behavior? Suggest a fix to the code to make it work for all BLOCK_SIZE values.

Root cause : missing barrier between the shared-memory write and the transposed read. Add a `__syncthreads()` between the `blockA[...] = A_elements[...]` load and the `A_elements[...] = blockA[...]` read.

### 11.

Consider the following CUDA kernel and the corresponding host function that calls it:
```cpp
__global__ void foo_kernel(float* a, float* b) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    float x[4];
    __shared__ float y_s;
    __shared__ float b_s[128];
    for(unsigned int j = 0; j < 4; ++j) {
        x[j] = a[j*blockDim.x*gridDim.x + i];
    }
    if(threadIdx.x == 0) {
        y_s = 7.4f;
    }
    b_s[threadIdx.x] = b[i];
    __syncthreads();
    b[i] = 2.5f*x[0] + 3.7f*x[1] + 6.3f*x[2] + 8.5f*x[3]
         + y_s*b_s[threadIdx.x] + b_s[(threadIdx.x + 3)%128];
}
void foo(int* a_d, int* b_d) {
    unsigned int N = 1024;
    foo_kernel <<<(N + 128 - 1)/128, 128 >>>(a_d, b_d);
}
```
a. How many versions of the variable i are there?
Variable i is a local variable recreated in each thread. There are 1024 versions of i.

b. How many versions of the array x[] are there?
Array x is a local array recreated in each thread. There are 1024 versions of x.

c. How many versions of the variable y_s are there?
Variable y_s is a shared variable recreated in each block. There are 8 versions of y_s.

d. How many versions of the array b_s[] are there?
Array b_s is a shared array recreated in each block. There are 8 versions of b_s.

e. What is the amount of shared memory used per block (in bytes)?
Each block uses y_s (1 float) + b_s (128 floats) = 129 floats = 516 bytes.

f. What is the floating-point to global memory access ratio of the kernel (in OP/B)?
Global accesses per thread : 4 reads of a[], 1 read of b[i], 1 write of b[i] = 6 accesses × 4 B = 24 B.
FLOPs of the final expression : 5 multiplications + 5 additions = 10 FLOP.
Ratio = 10/24 ≈ 0.42 OP/B.
(b_s and y_s are shared memory, not global, so they don't count here.)

### 12.

Consider a GPU with the following hardware limits: 2048 threads/SM, 32 blocks/SM, 64K (65,536) registers/SM, and 96 KB of shared memory/SM. For each of the following kernel characteristics, specify whether the kernel can achieve full occupancy. If not, specify the limiting factor.

a. The kernel uses 64 threads/block, 27 registers/thread, and 4 KB of shared memory/block.
32 blocks/SM would need : 2048 threads, 55 296 registers, 128 KB of shared memory > 96 KB/SM.
Shared memory is the limiting factor : max 96/4 = 24 blocks/SM (1 536 threads).
The kernel can achieve at most 75% occupancy.

b. The kernel uses 256 threads/block, 31 registers/thread, and 8 KB of shared memory/block.
Threads limit : 8 blocks/SM → 2048 threads.
Registers : 8 × 256 × 31 = 63 488 ≤ 65 536.
Shared memory : 8 × 8 = 64 KB ≤ 96 KB.
The kernel can achieve 100% occupancy.