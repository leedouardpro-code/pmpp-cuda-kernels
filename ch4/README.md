# Chapter 4 — Compute Architecture and Scheduling

## Exercises

### 1. Warp analysis of foo_kernel

Launch config: N = 1024, 128 threads per block, so (1024+127)/128 = 8 blocks.

a. **Warps per block**: 128/32 = 4

b. **Warps in the grid**: 4 x 8 = 32

c. Line 04: `if (threadIdx.x < 40 || threadIdx.x >= 104)`

Per block:
- warp 0 (threads 0-31): all true, active
- warp 1 (32-63): threads 32-39 true, rest false → active and divergent
- warp 2 (64-95): all false → inactive
- warp 3 (96-127): threads 104-127 true → active and divergent

i. Active warps in the grid: 3 per block, so 3x8 = 24. A warp is active as soon as at least one of its threads executes the statement, so divergent warps count too.

ii. Divergent warps: 2 per block, 2x8 = 16.

iii. SIMD efficiency of warp 0, block 0: 32/32 = 100%

iv. Warp 1: 8 active threads (32-39), 8/32 = 25%

v. Warp 3: 24 active threads (104-127), 24/32 = 75%

SIMD efficiency = active threads / 32. An active warp consumes all 32 lanes worth of execution resources, masked threads included.

d. Line 07: `if (i % 2 == 0)`

i. All 32 warps are active (every warp has even indices).
ii. All 32 warps are divergent (every warp mixes even and odd).
iii. Warp 0 of block 0: 16/32 = 50%

e. Line 09: `for (j = 0; j < 5 - (i % 3); ++j)`

Trip count is 5 - (i%3), so 3, 4 or 5 iterations depending on the thread. Every warp contains the three residues of i%3.

i. Iterations j = 0, 1, 2 are executed by all threads → 3 iterations without divergence.
ii. Iterations j = 3 and j = 4 have part of the warp masked out → 2 divergent iterations.

### 2. Grid size for vector addition

Vector length 2000, blocks of 512: we need ceil(2000/512) = 4 blocks, so 4x512 = 2048 threads in the grid.

### 3. Divergent warps from the boundary check

The boundary i < 2000 falls inside warp 62 (threads 1984-2015), since 2000 is not a multiple of 32. That warp mixes valid and invalid indices: 1 divergent warp.
Note: warp 63 (2016-2047) fails the check entirely, so it is inactive, not divergent.

### 4. Time wasted at a barrier

Times in µs: 2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, 2.9. The barrier releases when the slowest thread (3.0) arrives.

Total time consumed: 8 x 3.0 = 24 µs. Total work: sum = 19.9 µs. Waiting: 4.1 µs, i.e. 4.1/24 ≈ 17%.

The cost of a barrier is set by the slowest thread — same reason divergence and load imbalance hurt at warp scale.

### 5. Skipping __syncthreads() with 32-thread blocks

Not a good idea. The intuition relies on the SIMD lock-step of the warp (implicit synchronization). But with control divergence, and since Volta with independent thread scheduling (per-thread program counters, interleaved passes), thread timing within a warp is not guaranteed. Keep an explicit sync: __syncthreads(), or __syncwarp() for warp scope.

### 6. Best block size (SM limits: 1536 threads, 4 blocks)

- 128: limited to 4 blocks → 512 threads
- 256: 4 blocks → 1024 threads
- 512: 3 blocks fit → 1536 threads, full SM
- 1024: 1 block → 1024 threads

Answer: c, 512 threads per block.

### 7. Feasible SM assignments (64 blocks, 2048 threads per SM)

All five configs are feasible.
- a. 8x128 = 1024 threads → 50%
- b. 16x64 = 1024 → 50%
- c. 32x32 = 1024 → 50%
- d. 64x32 = 2048 → 100%
- e. 32x64 = 2048 → 100%

### 8. Occupancy limiting factors (2048 threads, 32 blocks, 64K registers per SM)

a. 128 threads/block, 30 reg/thread: 16 blocks give 2048 threads, and 2048x30 = 61440 < 65536 registers. Full occupancy.

b. 32 threads/block, 29 reg/thread: capped at 32 blocks → 1024 threads, 50% occupancy. Limiting factor: blocks per SM.

c. 256 threads/block, 34 reg/thread: 2048x34 = 69632 > 65536. Only 7 blocks fit (7x256x34 = 60928 registers), i.e. 1792/2048 = 87.5%. Limiting factor: registers.

### 9. 32x32 blocks on a device limited to 512 threads/block

32x32 = 1024 threads per block, above the 512 limit: the launch would simply fail. The student's claim is impossible on that device.