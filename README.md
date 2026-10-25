# pmpp-cuda-kernels

**Hand-written CUDA kernels from *Programming Massively Parallel Processors*, tested, measured and compared with NVIDIA's libraries on an RTX 5060 Laptop GPU.**

> **Summary.** *Programming Massively Parallel Processors* (PMPP, 4th ed.) teaches GPU programming through a sequence of kernels and optimization techniques. This repository implements those kernels from scratch and asks a question the book only partly answers: *how much does each technique actually gain on a modern GPU, and why?* Every kernel is first checked for correctness, then timed with a statistical benchmark, compared with the hardware limits and with NVIDIA's own library (cuBLAS), and the results are explained with the hardware model and the Nsight Compute profiler.
> The two memory-bound kernels reach **87-90 % of the theoretical DRAM bandwidth**. For matrix multiplication, the techniques of chapters 5-6 bring a **1.9× speedup** over the naive kernel but stay at **15 % of cuBLAS**, in line with an independent published study. The profiler shows why: these techniques reduce traffic to off-chip memory, whereas the bottleneck has moved to the on-chip load/store pipeline that feeds shared memory. Acting on this diagnosis with register tiling, each thread keeping reused values in registers, brings the hand-written kernel to **55 % of cuBLAS (3.8× faster than the best chapter 6 kernel)**.

## Contents

1. [Motivation and scope](#1-motivation-and-scope)
2. [Background: the PMPP model of a GPU](#2-background-the-pmpp-model-of-a-gpu)
3. [Methodology: correctness, timing, clock](#3-methodology-correctness-timing-clock)
4. [Memory-bound kernels: how close to the bandwidth limit?](#4-memory-bound-kernels-how-close-to-the-bandwidth-limit)
5. [Matrix multiplication: testing the PMPP techniques against cuBLAS](#5-matrix-multiplication-testing-the-pmpp-techniques-against-cublas)
6. [Conclusion and next steps](#6-conclusion-and-next-steps)
7. [Reproducing the results](#7-reproducing-the-results)
8. [Repository layout](#8-repository-layout)
9. [Chapter notes and exercises](#9-chapter-notes-and-exercises)
10. [References](#10-references)

---

## 1. Motivation and scope

A GPU kernel can be correct and still use a small fraction of the hardware. Knowing *how far* a kernel is from what the GPU can do, and *what* stops it, is the core of GPU performance work. PMPP introduces the techniques (shared-memory tiling, thread coarsening, ...) and the reasoning tools (memory hierarchy, occupancy, roofline), but a textbook cannot say how they behave on the GPU in front of you.

This repository therefore goes one step beyond implementing the book's kernels. For each kernel, it answers three questions:

1. **Is it correct?** Tested against a CPU reference on shapes chosen to expose indexing bugs, and checked with `compute-sanitizer`.
2. **How fast is it, and how reliable is that number?** Measured with a statistical benchmark on a laptop GPU whose clock depends on its power mode.
3. **How good is it, and why?** Compared with the hardware limits, with NVIDIA's library, and with published results; explained with the hardware model and profiler counters.

**Scope.** Chapters 2 to 6 of PMPP: element-wise kernels (vector addition, image conversion) and matrix multiplication. Reductions (chapter 10) and further profiling are the next steps ([section 6](#6-conclusion-and-next-steps)). All kernels are written by hand; the cuBLAS folder is the vendor reference used for comparison.

The next section introduces the few hardware notions needed to read the results.

## 2. Background: the PMPP model of a GPU

This section summarizes, with the vocabulary of PMPP, the notions needed to read the results. Readers familiar with the book can skip to [section 3](#3-methodology-correctness-timing-clock).

**Threads, blocks and warps (PMPP ch. 2-4).** A kernel launch creates a *grid* of *thread blocks*. Blocks are assigned to the *streaming multiprocessors* (SMs) block by block (*block scheduling*); the RTX 5060 Laptop has 26 SMs. Inside an SM, each block is partitioned into *warps* of 32 threads with consecutive `threadIdx` values (a 2D block is first linearized in row-major order, `threadIdx.x` varying fastest). The warp is the unit of scheduling and executes in SIMD fashion: one instruction for its 32 threads. When threads of a warp take different paths, the warp suffers *control divergence* and executes each path in a separate pass.

**Latency tolerance and occupancy (ch. 4.6-4.7).** An SM holds more warps than it can execute at once. When a warp waits for a long-latency operation, such as a global memory access, the scheduler selects another ready warp at no cost (*zero-overhead scheduling*). The ratio of resident warps to the maximum supported is the *occupancy*; here the maximum is 48 warps (1,536 threads) per SM. Occupancy is limited by the resources that blocks share dynamically: thread slots, block slots (24 per SM), registers (64 K per SM) and shared memory.

**Memory types (ch. 5.2).** *Registers* are on-chip and private to a thread; their values are used directly as instruction operands. *Shared memory* is on-chip and shared by the threads of a block, but reading it requires a load instruction, so it has "longer latency and lower bandwidth than registers". *Global memory* is the off-chip DRAM (8 GB, 384 GB/s), behind a 32 MiB L2 cache and per-SM L1 caches. Most optimizations move data reuse to a faster level.

**Memory coalescing (ch. 6.1).** DRAM delivers data in *bursts* of consecutive locations. When the 32 threads of a warp execute a load instruction on consecutive addresses, the hardware *coalesces* their accesses into a single request that uses the whole burst. Scattered addresses require more requests and waste part of each burst.

**Compute to global memory access ratio and roofline (ch. 5.1).** PMPP measures how much work a kernel does per byte with the *compute to global memory access ratio*, also called *arithmetic intensity*: floating-point operations per byte accessed in global memory (FLOP/B). Compared with the ratio of the GPU's peaks, it tells which limit applies:

| | This GPU (RTX 5060 Laptop, Blackwell, cc 12.0) |
|---|---|
| FP32 compute peak | 128 FP32 lanes × 26 SMs × 2 FLOP (one fused multiply-add) × 2.75 GHz ≈ **18.3 TFLOP/s** |
| DRAM bandwidth peak | 128-bit bus × 24 Gbps (GDDR7) = **384 GB/s** |
| Ridge point of the roofline | 18.3 TFLOP/s / 384 GB/s ≈ **48 FLOP/B** |

A kernel well below 48 FLOP/B is **memory-bound**: its throughput is at most the bandwidth times its ratio. A kernel well above is **compute-bound**. This is the *roofline model*. The 2.75 GHz clock is measured, not taken from the specification sheet ([section 3](#3-methodology-correctness-timing-clock)).

The two kernel families studied here sit on opposite sides of the ridge point: element-wise kernels ([section 4](#4-memory-bound-kernels-how-close-to-the-bandwidth-limit)) are memory-bound, matrix multiplication ([section 5](#5-matrix-multiplication-testing-the-pmpp-techniques-against-cublas)) is compute-bound in principle. Before looking at them, the next section describes how correctness and speed are established.

## 3. Methodology: correctness, timing, clock

A performance number is only meaningful if the kernel is correct and the measurement is reproducible. This section describes the protocol applied to every kernel; the results in sections 4 and 5 all follow it.

### 3.1 Correctness before speed

Each kernel is compared with a CPU reference before it is ever timed. For matrix multiplication, which has the most room for indexing errors, the test ([`ch3/matmul/test_matmul.cu`](ch3/matmul/test_matmul.cu), shared by all variants including cuBLAS) uses:

- **9 matrix shapes** including degenerate (1×1×1), thin (one row or one column), rectangular, and sizes that are not multiples of the tile size, where boundary bugs appear. PMPP (section 5.5) stops at square matrices and leaves the rectangular case as an exercise; an earlier version of this repository tested only square matrices and missed a boundary check that compared a row index of B with the height of C, an error invisible when all dimensions are equal.
- **A double-precision CPU reference**, so that any discrepancy comes from the GPU result.
- **A tolerance derived from floating-point error analysis** rather than an arbitrary threshold: each FP32 rounding introduces a relative error of at most u = 2⁻²⁴, and a dot product of K non-negative terms accumulates at most about K·u (Higham, ch. 3). The test uses K·ε = 2K·u.
- **An output pre-filled with NaN**, so that an element the kernel forgets to write fails the test.

Kernels using shared memory are also run under `compute-sanitizer` *memcheck* (out-of-bounds accesses) and *racecheck* (missing synchronization), with zero errors.

### 3.2 Timing

The benchmark harness ([`utils/bench.h`](utils/bench.h)) measures **kernel time only**, with CUDA events recorded around each launch; host-device transfers are measured separately and excluded. It first runs the kernel for at least 300 ms, so that the GPU leaves its idle low-power state, then times 30 launches and reports the **median**, which is robust to occasional disturbances. Minimum, maximum and standard deviation are stored with every result.

Throughput is reported as **GFLOP/s** (10⁹ floating-point operations per second) for compute-bound kernels, counting 2·M·N·K operations for a matrix multiplication as cuBLAS does, and as **effective bandwidth** in GB/s (bytes read and written / time) for memory-bound kernels.

### 3.3 Clock and power mode on a laptop

A laptop GPU changes its clock with its power budget and temperature, so the clock must be measured, not assumed. A dedicated kernel ([`utils/clock_probe.cu`](utils/clock_probe.cu)) reads the SM cycle counter (`clock64()`) and the GPU nanosecond timer (`%globaltimer`) under a full FP32 load; cycles divided by nanoseconds give the actual clock. It is run before and after each benchmark sweep, and the clock during the cuBLAS kernel was also measured with Nsight Compute.

This turned out to matter a lot. Under the same FP32 load, the measured clock is **1.56 GHz in the laptop's "Quiet" mode and 2.75 GHz in "Balanced" or "Performance" mode**, a factor 1.76. The fastest matrix-multiplication kernels and cuBLAS also ran 8-11 % faster in "Performance" mode than in an earlier sweep made before switching to it, while the slower kernels did not change: the fastest ones run at the 80 W power limit, where the power mode decides the clock. All results below were obtained **in "Performance" mode, on AC power**.

| | |
|---|---|
| **GPU** | NVIDIA GeForce RTX 5060 Laptop GPU (Blackwell, cc 12.0, 26 SMs, 8 GB GDDR7, 32 MiB L2, 80 W) |
| **Software** | Windows + WSL2 (Ubuntu), CUDA 13.3, driver 592.15, Nsight Compute 2026.2, Nsight Systems 2026.1 |

With a protocol that gives correct kernels and reproducible numbers (two complete sweeps agree within 1 % for most configurations, and within 4 % for the noisiest ones, the 16×16-tile kernels at n = 4096), the next sections can compare the kernels with the hardware limits.

## 4. Memory-bound kernels: how close to the bandwidth limit?

The first kernels of PMPP perform very few operations per element. Their compute to global memory access ratio is far below the ridge point of [section 2](#2-background-the-pmpp-model-of-a-gpu), so their speed is limited by DRAM bandwidth, and the right question is: **what fraction of the 384 GB/s do they achieve?**

### 4.1 Kernels and their memory access patterns

For a memory-bound kernel, what matters is not only how many bytes it moves, but whether the threads of each warp access them in a coalesced way.

- **Vector addition** (`vecadd`, PMPP ch. 2): C[i] = A[i] + B[i], with 1D blocks of 256 threads and `i = blockIdx.x*blockDim.x + threadIdx.x`. One addition per 12 bytes moved (two 4-byte reads, one write): **0.08 FLOP/B**. Consecutive threads access consecutive elements, so each warp-wide load or store covers 32 × 4 = 128 contiguous bytes: all accesses are **fully coalesced**, and every byte of each DRAM burst is used.
- **RGB to grayscale** (`color2gray`, PMPP ch. 3): each output pixel is 0.21·r + 0.72·g + 0.07·b, with 2D blocks of 32×32 threads. About 5 operations per 16 bytes moved (three reads, one write): **0.3 FLOP/B**. Each warp is one row of the block, i.e. 32 consecutive pixels, and the output write `Pout[p]` is fully coalesced. The input, however, is stored *interleaved*, (r g b)(r g b)…, and each thread reads `Pin[3p]`, `Pin[3p+1]`, `Pin[3p+2]`. For one load instruction (say the `r` component), the 32 threads of a warp access addresses 12 bytes apart: the warp spans 384 bytes to use 128 of them, so this access is **only partially coalesced**. The `g` and `b` loads touch the same bytes again. On recent GPUs these repeated touches are largely served by the L1 cache (PMPP notes that caches "automatically coalesce more of the kernel access patterns"), so DRAM traffic stays close to the minimum, but each pixel costs three times more memory requests than with contiguous accesses.

Both kernels are 150 to 600 times below the ridge point: even a perfect implementation is limited by bandwidth. Input sizes are chosen far larger than the 32 MiB L2 cache, otherwise the measurement would reflect the cache rather than DRAM.

### 4.2 Results

| Kernel | Problem size | Data moved | Median time | Effective bandwidth | % of 384 GB/s |
|---|---|---|---|---|---|
| `vecadd` | 2²⁶ floats per array | 768 MiB | 2.317 ms | **347.6 GB/s** | **90.5 %** |
| `color2gray` | 4096 × 4096 pixels | 256 MiB | 0.806 ms | **333.2 GB/s** | **86.8 %** |

### 4.3 Discussion

Reaching 85-95 % of theoretical DRAM bandwidth is what a well-written streaming kernel is expected to achieve: the theoretical figure is never fully attainable in practice (DRAM refresh, read/write turnarounds). `vecadd`, whose accesses are all fully coalesced, reaches 90.5 %.

`color2gray` is 4 % slower in bandwidth terms. This is consistent with the partially coalesced reads of its interleaved RGB input described in section 4.1: the cache keeps the extra DRAM traffic small, but the extra memory requests are not free. This explanation has not been confirmed with the profiler yet. If confirmed, PMPP's checklist (table 6.1) gives two remedies: rearrange the data layout (three separate R, G and B arrays, so that each load is fully coalesced), or load the pixels into shared memory with coalesced accesses and perform the strided reads there (shared memory "does not require coalescing").

An earlier version of `color2gray` used the constants `0.21`, `0.72`, `0.07`, which C++ treats as **double-precision** values: the computation silently ran in FP64, which consumer GPUs execute at a small fraction of their FP32 rate. Writing `0.21f` removed the FP64 instructions and reduced register usage from 15 to 12. The kernel being memory-bound, the effect on its speed is small, but the same mistake in a compute-bound kernel would be costly.

**Conclusion of section 4.** For memory-bound kernels, the PMPP implementations are already close to the hardware limit when their accesses are coalesced, and little remains to gain. Matrix multiplication is the opposite case: its compute to global memory access ratio allows it, in principle, to be limited by compute, and the techniques of the book aim precisely at reaching that regime. It is studied next.

## 5. Matrix multiplication: testing the PMPP techniques against cuBLAS

Matrix multiplication (*SGEMM* in BLAS terminology: Single-precision GEneral Matrix Multiply) computes C = A·B, where each element of C is a dot product of a row of A and a column of B. It is the reference kernel of GPU computing: it dominates the cost of deep learning and many scientific codes, and NVIDIA's cuBLAS library provides a highly tuned implementation.

For n × n matrices, SGEMM performs 2n³ operations on 3n² values. At n = 4096 this is 137 GFLOP for 201 MB of data, so the compute to global memory access ratio can reach 680 FLOP/B if every value is loaded only once, far above the ridge point of 48. **SGEMM can therefore be compute-bound, but only if each value loaded from DRAM is reused many times.** PMPP chapters 3, 5 and 6 introduce successive techniques to achieve this reuse. This section measures what each technique gains, compares the result with cuBLAS, and explains the remaining gap.

### 5.1 Kernels studied

| Kernel | PMPP | Idea | Expected effect |
|---|---|---|---|
| **Naive** | 3.4 | One thread computes one element of C as a dot product of a row of A and a column of B, read directly from global memory. Accesses to B (`B[k*N + col]`) are coalesced; each element of A is broadcast to the whole warp. | Compute to global memory access ratio of **0.25 FLOP/B** (two 4-byte loads per multiply-add, section 5.1): memory-bound in the book's model. |
| **Tiled** (16×16 or 32×32) | 5.3-5.5 | The dot product is split into *phases* (*strip-mining*). In each phase, the threads of a block collaboratively load one tile of A and one tile of B into shared memory, then compute from these copies. Two barriers protect the read-after-write and write-after-read dependences. | Global memory traffic divided by the tile width: **4 FLOP/B** (16×16) or **8 FLOP/B** (32×32). |
| **Tiled + thread coarsening** (factor 1, 2, 4) | 6.3 | Each thread computes several output elements in adjacent tiles, so a block loads each tile of A once for several tiles of C. | Reduces the *price of parallelism*: the redundant loading of the same A tile by different blocks. |
| **cuBLAS** | 5.4 | One of the "highly optimized libraries" the book points to, used as the reference. | Best achievable performance in practice. |

Tile size and coarsening factor are compile-time parameters, so a single script ([`scripts/sweep_matmul.sh`](scripts/sweep_matmul.sh)) builds, tests and benchmarks all 10 configurations at 4 matrix sizes. Each configuration must pass the correctness test of [section 3.1](#31-correctness-before-speed) before being timed.

### 5.2 Results

![SGEMM throughput vs matrix size](results/matmul_gflops.png)

| Kernel | n = 512 | n = 1024 | n = 2048 | n = 4096 |
|---|---|---|---|---|
| Naive (ch. 3) | 1,018 GFLOP/s (17 %) | 1,066 (12 %) | 1,084 (11 %) | 770 (8 %) |
| Tiled 16×16 (ch. 5) | 1,369 (22 %) | 1,462 (17 %) | 1,482 (15 %) | 1,445 (14 %) |
| Tiled 32×32 (ch. 5) | 1,353 (22 %) | 1,407 (16 %) | 1,428 (15 %) | 1,265 (12 %) |
| **Tiled 16×16 + coarsening ×4 (ch. 6)** | 1,420 (23 %) | 1,533 (18 %) | **1,580 (16 %)** | **1,475 (15 %)** |
| *Register tiling 1D, 64 / ×8 (section 5.7)* | *3,370 (55 %)* | *4,486 (52 %)* | *4,640 (47 %)* | *4,426 (44 %)* |
| *Register tiling 2D, 64 / 4×4 (section 5.7)* | *4,161 (68 %)* | *5,717 (66 %)* | *6,002 (61 %)* | *5,598 (55 %)* |
| cuBLAS | 6,157 | 8,625 | 9,806 | 10,122 |

Median of 30 runs; percentages are relative to cuBLAS at the same size. All 72 measurements (18 configurations × 4 sizes): [`results/sweep_matmul.csv`](results/sweep_matmul.csv); clock and environment of the run: [`results/env_matmul.txt`](results/env_matmul.txt).

The best chapter 5-6 kernel is **1.9× faster than the naive one** at n = 4096, but reaches **15 % of cuBLAS**. In absolute terms, cuBLAS reaches about 55 % of the FP32 peak of section 2 and the best chapter 5-6 kernel about 8 %. Sections 5.3 to 5.6 explain these two observations, the modest gain of the PMPP techniques and the large gap to cuBLAS; section 5.7 then tests the resulting diagnosis with the two register tiling kernels shown in italics.

### 5.3 Effect of each technique

![Tile size x coarsening factor at n = 2048](results/matmul_configs.png)

**Tiling helps, but much less than the reduction in memory traffic would suggest.** A 16×16 tile divides global-memory traffic by 16, yet the speedup over the naive kernel is only 1.37× at n = 2048. The naive kernel was therefore not limited by DRAM, contrary to what the book's model predicts: at 0.25 FLOP/B and 384 GB/s it would be capped at 96 GFLOP/s, yet it reaches 1,084 GFLOP/s, eleven times more. Its redundant reads are served by the caches, which PMPP's model does not include (Nsight Compute measures a **95 % L1 hit rate**, [section 5.4](#54-what-the-profiler-shows)). The exception confirms the rule: at n = 4096, a single matrix (64 MiB) no longer fits in the 32 MiB L2 cache, and the naive kernel loses 29 % of its throughput, while the tiled kernels lose 3-11 % and cuBLAS nothing.

**A larger tile is slower.** A 32×32 tile halves global traffic again compared with 16×16, but is 4 % slower at n = 2048 and 12 % slower at n = 4096. Both tile sizes perform exactly the same amount of shared-memory work, so the difference must come from how well that work is overlapped. A 32×32 tile requires blocks of 1,024 threads, and only one such block fits on an SM (limit: 1,536 threads), so only 32 of the 48 warp slots are used (67 % occupancy, confirmed by the profiler). With 16×16 tiles, six independent blocks of 256 threads share the SM (100 % occupancy), and the profiler shows that they keep the load/store pipeline busy 97 % of the time, against 76 % for the 32×32 tile. With fewer resident warps, the 32×32 tile has less *latency tolerance* in PMPP's sense (section 4.6): fewer ready warps to schedule while others wait. This again shows that global traffic is not the limiting factor. *(Before profiling, the expected cause was that the 32 warps of the single block stall together at the `__syncthreads()` barriers; the profiler does not support it: barrier stalls are lower with the 32×32 tile. The exact mechanism of the lower pipeline utilization remains open, see section 6.)*

**Thread coarsening gives a small gain:** +5 % with factor 2 and +7 % with factor 4 at n = 2048, for 16×16 tiles. Sections [5.4](#54-what-the-profiler-shows) and [5.5](#55-why-coarsening-gains-little-comparison-with-a-published-study) explain why the gain is small. As a sanity check, coarsening with factor 1 reproduces the plain tiled kernel (1,474 vs 1,482 GFLOP/s), so the coarsened code adds no overhead and the measurements are consistent.

**Small problems expose wave quantization.** Blocks are distributed to SMs in "waves". With 32×32 tiles and coarsening ×4 at n = 512, the grid has 64 blocks for 26 SMs holding one block each: 3 waves, the last one using 12 of the 26 SMs. The expected efficiency is 64/78 = 82 %; the measured throughput is 85 % of the same kernel with coarsening ×2, whose 128 blocks fill 5 waves almost completely (128 of 130 slots).

In summary, the measurements show that DRAM traffic was not the bottleneck of these kernels, which explains why techniques designed to reduce it bring limited gains. The next subsection uses the profiler to identify what the bottleneck is.

### 5.4 What the profiler shows

Timings show *that* a kernel is slow; hardware counters show *where* the time goes. Each kernel was profiled with Nsight Compute at n = 4096 (one launch after warmup, real clock, full metric set; script [`scripts/profile_matmul.sh`](scripts/profile_matmul.sh), summary [`results/ncu_matmul.md`](results/ncu_matmul.md)). The table keeps the metrics needed for the argument; how to read them follows.

| Metric (n = 4096) | Naive | Tiled 16 | Tiled 32 | Tiled 16 + coarsening ×4 | cuBLAS |
|---|---|---|---|---|---|
| Kernel duration (ms) | 189.7 | 94.7 | 108.9 | 91.3 | 12.2 |
| **FMA pipe active** (%) | 8.7 | 10.2 | 7.9 | 9.6 | **69.3** |
| **Load/store (LSU) pipe active** (%) | 63.8 | **96.5** | 76.3 | **95.1** | 58.7 |
| **Shared-memory load wavefronts** (millions) | 0 | 3,221 | 3,221 | 3,221 | **269** |
| Shared-memory bank conflicts (millions) | 0 | 0.1 | 0.0 | 0.0 | 0.1 |
| L1 hit rate (%) | 95.0 | 0.0 | 0.0 | 0.2 | 0.0 |
| Achieved occupancy (%) | 66.7 | 99.3 | 66.6 | 98.6 | 24.8 |
| Instructions executed (millions) | 10,489 | 7,807 | 6,330 | 6,384 | 2,614 |
| Main stall reasons | global-memory queue full, waiting for global loads | shared-memory queue full, waiting for global loads | shared-memory queue full | waiting for global loads, shared-memory queue full | almost none |

*How to read it.* A GPU instruction is executed by a *pipe*: the FMA pipe does the multiply-adds, the LSU (load/store unit) pipe issues every memory access, including shared-memory loads. A *wavefront* is one pass of the shared-memory hardware; it counts the real shared-memory work. *Stall reasons* tell what warps were waiting for when they could not issue an instruction ("shared-memory queue full" is the profiler's *MIO throttle*, "waiting for global loads" its *long scoreboard*).

Four observations follow:

1. **The tiled kernels are limited by the load/store pipe, not by arithmetic or DRAM.** Their LSU pipe is busy 95-97 % of the time while the FMA pipe is busy only 10 %: the GPU spends its time moving operands, not computing. The two main stall reasons (shared-memory queue full, waiting for global loads) say the same thing. This is the situation described in PMPP section 5.2: an operand in shared memory, unlike one in a register, requires a separate load instruction, and "the processor can fetch and execute only a limited number of instructions per clock cycle". With two shared-memory loads per multiply-add, load instructions, not arithmetic, set the pace.
2. **Neither the tile size nor coarsening changes the shared-memory work.** The three tiled kernels execute *exactly* the same number of shared-memory wavefronts (3,221 million), because all of them read two values from shared memory per multiply-add. Coarsening only removes other instructions (−18 %: fewer global loads of A tiles and fewer index computations), hence its small gain. This is the direct measurement behind the analysis of section 5.5.
3. **Bank conflicts are not the problem**: they are negligible in every kernel.
4. **The naive kernel is served by the L1 cache** (95 % hit rate): its redundant global reads rarely reach DRAM, which is why tiling, designed to reduce DRAM traffic, gains only 1.4-1.9×.

The profiler thus confirms the explanation suggested by the timings, and quantifies it. The next subsection compares these kernels with a published study to show what a better kernel changes.

### 5.5 Why coarsening gains little: comparison with a published study

Simon Boehm's [worklog on optimizing SGEMM](https://siboehm.com/articles/22/CUDA-MMM) applies a similar sequence of techniques on another GPU (RTX A6000, Ampere) and reports each step as a percentage of cuBLAS. This makes it possible to check our results against an independent measurement and to locate where they diverge.

| Step | Boehm (A6000, n = 4092) | This repository (n = 4096) |
|---|---|---|
| Naive with coalesced accesses | 8.5 % of cuBLAS | 7.6 % |
| 32×32 shared-memory tiling | 12.8 % | 12.5 % |
| Several outputs per thread | 36.5 % (8 outputs per thread), **×2.85** | 14.6 % (coarsening ×4), **×1.02** over 16×16 tiling |

The first two steps agree within about one point, on a different architecture: this supports the validity of the measurements. The third step differs sharply, although in both cases each thread computes several outputs. The difference lies in **which memory level the extra outputs save accesses to**. Counting memory accesses per element of C, as Boehm does:

| Kernel | Global-memory loads per output | Shared-memory loads per output |
|---|---|---|
| Boehm, 32×32 tiles | K/16 | 2K |
| Boehm, 8 outputs per thread ("1D blocktiling") | K/32 | **9K/8** |
| This repository, 16×16 tiles, coarsening ×4 | ≈ K/13 | **2K** (unchanged) |

PMPP coarsening reuses the tile of A for several outputs, which reduces **global**-memory traffic. But in the inner loop each multiply-add still reads two values from shared memory, because the loop over outputs is the outermost one and each output requires a new tile of B and a synchronization barrier:

```cpp
for (int c = 0; c < COARSE_FACTOR; c++) {      // outer loop: one B tile per output
    /* load the B tile for output c */ __syncthreads();
    for (int k = 0; k < TILE_WIDTH; k++)
        Pval[c] += As[ty][k] * Bs[k][tx];       // 2 shared-memory loads per multiply-add
    __syncthreads();
}
```

Boehm's kernel puts the loop over K outside and the loop over outputs inside: each value read from shared memory is kept in a **register** and reused for 8 multiply-adds, which almost halves shared-memory traffic per output. Boehm's profiler shows that his tiled kernel spends most of its time waiting on the shared-memory pipeline (*MIO throttle* stalls); the profile of section 5.4 shows the same on this GPU, and shows that coarsening leaves the shared-memory work unchanged.

**This is the central result of section 5.** PMPP coarsening does what the book says: it reduces the price of parallelism, the redundant loading of A tiles (18 % fewer instructions). But that price was not the bottleneck. PMPP section 6.5 warns that "if the optimization that is applied does not target the bottleneck resource, there may be no benefit": the chapter 5-6 techniques reduce traffic to global memory, whereas once the data is in caches and shared memory, the limit becomes the rate at which shared memory can feed the arithmetic units. Closing the gap requires keeping more data in registers, a technique called *register tiling*; PMPP introduces it for stencils (section 8.5), where "many reads from and writes to the shared memory are now shifted into registers". The next subsection shows that this is exactly what cuBLAS does, and section 5.7 tests the idea on this repository's own kernels.

### 5.6 What cuBLAS does differently

To know which algorithm cuBLAS runs, Nsight Compute was used to name the kernel it launches at n = 4096: `cutlass_80_simt_sgemm_128x64_8x5_nn_align1`. The name describes its design:

- `simt`: it runs on the standard FP32 units, **not on Tensor Cores** (which would use reduced-precision TF32 arithmetic). The comparison with the hand-written FP32 kernels is therefore fair.
- `128x64`: each block of 128 threads computes a 128×64 tile of C, i.e. **64 outputs per thread kept in registers** (1 to 4 in the hand-written kernels). This is the register tiling identified in section 5.5.
- `8x5`: tiles of A and B are processed 8 columns at a time through a **5-stage pipeline**, so that the next tiles are loaded while the current one is computed.

The profile of section 5.4 shows the effect of these choices. For the same 137 GFLOP, cuBLAS performs **12× fewer shared-memory wavefronts** (269 million against 3,221 million) and 2.4× fewer instructions than the best chapter 6 kernel. Its FMA pipe is busy **69 %** of the time (10 % for the chapter 5-6 kernels), at a clock of 2.6-2.7 GHz measured during the kernel. It also qualifies a rule of PMPP chapter 4, "the higher the occupancy, the better it can hide long-latency operations": cuBLAS runs at only **25 % occupancy**, yet its warps almost never stall, because each warp has many independent multiply-adds to issue (instruction-level parallelism) instead of relying on other warps for latency tolerance.

### 5.7 Testing the diagnosis: register tiling (beyond PMPP)

Sections 5.4 to 5.6 lead to a precise prediction: if the tiled kernels are slow because each multiply-add needs two load instructions from shared memory, then **keeping reused values in registers** should remove the bottleneck, without changing anything else. Two kernels were written to test it, starting from the tiled kernel of chapter 5 and keeping its structure (square tiles `As` and `Bs` of `TILE_WIDTH × TILE_WIDTH`, phases along K, two barriers, boundary checks). They are in [`extra/`](extra) because they go beyond the techniques of PMPP chapters 5-6, and pass the same correctness test and `compute-sanitizer` checks as the other kernels.

| Kernel | Configuration | Outputs per thread | Threads per block |
|---|---|---|---|
| **Register tiling 1D** ([`extra/register_tiling1D`](extra/register_tiling1D)) | `TILE_WIDTH` = 64, `COARSE_FACTOR` = 8 | 8 consecutive rows of one column | 64 × 8 = 512 |
| **Register tiling 2D** ([`extra/register_tiling2D`](extra/register_tiling2D)) | `TILE_WIDTH` = 64, `COARSE_FACTOR` = 4 | a 4 × 4 square of outputs | 16 × 16 = 256 |

**Why it should help.** In the chapter 5 kernel, a thread computes one output and, for each `k`, reads `As[ty][k]` and `Bs[k][tx]` for a single multiply-add. When a thread computes several outputs of the same tile, several of them need the same value: in the 1D kernel, `Bs[k][tx]` is the same for the 8 outputs of a column; in the 2D kernel, each value of `As` serves 4 outputs and each value of `Bs` serves 4 others. Read once and kept in a register, a value feeds several multiply-adds, since a register operand costs no extra instruction (PMPP 5.2). Per multiply-add, the number of values read from shared memory drops from 2 (chapter 5) to 9/8 (1D) and 0.5 (2D).

**Results (n = 4096).**

| Kernel | Median time | GFLOP/s | % of cuBLAS | Speedup vs best chapter 6 kernel |
|---|---|---|---|---|
| Tiled 16×16 + coarsening ×4 (ch. 6) | 93.2 ms | 1,475 | 15 % | 1.0× |
| **Register tiling 1D** (64 / ×8) | 31.1 ms | **4,426** | **44 %** | **3.0×** |
| **Register tiling 2D** (64 / 4×4) | 24.6 ms | **5,598** | **55 %** | **3.8×** |
| cuBLAS | 13.6 ms | 10,122 | 100 % | 6.9× |

Targeting the bottleneck identified by the profiler gives a 3-4× speedup, where the chapter 5-6 techniques gave 1.4-1.9× over the naive kernel. At n = 2048 the 2D kernel reaches 6,002 GFLOP/s, 61 % of cuBLAS.

**What the profiler shows.** Same profile as section 5.4 ([`results/ncu_matmul.md`](results/ncu_matmul.md)):

| Metric (n = 4096) | Tiled 16 | Tiled 16 + coarsening ×4 | **Register tiling 1D** | **Register tiling 2D** | cuBLAS |
|---|---|---|---|---|---|
| Kernel duration (ms) | 94.7 | 91.3 | 29.5 | 22.5 | 12.2 |
| Shared-memory load wavefronts (millions) | 3,221 | 3,221 | **1,345** | **808** | 269 |
| Load/store (LSU) pipe active (%) | 96.5 | 95.1 | 97.7 | **61.1** | 58.7 |
| FMA pipe active (%) | 10.2 | 9.6 | 28.1 | **37.8** | 69.3 |
| Instructions executed (millions) | 7,807 | 6,384 | 3,599 | **3,021** | 2,614 |
| Achieved occupancy (%) | 99.3 | 98.6 | 66.4 | 49.7 | 24.8 |
| "Shared-memory queue full" stalls (cycles per instruction) | 16.7 | 11.0 | 9.0 | **3.6** | 0.4 |

1. **The shared-memory work drops as predicted.** Relative to the tiled kernel, shared-memory wavefronts fall to 0.42× in 1D (predicted 0.56×; reads that serve the whole warp make it even lower) and to **0.25× in 2D, exactly the prediction**.
2. **In 2D, the bottleneck is gone.** The load/store pipe is no longer saturated (61 % instead of 97 %), the arithmetic units are busy almost four times more often (38 % instead of 10 %), and the kernel executes 2.1× fewer instructions than the best chapter 6 kernel, close to cuBLAS. Occupancy is limited to 50 % by shared memory: each block uses 32 KiB (two 64 × 64 float tiles), so only 3 blocks of 256 threads fit in the 100 KiB of an SM (PMPP 5.6). As with cuBLAS, the kernel nevertheless runs faster with fewer resident warps, because each warp has more independent multiply-adds to issue.
3. **In 1D, the load/store pipe is still saturated** (97.7 %), but each of its instructions now feeds about 3 times more arithmetic (FMA pipe at 28 %): 9 values read per 8 multiply-adds is better than 2 per 1, but not yet enough to make arithmetic the limit. The 2D kernel, with 0.5 values per multiply-add, crosses that threshold.

**Choice of configuration.** The sweep also covers other tile sizes and coarsening factors ([`results/sweep_matmul.csv`](results/sweep_matmul.csv)). For both kernels, larger tiles and more outputs per thread help, up to a cliff: the 2D kernel with `COARSE_FACTOR` = 8 (64 outputs per thread) needs more than the 255 registers a thread can have, so the compiler *spills* registers to local memory (1,204 bytes per thread with 64 × 64 tiles, `ptxas -v` in [`results/build.log`](results/build.log)), and throughput collapses to about 900 GFLOP/s, six times slower than with `COARSE_FACTOR` = 4. This is the *register spilling* and *performance cliff* described in PMPP section 4.7.

**What the compiler does.** In the 2D kernel as written, the loop over `k` is the innermost one, so the source code reads each value of `As` and `Bs` several times. The compiler, however, fully unrolls this loop (its bounds are compile-time constants) and removes the repeated reads. The machine code (`cuobjdump -sass`) contains, per phase, **128 shared-memory load instructions, all 128-bit (`LDS.128`, 4 floats each)**: exactly the 512 values the thread needs, each read once, four at a time. A separate profile confirms it: the kernel executes **268,435,456** shared-memory load instructions, exactly the predicted 32,768 warps × 64 phases × 128. A version that loads the registers explicitly, with the loop over `k` outermost, reads the same values but was 5-8 % slower: the compiler only unrolled its loop by 2, so it read `As` two floats at a time (192 load instructions per phase instead of 128).

**What remains between this kernel and cuBLAS.** The FMA pipe is busy 38 % of the time against 69 % for cuBLAS. Likely causes, to be investigated: the tiles are loaded with partially coalesced accesses (neighbouring threads load addresses 4 floats apart, the pattern of section 4.1; the 60 % L1 hit rate of the 2D kernel, against 0 % for the tiled kernels, is consistent with lines being fetched once and touched by several load instructions), there are some shared-memory bank conflicts (2.3 million), and cuBLAS overlaps the loading of the next tiles with the computation (its 5-stage pipeline), while this kernel waits at a barrier in every phase.

**Conclusion of section 5.** The PMPP techniques were measured one by one: tiling and coarsening give a combined 1.9× speedup, consistent with an independent study, but leave the kernel at 15 % of cuBLAS. The profiler shows that the bottleneck has moved from global memory to the load/store pipeline feeding shared memory, that neither tile size nor coarsening changes the shared-memory work, and that cuBLAS removes this bottleneck with register tiling. Applying the same idea to the chapter 5 kernel confirms the diagnosis: register tiling divides the shared-memory work by 4 as predicted, relieves the load/store pipe (61 % instead of 97 %), quadruples the activity of the arithmetic units, and brings the hand-written kernel from 15 % to 55 % of cuBLAS.

## 6. Conclusion and next steps

This repository implemented the kernels of PMPP chapters 2-6 and measured them with a protocol designed for a laptop GPU (correctness tests, statistical timing, measured clock). Four conclusions emerge:

1. **Memory-bound kernels** are close to the hardware limit (87-90 % of DRAM bandwidth) with the book's straightforward implementations.
2. **For matrix multiplication**, tiling and coarsening bring a 1.9× speedup but stay at 15 % of cuBLAS, in agreement with an independent published study.
3. **The reason**, confirmed with Nsight Compute, is that these techniques reduce global memory traffic, whereas the bottleneck is the load/store pipeline feeding shared memory (busy 97 % of the time, against 10 % for the arithmetic units). This is PMPP's lesson of section 6.5 measured on real hardware: an optimization helps only if it targets the bottleneck resource, and finding that resource requires the profiler.
4. **Acting on the diagnosis works**: register tiling, which targets this bottleneck, divides the shared-memory work by 4, relieves the load/store pipe (61 %), quadruples the activity of the arithmetic units and brings the kernel from 15 % to 55 % of cuBLAS.

Some questions remain open, and some techniques of the book remain to be studied. Planned work, in order of priority:

- [x] **Profile the hand-written SGEMM kernels with Nsight Compute**: load/store-pipe bottleneck confirmed ([section 5.4](#54-what-the-profiler-shows)).
- [ ] **Explain the lower pipe utilization of the 32×32 tile** (76 % vs 97 %, same shared-memory work); the barrier-stall hypothesis was not supported.
- [ ] **Measure DRAM traffic per kernel** (the DRAM counters were not collected in this run) to quantify the L2 effect on the naive kernel at n = 4096.
- [ ] **Explain the 4 % gap of `color2gray`** with the profiler, and test a planar (structure-of-arrays) RGB layout.
- [ ] **Parallel reduction** (PMPP ch. 10): reduction tree with control divergence, convergent version (less control and memory divergence), shared memory, segmented multiblock reduction with atomics, thread coarsening, plus warp shuffles; compared with NVIDIA's CUB library.
- [ ] **Roofline plot** of all kernels from profiler counters.
- [x] **Register tiling for SGEMM** (1D and 2D): 44 % and 55 % of cuBLAS ([section 5.7](#57-testing-the-diagnosis-register-tiling-beyond-pmpp)).
- [ ] **Close more of the gap to cuBLAS**: coalesced tile loading in the 2D kernel (neighbouring threads loading neighbouring columns), removing bank conflicts, overlapping the loading of the next tiles with computation.

## 7. Reproducing the results

Requirements: an NVIDIA GPU, CUDA toolkit ≥ 12.8 (needed for Blackwell `sm_120`), GNU Make, and Python 3 with `pandas` and `matplotlib` for the figures. Kernels are built for the local GPU (`-arch=native`); use `make ARCH=sm_XX` to target another architecture.

```bash
git clone https://github.com/leedouardpro-code/pmpp-cuda-kernels.git
cd pmpp-cuda-kernels

make -C utils query probe              # GPU properties, then measured SM clock under load
make -C ch2/vecadd test bench          # one kernel: correctness tests, then benchmark
make -C ch5/matmul_tiling EXTRA="-DTILE_WIDTH=16" clean test bench   # another tile size

scripts/sweep_matmul.sh                # all SGEMM variants x 4 sizes -> results/sweep_matmul.csv
python3 scripts/plot_matmul.py         # tables and figures in results/
scripts/profile_matmul.sh              # Nsight Compute profiles (needs GPU counter access)
python3 scripts/summarize_ncu.py       # -> results/ncu_matmul.md
```

On a laptop, connect the charger and select the highest performance mode before benchmarking ([section 3.3](#33-clock-and-power-mode-on-a-laptop)). The sweep records the measured clock before and after the run in `results/env_matmul.txt`.

## 8. Repository layout

Each kernel has its own folder with the same structure: `xxx.cu` (kernel, launch function and host wrapper), `test_xxx.cu` (correctness test against the CPU), `bench_xxx.cu` (benchmark), and a `Makefile` with `test`, `bench` and `clean` targets.

<details>
<summary>Full tree</summary>

```
pmpp-cuda-kernels/
├── common.mk                   # shared build settings (compiler flags, EXTRA defines, header tracking)
├── utils/
│   ├── bench.h                 # statistical timing harness (section 3.2)
│   ├── clock_probe.cu          # measured SM clock and practical FP32 peak (section 3.3)
│   ├── cuda_check.h            # CUDA_CHECK error-handling macro
│   ├── device_query.{h,cu}     # device limits and theoretical peaks
│   ├── device_query_main.cu
│   └── Makefile                # `make query`, `make probe`
├── ch2/vecadd/                 # vector addition (section 4)
├── ch3/color2gray/             # RGB to grayscale (section 4)
├── ch3/matmul/                 # naive SGEMM; its test and benchmark files are shared by all SGEMM folders
├── ch4/README.md               # exercises: warps, divergence, occupancy
├── ch5/README.md               # exercises: tiling, roofline, occupancy
├── ch5/matmul_tiling/          # shared-memory tiled SGEMM (TILE_WIDTH set at compile time)
├── ch6/matmul_coarsening/      # tiling + thread coarsening (TILE_WIDTH, COARSE_FACTOR)
├── extra/
│   ├── register_tiling1D/      # tiled SGEMM, each thread computes COARSE_FACTOR outputs of a column (section 5.7)
│   └── register_tiling2D/      # tiled SGEMM, each thread computes a COARSE_FACTOR x COARSE_FACTOR block (section 5.7)
├── cublas/                     # cuBLAS SGEMM reference, same interface as the hand-written kernels
├── scripts/
│   ├── sweep_matmul.sh         # builds, tests and benchmarks every SGEMM configuration
│   ├── plot_matmul.py          # tables and figures from the sweep results
│   ├── profile_matmul.sh       # Nsight Compute profile of the SGEMM kernels (section 5.4)
│   └── summarize_ncu.py        # metric and stall tables from the profiles
└── results/                    # raw data behind the numbers in this README
```

</details>

## 9. Chapter notes and exercises

| Chapter | Topic | Code | Notes |
|---|---|---|---|
| 2 | Heterogeneous data parallel computing | [`vecadd`](ch2/vecadd) | |
| 3 | Multidimensional grids and data | [`color2gray`](ch3/color2gray), [`matmul`](ch3/matmul) | |
| 4 | Compute architecture and scheduling | | [exercises](ch4/README.md) |
| 5 | Memory architecture and data locality | [`matmul_tiling`](ch5/matmul_tiling) | [exercises](ch5/README.md) |
| 6 | Performance considerations | [`matmul_coarsening`](ch6/matmul_coarsening) | |

Code conventions: `_h` / `_d` suffixes for host / device pointers (as in the book), every CUDA API call wrapped in `CUDA_CHECK`, `cudaGetLastError()` after each kernel launch.

## 10. References

- W. Hwu, D. Kirk, I. El Hajj, *Programming Massively Parallel Processors: A Hands-on Approach*, 4th ed., Morgan Kaufmann, 2023.
- S. Boehm, [*How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog*](https://siboehm.com/articles/22/CUDA-MMM), 2022.
- N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002 (floating-point error bound used for the test tolerance).
- NVIDIA, [*CUDA C++ Programming Guide*](https://docs.nvidia.com/cuda/cuda-c-programming-guide/), [*CUDA C++ Best Practices Guide*](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/), [*Nsight Compute Kernel Profiling Guide*](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).
- [GPU MODE lecture series](https://github.com/gpu-mode/lectures).