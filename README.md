# pmpp-cuda-kernels

**Hand-written CUDA kernels from *Programming Massively Parallel Processors*, tested, measured and compared with NVIDIA's libraries on an RTX 5060 Laptop GPU.**

> **Summary.** *Programming Massively Parallel Processors* (PMPP, 4th ed.) teaches GPU programming through a sequence of kernels and optimization techniques. This repository implements those kernels from scratch and asks a question the book only partly answers: *how much does each technique actually gain on a modern GPU, and why?* Every kernel is first checked for correctness, then timed with a statistical benchmark, compared with the hardware limits and with NVIDIA's own library (cuBLAS), and the results are explained with the hardware model and the Nsight Compute profiler.
> The two memory-bound kernels reach **87-90 % of the theoretical DRAM bandwidth**. For matrix multiplication, the techniques of chapters 5-6 bring a **1.95× speedup** over the naive kernel but stay at **14 % of cuBLAS**, in line with an independent published study. The analysis shows why: these techniques reduce traffic to off-chip memory, whereas the bottleneck has moved to on-chip shared memory.

## Contents

1. [Motivation and scope](#1-motivation-and-scope)
2. [Background: what limits a GPU kernel](#2-background-what-limits-a-gpu-kernel)
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

## 2. Background: what limits a GPU kernel

This section gives the minimum vocabulary used in the rest of the document. Readers familiar with GPU architecture can skip to [section 3](#3-methodology-correctness-timing-clock).

**Execution model.** A GPU is made of *streaming multiprocessors* (SMs); the RTX 5060 Laptop has 26. A kernel launches a *grid* of *blocks*, each block being a group of threads that runs on one SM. Threads are executed in groups of 32 called *warps*. An SM keeps several warps resident at once (here at most 1,536 threads, i.e. 48 warps) and switches between them to hide the latency of memory accesses. The fraction of these 48 slots that a kernel actually fills is its *occupancy*.

**Memory hierarchy.** From slowest and largest to fastest and smallest: *DRAM* (global memory, 8 GB, 384 GB/s), the *L2 cache* (32 MiB, shared by all SMs), and inside each SM the *L1 cache*, part of which can be used as *shared memory* (up to 100 KiB per SM, managed explicitly by the programmer and shared by the threads of a block), and the *registers* (private to each thread, the fastest level). Most optimizations consist in moving data reuse to a faster level.

**Two limits: compute and bandwidth.** A kernel is limited either by how fast the GPU computes or by how fast it moves data. Which limit applies depends on the kernel's **arithmetic intensity**: the number of floating-point operations (FLOP) performed per byte transferred from DRAM.

| | This GPU (RTX 5060 Laptop, Blackwell, cc 12.0) |
|---|---|
| FP32 compute peak | 128 FP32 lanes × 26 SMs × 2 FLOP (one fused multiply-add) × 2.75 GHz ≈ **18.3 TFLOP/s** |
| DRAM bandwidth peak | 128-bit bus × 24 Gbps (GDDR7) = **384 GB/s** |
| Balance point ("ridge") | 18.3 TFLOP/s / 384 GB/s ≈ **48 FLOP per byte** |

A kernel with an arithmetic intensity well below 48 FLOP/byte is **memory-bound**: its best possible speed is set by DRAM bandwidth. A kernel well above it is **compute-bound**: its best possible speed is set by the compute peak. This is the *roofline model* (PMPP ch. 5). The 2.75 GHz clock is measured, not taken from the specification sheet ([section 3](#3-methodology-correctness-timing-clock)).

The two kernel families studied here sit on opposite sides of this balance point: element-wise kernels ([section 4](#4-memory-bound-kernels-how-close-to-the-bandwidth-limit)) are memory-bound, matrix multiplication ([section 5](#5-matrix-multiplication-testing-the-pmpp-techniques-against-cublas)) is compute-bound in principle. Before looking at them, the next section describes how correctness and speed are established.

## 3. Methodology: correctness, timing, clock

A performance number is only meaningful if the kernel is correct and the measurement is reproducible. This section describes the protocol applied to every kernel; the results in sections 4 and 5 all follow it.

### 3.1 Correctness before speed

Each kernel is compared with a CPU reference before it is ever timed. For matrix multiplication, which has the most room for indexing errors, the test ([`ch3/matmul/test_matmul.cu`](ch3/matmul/test_matmul.cu), shared by all variants including cuBLAS) uses:

- **9 matrix shapes** including degenerate (1×1×1), thin (one row or one column), rectangular, and sizes that are not multiples of the tile size, where boundary bugs appear. An earlier version tested only square matrices and missed a bounds error that only shows on rectangular ones.
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

The first kernels of PMPP perform very few operations per element. Their arithmetic intensity is far below the balance point of [section 2](#2-background-what-limits-a-gpu-kernel), so their speed is limited by DRAM bandwidth, and the right question is: **what fraction of the 384 GB/s do they achieve?**

### 4.1 Kernels

- **Vector addition** (`vecadd`, PMPP ch. 2): C[i] = A[i] + B[i]. One addition per 12 bytes moved (two 4-byte reads, one write), i.e. 0.08 FLOP/byte.
- **RGB to grayscale** (`color2gray`, PMPP ch. 3): each pixel's three color values are combined as 0.21·r + 0.72·g + 0.07·b. About 5 operations per 16 bytes moved (three reads, one write), i.e. 0.3 FLOP/byte.

Both are 150 to 600 times below the balance point: even a perfect implementation is limited by bandwidth. Input sizes are chosen far larger than the 32 MiB L2 cache, otherwise the measurement would reflect the cache rather than DRAM.

### 4.2 Results

| Kernel | Problem size | Data moved | Median time | Effective bandwidth | % of 384 GB/s |
|---|---|---|---|---|---|
| `vecadd` | 2²⁶ floats per array | 768 MiB | 2.317 ms | **347.6 GB/s** | **90.5 %** |
| `color2gray` | 4096 × 4096 pixels | 256 MiB | 0.806 ms | **333.2 GB/s** | **86.8 %** |

### 4.3 Discussion

Reaching 85-95 % of theoretical DRAM bandwidth is what a well-written streaming kernel is expected to achieve: the theoretical figure is never fully attainable in practice (DRAM refresh, read/write turnarounds). Both kernels use *coalesced* accesses, meaning that the 32 threads of a warp read consecutive addresses, which the hardware combines into a few wide memory transactions.

`color2gray` is slightly below `vecadd`. A plausible explanation is its access pattern: each thread reads three consecutive floats, so for a given load instruction the threads of a warp access addresses 12 bytes apart instead of 4, and each instruction touches more memory segments; the L1 cache recovers most of the waste. This has not been profiled yet.

An earlier version of `color2gray` used the constants `0.21`, `0.72`, `0.07`, which C++ treats as **double-precision** values: the computation silently ran in FP64, which consumer GPUs execute at a small fraction of their FP32 rate. Writing `0.21f` removed the FP64 instructions and reduced register usage from 15 to 12. The kernel being memory-bound, the effect on its speed is small, but the same mistake in a compute-bound kernel would be costly.

**Conclusion of section 4.** For memory-bound kernels, the PMPP implementations are already close to the hardware limit, and little remains to gain. Matrix multiplication is the opposite case: its arithmetic intensity allows it, in principle, to be limited by compute, and the techniques of the book aim precisely at reaching that regime. It is studied next.

## 5. Matrix multiplication: testing the PMPP techniques against cuBLAS

Matrix multiplication (*SGEMM* in BLAS terminology: Single-precision GEneral Matrix Multiply) computes C = A·B, where each element of C is a dot product of a row of A and a column of B. It is the reference kernel of GPU computing: it dominates the cost of deep learning and many scientific codes, and NVIDIA's cuBLAS library provides a highly tuned implementation.

For n × n matrices, SGEMM performs 2n³ operations on 3n² values. At n = 4096 this is 137 GFLOP for 201 MB of data, i.e. an arithmetic intensity of up to 680 FLOP/byte, far above the balance point of 48. **SGEMM can therefore be compute-bound, but only if each value loaded from DRAM is reused many times.** PMPP chapters 3, 5 and 6 introduce successive techniques to achieve this reuse. This section measures what each technique gains, compares the result with cuBLAS, and explains the remaining gap.

### 5.1 Kernels studied

| Kernel | PMPP | Idea | Expected effect |
|---|---|---|---|
| **Naive** | ch. 3 | One thread computes one element of C, reading its row of A and column of B directly from global memory. | Each value is loaded from memory n times in the worst case; relies on caches for reuse. |
| **Tiled** (16×16 or 32×32) | ch. 5 | Each block loads square *tiles* of A and B into shared memory, then all its threads compute from these on-chip copies. | Global-memory traffic divided by the tile width T. |
| **Tiled + thread coarsening** (×1, ×2, ×4) | ch. 6 | Each thread computes several elements of C (in different column tiles), reusing the same tile of A. | Less redundant loading of A tiles, fewer blocks. |
| **cuBLAS** | — | NVIDIA's library, used as the reference. | Best achievable performance in practice. |

Tile size and coarsening factor are compile-time parameters, so a single script ([`scripts/sweep_matmul.sh`](scripts/sweep_matmul.sh)) builds, tests and benchmarks all 10 configurations at 4 matrix sizes. Each configuration must pass the correctness test of [section 3.1](#31-correctness-before-speed) before being timed.

### 5.2 Results

![SGEMM throughput vs matrix size](results/matmul_gflops.png)

| Kernel | n = 512 | n = 1024 | n = 2048 | n = 4096 |
|---|---|---|---|---|
| Naive (ch. 3) | 998 GFLOP/s (16 %) | 1,065 (12 %) | 1,084 (11 %) | 769 (7 %) |
| Tiled 16×16 (ch. 5) | 1,364 (22 %) | 1,463 (17 %) | 1,482 (14 %) | 1,425 (13 %) |
| Tiled 32×32 (ch. 5) | 1,344 (22 %) | 1,406 (16 %) | 1,428 (14 %) | 1,263 (12 %) |
| **Tiled 16×16 + coarsening ×4 (ch. 6)** | 1,423 (23 %) | 1,533 (18 %) | **1,580 (15 %)** | **1,500 (14 %)** |
| cuBLAS | 6,191 | 8,595 | 10,291 | 10,628 |

Median of 30 runs; percentages are relative to cuBLAS at the same size. All 40 measurements: [`results/sweep_matmul.csv`](results/sweep_matmul.csv); clock and environment of the run: [`results/env_matmul.txt`](results/env_matmul.txt).

The best hand-written kernel is **1.95× faster than the naive one** at n = 4096, but reaches **14 % of cuBLAS**. In absolute terms, cuBLAS reaches about 58 % of the FP32 peak of section 2 and the best hand-written kernel about 8 %. The rest of this section explains these two observations: the modest gain of the PMPP techniques, and the large gap to cuBLAS.

### 5.3 Effect of each technique

![Tile size x coarsening factor at n = 2048](results/matmul_configs.png)

**Tiling helps, but much less than the reduction in memory traffic would suggest.** A 16×16 tile divides global-memory traffic by 16, yet the speedup over the naive kernel is only 1.37× at n = 2048. The naive kernel was therefore not limited by DRAM: its redundant reads were already served by the L1 and L2 caches. The exception confirms the rule: at n = 4096, a single matrix (64 MiB) no longer fits in the 32 MiB L2 cache, and the naive kernel loses 29 % of its throughput, while the tiled kernels lose 4-12 % and cuBLAS nothing.

**A larger tile is slower.** A 32×32 tile halves global traffic again compared with 16×16, but is 4 % slower at n = 2048 and 13 % slower at n = 4096. The reason lies in occupancy. A 32×32 tile requires blocks of 1,024 threads, and only one such block fits on an SM (limit: 1,536 threads), so only 32 of the 48 warp slots are used (67 % occupancy), and all of them wait at the same synchronization barriers (`__syncthreads()`). With 16×16 tiles, six independent blocks of 256 threads share the SM, and one block can compute while another waits for memory. This again shows that global traffic is not the limiting factor. *(The barrier-stall explanation is a hypothesis to be confirmed with Nsight Compute.)*

**Thread coarsening gives a small gain:** +5 % with factor 2 and +7 % with factor 4 at n = 2048, for 16×16 tiles. [Section 5.4](#54-why-coarsening-gains-little-comparison-with-a-published-study) explains why the gain is small. As a sanity check, coarsening with factor 1 reproduces the plain tiled kernel (1,474 vs 1,482 GFLOP/s), so the coarsened code adds no overhead and the measurements are consistent.

**Small problems expose wave quantization.** Blocks are distributed to SMs in "waves". With 32×32 tiles and coarsening ×4 at n = 512, the grid has 64 blocks for 26 SMs holding one block each: 3 waves, the last one using 12 of the 26 SMs. The expected efficiency is 64/78 = 82 %; the measured throughput is 86 % of the same kernel with coarsening ×2, whose 128 blocks fill 5 waves almost completely (128 of 130 slots).

In summary, the measurements show that DRAM traffic was not the bottleneck of these kernels, which explains why techniques designed to reduce it bring limited gains. The next subsection identifies what the bottleneck is, using a published study as a point of comparison.

### 5.4 Why coarsening gains little: comparison with a published study

Simon Boehm's [worklog on optimizing SGEMM](https://siboehm.com/articles/22/CUDA-MMM) applies a similar sequence of techniques on another GPU (RTX A6000, Ampere) and reports each step as a percentage of cuBLAS. This makes it possible to check our results against an independent measurement and to locate where they diverge.

| Step | Boehm (A6000, n = 4092) | This repository (n = 4096) |
|---|---|---|
| Naive with coalesced accesses | 8.5 % of cuBLAS | 7.2 % |
| 32×32 shared-memory tiling | 12.8 % | 11.9 % |
| Several outputs per thread | 36.5 % (8 outputs per thread), **×2.85** | 14.1 % (coarsening ×4), **×1.05** over tiling |

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

Boehm's kernel puts the loop over K outside and the loop over outputs inside: each value read from shared memory is kept in a **register** and reused for 8 multiply-adds, which almost halves shared-memory traffic per output. Boehm's profiler shows that the tiled kernel spends most of its time waiting on the shared-memory pipeline (*MIO throttle* stalls), which is consistent with this explanation; it remains to be confirmed with Nsight Compute on this GPU.

**This is the central result of section 5.** The chapter 5-6 techniques reduce traffic to global memory, but once the data is in caches and shared memory, the limit becomes the rate at which shared memory can feed the arithmetic units. Closing the gap requires keeping more data in registers, a technique called *register tiling*. The next subsection shows that this is exactly what cuBLAS does.

### 5.5 What cuBLAS does differently

To know which algorithm cuBLAS runs, Nsight Compute was used to name the kernel it launches at n = 4096: `cutlass_80_simt_sgemm_128x64_8x5_nn_align1`. The name describes its design:

- `simt`: it runs on the standard FP32 units, **not on Tensor Cores** (which would use reduced-precision TF32 arithmetic). The comparison with the hand-written FP32 kernels is therefore fair.
- `128x64`: each block of 128 threads computes a 128×64 tile of C, i.e. **64 outputs per thread kept in registers** (1 to 4 in the hand-written kernels). This is the register tiling identified in section 5.4.
- `8x5`: tiles of A and B are processed 8 columns at a time through a **5-stage pipeline**, so that the next tiles are loaded while the current one is computed.

Its FP32 multiply-add units are busy **69 %** of the time according to the profiler (`sm__pipe_fma_cycles_active`), at a clock of 2.69 GHz measured during the kernel.

**Conclusion of section 5.** The PMPP techniques were measured one by one: tiling and coarsening give a combined 1.95× speedup, consistent with an independent study, but leave the kernel at 14 % of cuBLAS. The measurements and the comparison show that the bottleneck has moved from global memory to shared memory, and the analysis of cuBLAS confirms that register tiling and software pipelining are what is missing.

## 6. Conclusion and next steps

This repository implemented the kernels of PMPP chapters 2-6 and measured them with a protocol designed for a laptop GPU (correctness tests, statistical timing, measured clock). Three conclusions emerge:

1. **Memory-bound kernels** are close to the hardware limit (87-90 % of DRAM bandwidth) with the book's straightforward implementations.
2. **For matrix multiplication**, tiling and coarsening bring a 1.95× speedup but stay at 14 % of cuBLAS, in agreement with an independent published study.
3. **The reason** is that these techniques reduce global-memory traffic, whereas the bottleneck becomes shared-memory traffic per operation; cuBLAS removes it with register tiling.

Some explanations above are still hypotheses, and some techniques of the book remain to be studied. Planned work, in order of priority:

- [ ] **Profile the hand-written SGEMM kernels with Nsight Compute** to confirm the shared-memory bottleneck (stall reasons, pipe utilization) and the barrier stalls of the 32×32 tile.
- [ ] **Parallel reduction** (PMPP ch. 10), from the naive version to warp-level primitives, compared with NVIDIA's CUB library.
- [ ] **Roofline plot** of all kernels from profiler counters.
- [ ] **Register tiling for SGEMM**: move the K loop outside the output loop, then compute 2D blocks of outputs per thread with vectorized loads (36.5 % then 78.4 % of cuBLAS in Boehm's study).

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
├── cublas/                     # cuBLAS SGEMM reference, same interface as the hand-written kernels
├── scripts/
│   ├── sweep_matmul.sh         # builds, tests and benchmarks every SGEMM configuration
│   └── plot_matmul.py          # tables and figures from the sweep results
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