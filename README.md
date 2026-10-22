# pmpp-cuda-kernels

Hand-written CUDA kernels following *Programming Massively Parallel Processors* (4th edition, Hwu, Kirk & El Hajj), studied alongside the [GPU MODE](https://github.com/gpu-mode/lectures) lecture series.

**Goal:** build a solid understanding of GPU architecture (memory hierarchy, coalescing, occupancy, warp scheduling) as groundwork for deploying quantized TensorRT detection models on a **Jetson Orin Nano**, with Nsight profiling.

Every kernel here is written from scratch.

## Progress

| Chapter | Topic | Key concept(s) | Kernel(s) | Status |
|---|---|---|---|---|
| 2 | Heterogeneous data parallel computing | `parallel computing`, `kernels`, `CUDA` | `vecadd` | ✅ |
| 3 | Multidimensional grids and data | `multidimensional threading` | `matmul`, `color2gray` | ✅ |
| 4 | Compute architecture and scheduling | `modern GPU architecture`, `thread scheduling`, `occupancy`, `SIMD efficiency` | — | ✅ |
| 5 | Memory architecture and data locality | `memory access efficiency`, `performance metrics`, `tiling` | `matmul_tiling` | ✅ |
| 6 | Performance considerations | `DRAM architecture`,`coarsening`,`coalescing`,`profiling` | `matmul_coarsening` | ✅ |

## Results: SGEMM (FP32)

C = A · B, row-major, square M = N = K = n. Every kernel passes the same correctness test (9 shapes including rectangular and non-tile-multiple ones, FP64 CPU reference, relative tolerance K·ε) and `compute-sanitizer` memcheck / racecheck with 0 errors before being timed.

![SGEMM throughput vs size](results/matmul_gflops.png)

| Kernel | n = 512 | n = 1024 | n = 2048 | n = 4096 |
|---|---|---|---|---|
| Naive (ch. 3) | 998 GFLOP/s (16 %) | 1,065 (12 %) | 1,084 (11 %) | 769 (7 %) |
| Tiled, 16×16 (ch. 5) | 1,364 (22 %) | 1,463 (17 %) | 1,482 (14 %) | 1,425 (13 %) |
| Tiled, 32×32 (ch. 5) | 1,344 (22 %) | 1,406 (16 %) | 1,428 (14 %) | 1,263 (12 %) |
| **Tiled 16×16 + coarsening ×4 (ch. 6)** | 1,423 (23 %) | 1,533 (18 %) | **1,580 (15 %)** | **1,500 (14 %)** |
| cuBLAS SGEMM | 6,191 | 8,595 | 10,291 | 10,628 |

Median of 30 runs, GFLOP/s = 2·M·N·K / time. Percentages are relative to cuBLAS at the same size. Raw data: [`results/sweep_matmul.csv`](results/sweep_matmul.csv), environment and clock check: [`results/env_matmul.txt`](results/env_matmul.txt).

**Best hand-written kernel: 1.95× the naive kernel and 14 % of cuBLAS at n = 4096.** This is where these techniques are expected to land: in [Simon Boehm's SGEMM worklog](https://siboehm.com/articles/22/CUDA-MMM) (RTX A6000, n = 4092), the coalesced naive kernel reaches 8.5 % of cuBLAS and 32×32 shared-memory tiling 12.8 %; here the naive kernel (already coalesced) reaches 7.2 % and 32×32 tiling 11.9 %. The PMPP chapter 5-6 techniques are not the ones that close the gap.

### What the measurements show

![Tile size x coarsening](results/matmul_configs.png)

- **Tile 16 beats tile 32**, by 4 % at n = 2048 and 13 % at n = 4096, although tile 32 halves global-memory traffic. With 1024-thread blocks only one block fits on an SM (1536 threads max), so occupancy is capped at 67 %, and all 32 warps of that block wait together at every `__syncthreads()`. Global traffic is therefore not the limiter. *(Hypothesis: exposed memory latency at the barriers; to be confirmed with Nsight Compute stall reasons.)*
- **Thread coarsening helps a little**: +5 % (×2) and +7 % (×4) at n = 2048 with tile 16. The reason is explained in the comparison with Boehm's kernels below.
- **Coarsening ×1 gives the same result as the plain tiled kernel** (1,474 vs 1,482 GFLOP/s at n = 2048): no overhead, and a check that the harness is reproducible.
- **Wave quantization is visible**: tile 32 × coarsening 4 at n = 512 launches 64 blocks of 1024 threads on 26 SMs (1 block/SM), i.e. 3 waves with the last one 12/26 full. Expected efficiency 64/78 = 82 %; measured 86 % of the coarsening ×2 result.
- **The naive kernel loses 29 % from n = 2048 to 4096**, when each matrix (64 MiB) no longer fits in the 32 MiB L2 and its redundant reads go to DRAM. cuBLAS is unaffected.

### Comparison with Boehm's kernels: why coarsening gains 1.05× here and 2.85× there

Boehm's worklog applies the same ideas to SGEMM, which makes it a useful reference to explain the gap.

| Step | Boehm (A6000, n = 4092) | This repo (RTX 5060 Laptop, n = 4096) |
|---|---|---|
| Naive, coalesced | kernel 2: 8.5 % of cuBLAS | ch. 3: 7.2 % |
| 32×32 shared-memory tiling | kernel 3: 12.8 % | ch. 5: 11.9 % |
| Several outputs per thread | kernel 4 (1D blocktiling, 8 outputs): **36.5 %, ×2.85** | ch. 6 (coarsening ×4, tile 16): **14.1 %, ×1.05 vs tile 16** |

The first two steps land within about one point of Boehm's results on a different architecture. The third one does not, although both kernels compute several outputs per thread. The difference is **what gets reused from registers**.

Memory accesses per output element (Boehm's counts for his kernels, same method applied to the ch. 6 kernel):

| Kernel | Global loads per output | Shared-memory loads per output |
|---|---|---|
| Boehm kernel 3 (32×32 tiles) | K/16 | 2K |
| Boehm kernel 4 (1D, 8 outputs/thread) | K/32 | **9K/8** |
| ch. 6, tile 16, coarsening ×4 | ≈ K/13 | **2K** (unchanged) |

PMPP coarsening (ch. 6) reuses the **A tile across several output columns**: it cuts global traffic and redundant tile loads. But the loop over the coarsened outputs is the *outer* loop, with a `__syncthreads()` and a new B tile for each output, so every `As[ty][k]` is read again from shared memory for each output:

```cpp
for (int c = 0; c < COARSE_FACTOR; c++) {      // outer: one B tile per output
    /* load Bs for column block c */ __syncthreads();
    for (int k = 0; k < TILE_WIDTH; k++)
        Pval[c] += As[ty][k] * Bs[k][tx];       // 2 shared loads per FMA
    __syncthreads();
}
```

Boehm's kernel 4 puts the loop over K *outside* and the loop over outputs *inside*: one value is read from shared memory into a register and reused for 8 FMAs, so shared-memory loads per output drop from 2K to 9K/8. Since the tiled kernels are limited by the shared-memory pipeline (MIO throttle stalls in Boehm's profile; to be confirmed here with Nsight Compute), only the second kind of reuse moves the ceiling.

**In short:** the PMPP chapter 5-6 kernels reduce *global*-memory traffic, which was not the bottleneck once the data sits in L1/L2 and shared memory. Closing the gap with cuBLAS requires reducing *shared-memory* traffic per FMA, i.e. register tiling.

### Why cuBLAS is ~7× faster

Nsight Compute shows that cuBLAS runs `cutlass_80_simt_sgemm_128x64_8x5_nn_align1` at n = 4096: a **plain FP32 (SIMT, no Tensor Core)** CUTLASS kernel, so the comparison is fair. Its name gives the missing techniques:

- **register tiling**: a 128×64 output tile per 128-thread block, i.e. 64 outputs per thread kept in registers (vs 1 to 4 here), which turns two shared-memory loads per FMA into a small fraction of a load per FMA;
- **a 5-stage software pipeline**: the next K-tiles are loaded while the current one is computed.

Its FMA pipe is active **69 %** of the time (`sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active`), at an SM clock of **2.69 GHz** measured during the kernel. At that clock the FP32 peak is 128 × 26 × 2 × 2.69 ≈ 17.9 TFLOP/s: cuBLAS reaches ~59 % of it in the sustained benchmark, the best hand-written kernel ~8 %.

**Next step (not done here):** reorder the ch. 6 loops so that the K loop is outside the output loop (1D register tiling), then 2D register tiling and vectorized `float4` loads. In Boehm's worklog these steps reach 36.5 %, 68.7 % (8×8 outputs per thread) and 78.4 % of cuBLAS.

### Method notes

- **Kernel time only**: CUDA events around each launch, transfers measured separately. Warmup of ≥ 300 ms so the laptop GPU leaves its idle state, then 30 timed runs; the **median** is reported, with min / max / stddev in the CSV.
- **Power mode matters on a laptop.** Measured SM clock under an FP32 FMA load: 1.56 GHz in Lenovo "Quiet" mode, 2.75 GHz in "Balanced" / "Performance". All results above: **Performance mode, on AC power**. The clock is measured in a kernel with `clock64()` / `%globaltimer` before and after each sweep (`utils/clock_probe`), and during the cuBLAS kernel with Nsight Compute.
- **Remaining variability**: tile-16 runs at n = 4096 show a 4-6 % stddev (min/max up to 1.2×) while tile 32 stays below 0.2 %. The fastest configurations run at the 80 W power limit, and the clock likely oscillates; the median is robust to this.
- `ncu` is run with `--clock-control none`: by default it locks the GPU to its base clock.

## Repository structure

```
pmpp-cuda-kernels/
├── .clang-format
├── .gitignore
├── README.md
├── common.mk                   # shared build settings (nvcc flags, -arch, EXTRA defines, header tracking)
├── utils/
│   ├── Makefile                # `make query`: GPU specs, `make probe`: measured SM clock
│   ├── cuda_check.h            # CUDA_CHECK error-handling macro
│   ├── bench.h                 # statistical timing harness (warmup, 30 runs, median/min/max/stddev)
│   ├── clock_probe.cu          # real SM clock (clock64 / globaltimer) and practical FP32 peak
│   ├── device_query.h
│   ├── device_query.cu         # print_device_props(): limits, memory, theoretical peaks
│   └── device_query_main.cu
├── ch2/
│   └── vecadd/                 # vector addition (memory-bound, effective bandwidth)
│       ├── Makefile
│       ├── vecadd.cuh
│       ├── vecadd.cu           # kernel + GPU wrapper + CPU reference
│       ├── test_vecadd.cu      # GPU vs CPU correctness tests
│       └── bench_vecadd.cu     # timing (GB/s)
├── ch3/
│   ├── color2gray/             # RGB to grayscale (memory-bound)
│   │   ├── Makefile
│   │   ├── color2gray.cuh
│   │   ├── color2gray.cu
│   │   ├── test_color2gray.cu
│   │   └── bench_color2gray.cu
│   └── matmul/                 # naive SGEMM (M x N x K)
│       ├── Makefile
│       ├── matmul.cuh
│       ├── matmul.cu
│       ├── test_matmul.cu      # shared by all SGEMM folders: 9 shapes, FP64 reference
│       └── bench_matmul.cu     # shared by all SGEMM folders: GFLOP/s, CSV output
├── ch4/
│   └── README.md               # exercises: warps, divergence, occupancy
├── ch5/
│   ├── README.md               # exercises: tiling, roofline, occupancy
│   └── matmul_tiling/          # shared-memory tiled SGEMM (TILE_WIDTH set at compile time)
│       ├── Makefile
│       ├── matmul.cuh
│       ├── matmul.cu
│       ├── test_matmul.cu
│       └── bench_matmul.cu
├── ch6/
│   └── matmul_coarsening/      # tiling + thread coarsening (TILE_WIDTH, COARSE_FACTOR)
│       ├── Makefile
│       ├── matmul.cuh
│       ├── matmul.cu
│       ├── test_matmul.cu
│       └── bench_matmul.cu
├── cublas/                     # cuBLAS SGEMM reference, same interface as the hand-written kernels
│   ├── Makefile
│   ├── matmul.cuh
│   ├── matmul.cu
│   ├── test_matmul.cu
│   └── bench_matmul.cu
├── scripts/
│   ├── sweep_matmul.sh         # builds, tests and benchmarks every SGEMM variant over sizes
│   └── plot_matmul.py          # README tables and figures from the sweep CSV
└── results/                    # raw data behind the README numbers (committed)
    ├── sweep_matmul.csv
    ├── env_matmul.txt          # date, commit, toolkit, driver, measured SM clock before/after
    ├── build.log               # compiler output (registers, shared memory per variant)
    ├── matmul_table.md
    ├── matmul_gflops.png
    └── matmul_configs.png

```

One folder per kernel. Each follows the same layout: `xxx.cu` (kernel, launch function, GPU wrapper), `test_xxx.cu` (correctness vs CPU), `bench_xxx.cu` (timing), and a `Makefile` with `test` / `bench` / `clean` targets. Tile size and coarsening factor are compile-time defines (`make EXTRA="-DTILE_WIDTH=16"`).

## Hardware & environment

| | |
|---|---|
| **GPU** | NVIDIA GeForce RTX 5060 Laptop GPU (Blackwell, cc 12.0, 26 SMs, 8 GB GDDR7, 80 W) |
| **Software** | Windows + WSL2 (Ubuntu), CUDA toolkit 13.3, driver 592.15 |
| **Profiling** | Nsight Compute 2026.2, Nsight Systems 2026.1, compute-sanitizer |
| **Earlier work** | Google Colab, NVIDIA T4 (Turing, cc 7.5) |

## Build & run

Requires a CUDA toolkit >= 12.8 (Blackwell `sm_120` support). Builds for the local GPU (`-arch=native`); override with `make ARCH=sm_XX`.

```bash
git clone https://github.com/leedouardpro-code/pmpp-cuda-kernels.git
cd pmpp-cuda-kernels/utils && make query     # print GPU specs
cd ../ch5/matmul_tiling
make test     # build, then run correctness tests (GPU vs CPU)
make bench    # build, then run the benchmark
make clean
```

## Conventions

- `_h` / `_d` suffixes for host/device pointers (book convention)
- Every CUDA API call wrapped in `CUDA_CHECK(...)`
- Kernel launches followed by `cudaGetLastError()`
- Commit messages: `chXX: <what changed>`

## References

- Hwu, W-m., Kirk, D., El Hajj, I. — *Programming Massively Parallel Processors*, 4th ed., 2023
- [GPU MODE lectures](https://github.com/gpu-mode/lectures)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
