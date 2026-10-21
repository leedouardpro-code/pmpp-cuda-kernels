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

## Repository structure

```
pmpp-cuda-kernels/
├── utils/
│   ├── Makefile              # execute device_query_main
│   ├── device_query_main.cu  # device_query_main, ask the hardware specifications
│   ├── device_query.h
│   ├── device_query.cu
│   └── cuda_check.h          # CUDA_CHECK error-handling macro, shared by all kernels
├── ch2/
│   └── vecadd/
│       ├── Makefile
│       ├── vecadd.cuh
│       ├── vecadd.cu
│       ├── test_vecadd.cu
│       └── bench_vecadd.cu
├── ch3/
│   ├── ...
...


```

One folder per chapter. Each folder contains the kernel(s), and (from chapter 5 onward) profiling notes.

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

On Colab:

```python
!git clone https://github.com/leedouardpro-code/pmpp-cuda-kernels.git
%cd pmpp-cuda-kernels/ch2
!nvidia-smi --query-gpu=name,compute_cap --format=csv   # check GPU & arch
!make test
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
