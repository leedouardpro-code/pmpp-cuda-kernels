# pmpp-cuda-kernels

Hand-written CUDA kernels following *Programming Massively Parallel Processors* (4th edition, Hwu, Kirk & El Hajj), studied alongside the [GPU MODE](https://github.com/gpu-mode/lectures) lecture series.

**Goal:** build a solid understanding of GPU architecture (memory hierarchy, coalescing, occupancy, warp scheduling) as groundwork for deploying quantized TensorRT detection models on a **Jetson Orin Nano**, with Nsight profiling.

Every kernel here is written from scratch.

## Progress

| Chapter | Topic | Kernel(s) | Status |
|---|---|---|---|
| 2 | Heterogeneous data parallel computing | `vecadd` | - [x] |
| 3 | Multidimensional grids and data | 'matmul', 'color2gray' | - [x] |
| 4 | Compute architecture and scheduling | — | - [x] |
| 5 | Memory architecture and data locality | — | in progress |
| 6 | Performance considerations | — | ⬜ |

## Repository structure

```
pmpp-cuda-kernels/
├── utils/
│   └── cuda_check.h      # CUDA_CHECK error-handling macro, shared by all kernels
├── ch02_vecadd/
│   ├── Makefile
│   ├── vecadd.cuh
│   ├── vecadd.cu
│   ├── test_vecadd.cu
│   └── bench_vecadd.cu
└── ...
```

One folder per chapter. Each folder contains the kernel(s), and (from chapter 5 onward) profiling notes.

## Hardware & environment

| | |
|---|---|
| **Current** | Google Colab — NVIDIA T4 (Turing, SM 7.5), CUDA 12.x |
| **Target** | Jetson Orin Nano (Ampere, SM 8.7) — planned |
| **Editor** | VS Code + Nsight Visual Studio Code Edition |

## Build & run

Each chapter folder ships a `Makefile` following the same conventions.

```bash
cd ch02_vecadd

make            # build everything (tests + bench)
make run        # build if needed, then run the demo
make test       # build if needed, then run the tests
make bench      # build if needed, then run the performance testing
make clean      # remove binaries and objects
```

The target GPU architecture defaults to `sm_75` (Colab T4) and can be
overridden without editing any file:

```bash
make ARCH=sm_87 test    # Jetson Orin Nano
```

On Colab:

```python
!git clone https://github.com/<user>/pmpp-cuda-kernels.git
%cd pmpp-cuda-kernels/ch02_vecadd
!nvidia-smi --query-gpu=name,compute_cap --format=csv   # check GPU & arch
!make test
```
## Conventions

- `_h` / `_d` suffixes for host/device pointers (book convention)
- Every CUDA API call wrapped in `CUDA_CHECK(...)`
- Kernel launches followed by `cudaGetLastError()`; `cudaDeviceSynchronize()` in debug builds only
- Commit messages: `chXX: <what changed>`

## References

- Hwu, W-m., Kirk, D., El Hajj, I. — *Programming Massively Parallel Processors*, 4th ed., 2023
- [GPU MODE lectures](https://github.com/gpu-mode/lectures)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
