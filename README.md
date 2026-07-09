# pmpp-cuda-kernels

Hand-written CUDA kernels following *Programming Massively Parallel Processors* (4th edition, Hwu, Kirk & El Hajj), studied alongside the [GPU MODE](https://github.com/gpu-mode/lectures) lecture series.

**Goal:** build a solid understanding of GPU architecture (memory hierarchy, coalescing, occupancy, warp scheduling) as groundwork for deploying quantized TensorRT detection models on a **Jetson Orin Nano**, with Nsight profiling.

Every kernel here is written from scratch.

## Progress

| Chapter | Topic | Kernel(s) | Status | Key takeaway |
|---|---|---|---|---|
| 2 | Heterogeneous data parallel computing | `vecadd` | 🔄 in progress | — |
| 3 | Multidimensional grids and data | — | ⬜ | — |
| 4 | Compute architecture and scheduling | — | ⬜ | — |
| 5 | Memory architecture and data locality | — | ⬜ | — |
| 6 | Performance considerations | — | ⬜ | — |

*(Table updated as I progress. "Key takeaway" = the one thing I'd mention about this kernel in an interview.)*

## Repository structure

```
pmpp-cuda-kernels/
├── common/
│   └── cuda_check.h      # CUDA_CHECK error-handling macro, shared by all kernels
├── ch02_vecadd/
│   └── vecadd.cu
└── ...
```

One folder per chapter. Each folder contains the kernel(s), and (from chapter 5 onward) profiling notes.

## Hardware & environment

| | |
|---|---|
| **Current** | Google Colab — NVIDIA T4 (Turing, SM 7.5), CUDA 12.x |
| **Target** | Jetson Orin Nano (Ampere, SM 8.7) — planned |
| **Editor** | VS Code + Nsight Visual Studio Code Edition, AI completions disabled |

## Build & run

```bash
# On any machine with nvcc (adapt -arch to your GPU, check with nvidia-smi)
nvcc -arch=sm_75 ch02_vecadd/vecadd.cu -o vecadd
./vecadd
```

On Colab:

```python
!git clone https://github.com/<user>/pmpp-cuda-kernels.git
%cd pmpp-cuda-kernels
!nvcc -arch=sm_75 ch02_vecadd/vecadd.cu -o vecadd && ./vecadd
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