### Nsight Compute, SGEMM kernels (one launch, real clock)

| Metric | `naive` | `tiled16` | `tiled32` | `tiled16_c4` | `cublas` |
|---|---|---|---|---|---|
| Duration (ms) | 189.66 | 95.30 | 109.25 | 90.24 | 12.19 |
| SM clock (GHz) | 2.74 | 2.73 | 2.74 | 2.70 | 2.60 |
| Compute SOL (%) | 63.7 | 95.3 | 76.0 | 95.5 | 79.2 |
| Memory SOL (%) | 63.7 | 95.3 | 76.0 | 95.5 | 67.4 |
| DRAM throughput (%) | n/a | n/a | n/a | n/a | n/a |
| DRAM read (MB) | n/a | n/a | n/a | n/a | n/a |
| L2 hit rate (%) | 49.3 | 49.8 | 49.4 | 20.1 | 94.0 |
| L1 hit rate (%) | 95.0 | 0.0 | 0.0 | 0.2 | 0.0 |
| Theoretical occupancy (%) | 66.7 | 100.0 | 66.7 | 100.0 | 25.0 |
| Achieved occupancy (%) | 66.7 | 100.0 | 66.7 | 98.5 | 24.8 |
| FMA pipe active (%) | 8.7 | 10.2 | 7.9 | 9.7 | 69.3 |
| LSU pipe active (%) | 63.7 | 96.7 | 76.4 | 95.7 | 58.7 |
| Shared-mem load wavefronts (M) | 0.0 | 3,221.3 | 3,221.2 | 3,221.3 | 269.1 |
| Shared-mem load bank conflicts (M) | 0.0 | 0.1 | 0.0 | 0.0 | 0.1 |
| Instructions executed (M) | 10,489.4 | 7,806.6 | 6,330.3 | 6,384.3 | 2,613.7 |

### Top warp-stall reasons (cycles per issued instruction)

| Kernel | 1st | 2nd | 3rd | 4th |
|---|---|---|---|---|
| `naive` | lg_throttle (17.1) | long_scoreboard (16.5) | not_selected (2.9) | wait (2.2) |
| `tiled16` | long_scoreboard (16.3) | mio_throttle (15.2) | barrier (4.2) | not_selected (2.5) |
| `tiled32` | mio_throttle (21.1) | long_scoreboard (8.5) | not_selected (3.2) | barrier (2.3) |
| `tiled16_c4` | long_scoreboard (20.1) | mio_throttle (11.1) | short_scoreboard (8.9) | barrier (4.2) |
| `cublas` | not_selected (1.3) | selected (1.0) | barrier (0.4) | mio_throttle (0.4) |

Kernel names: `naive` = Matmul(const float *, const float *, float *, int, int, int); `tiled16` = Matmul_tiling(const float *, const float *, float *, int, int, int); `tiled32` = Matmul_tiling(const float *, const float *, float *, int, int, int); `tiled16_c4` = Matmul_coarsening(const float *, const float *, float *, int, int, int); `cublas` = void Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>(Params)
