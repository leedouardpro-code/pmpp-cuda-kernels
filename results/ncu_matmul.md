### Nsight Compute, SGEMM kernels (one launch, real clock)

| Metric | `naive` | `tiled16` | `tiled32` | `tiled16_c4` | `regtile1d` | `regtile2d` | `cublas` |
|---|---|---|---|---|---|---|---|
| Duration (ms) | 189.66 | 94.68 | 108.86 | 91.30 | 29.47 | 22.52 | 12.18 |
| SM clock (GHz) | 2.74 | 2.73 | 2.74 | 2.73 | 2.72 | 2.68 | 2.61 |
| Compute SOL (%) | 63.7 | 95.8 | 76.3 | 93.1 | 97.3 | 60.8 | 79.1 |
| Memory SOL (%) | 63.7 | 95.8 | 76.3 | 93.1 | 97.3 | 92.3 | 67.4 |
| DRAM throughput (%) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| DRAM read (MB) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| L2 hit rate (%) | 49.4 | 49.8 | 49.4 | 20.1 | 48.7 | 68.1 | 94.0 |
| L1 hit rate (%) | 95.0 | 0.0 | 0.0 | 0.2 | 0.0 | 60.0 | 0.0 |
| Theoretical occupancy (%) | 66.7 | 100.0 | 66.7 | 100.0 | 66.7 | 50.0 | 25.0 |
| Achieved occupancy (%) | 66.7 | 99.3 | 66.6 | 98.6 | 66.4 | 49.7 | 24.8 |
| FMA pipe active (%) | 8.7 | 10.2 | 7.9 | 9.6 | 28.1 | 37.8 | 69.3 |
| LSU pipe active (%) | 63.8 | 96.5 | 76.3 | 95.1 | 97.7 | 61.1 | 58.7 |
| Shared-mem load wavefronts (M) | 0.0 | 3,221.3 | 3,221.2 | 3,221.3 | 1,345.0 | 807.6 | 269.1 |
| Shared-mem load instructions (M) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Shared-mem load bank conflicts (M) | 0.0 | 0.1 | 0.0 | 0.0 | 2.8 | 2.3 | 0.1 |
| Instructions executed (M) | 10,489.4 | 7,806.6 | 6,330.3 | 6,384.3 | 3,599.0 | 3,020.6 | 2,613.7 |

### Top warp-stall reasons (cycles per issued instruction)

| Kernel | 1st | 2nd | 3rd | 4th |
|---|---|---|---|---|
| `naive` | lg_throttle (17.1) | long_scoreboard (16.5) | not_selected (2.9) | wait (2.2) |
| `tiled16` | mio_throttle (16.7) | long_scoreboard (13.5) | barrier (4.4) | not_selected (2.7) |
| `tiled32` | mio_throttle (21.1) | long_scoreboard (8.4) | not_selected (3.2) | barrier (2.3) |
| `tiled16_c4` | long_scoreboard (20.8) | mio_throttle (11.0) | short_scoreboard (8.9) | barrier (4.2) |
| `regtile1d` | mio_throttle (9.0) | not_selected (2.3) | short_scoreboard (2.0) | long_scoreboard (1.8) |
| `regtile2d` | mio_throttle (3.6) | long_scoreboard (3.5) | not_selected (1.8) | selected (1.0) |
| `cublas` | not_selected (1.3) | selected (1.0) | barrier (0.4) | mio_throttle (0.4) |

Kernel names: `naive` = Matmul(const float *, const float *, float *, int, int, int); `tiled16` = Matmul_tiling(const float *, const float *, float *, int, int, int); `tiled32` = Matmul_tiling(const float *, const float *, float *, int, int, int); `tiled16_c4` = Matmul_coarsening(const float *, const float *, float *, int, int, int); `regtile1d` = Matmul_register_tiling(const float *, const float *, float *, int, int, int); `regtile2d` = Matmul_register_tiling(const float *, const float *, float *, int, int, int); `cublas` = void Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>(Params)
