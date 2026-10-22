### SGEMM, square M = N = K, FP32 (median of 30 runs)

| Kernel | n = 512 | n = 1024 | n = 2048 | n = 4096 |
|---|---|---|---|---|
| `naive` | 998 GFLOP/s (16 %) | 1,065 GFLOP/s (12 %) | 1,084 GFLOP/s (11 %) | 769 GFLOP/s (7 %) |
| `tiled t16 c1` | 1,364 GFLOP/s (22 %) | 1,463 GFLOP/s (17 %) | 1,482 GFLOP/s (14 %) | 1,425 GFLOP/s (13 %) |
| `tiled t32 c1` | 1,344 GFLOP/s (22 %) | 1,406 GFLOP/s (16 %) | 1,428 GFLOP/s (14 %) | 1,263 GFLOP/s (12 %) |
| `tiled coarsened t16 c4` | 1,423 GFLOP/s (23 %) | 1,533 GFLOP/s (18 %) | 1,580 GFLOP/s (15 %) | 1,500 GFLOP/s (14 %) |
| `cublas` | 6,191 GFLOP/s (100 %) | 8,595 GFLOP/s (100 %) | 10,291 GFLOP/s (100 %) | 10,628 GFLOP/s (100 %) |

_Percentages are relative to cuBLAS at the same size._

### Tile x coarsening grid, n = 2048 (GFLOP/s, % of cuBLAS)

| Tile \ coarse | 1 | 2 | 4 |
|---|---|---|---|
| 16 | 1,474 (14.3 %) | 1,544 (15.0 %) | 1,580 (15.4 %) |
| 32 | 1,420 (13.8 %) | 1,474 (14.3 %) | 1,478 (14.4 %) |

_Runs with stddev > 2 % of the median:_ `naive` n=512 (3.9 %), `tiled t16 c1` n=512 (13.6 %), `tiled t16 c1` n=4096 (4.6 %), `tiled coarsened t16 c1` n=4096 (4.8 %), `tiled coarsened t16 c2` n=4096 (6.3 %), `tiled coarsened t16 c4` n=512 (3.8 %), `tiled coarsened t16 c4` n=4096 (4.4 %), `tiled coarsened t32 c1` n=512 (6.3 %), `tiled coarsened t32 c2` n=512 (5.7 %), `cublas` n=512 (14.3 %), `cublas` n=1024 (5.3 %), `cublas` n=4096 (2.6 %)
