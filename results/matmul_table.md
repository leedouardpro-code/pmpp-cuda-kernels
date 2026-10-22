### SGEMM, square M = N = K, FP32 (median of 30 runs)

| Kernel | n = 512 | n = 1024 | n = 2048 | n = 4096 |
|---|---|---|---|---|
| `naive` | 1,017 GFLOP/s (16 %) | 1,054 GFLOP/s (12 %) | 1,084 GFLOP/s (10 %) | 769 GFLOP/s (7 %) |
| `tiled t16 c1` | 1,369 GFLOP/s (21 %) | 1,463 GFLOP/s (16 %) | 1,475 GFLOP/s (14 %) | 1,406 GFLOP/s (13 %) |
| `tiled t32 c1` | 1,350 GFLOP/s (21 %) | 1,408 GFLOP/s (16 %) | 1,420 GFLOP/s (14 %) | 1,262 GFLOP/s (12 %) |
| `tiled coarsened t16 c4` | 1,426 GFLOP/s (22 %) | 1,533 GFLOP/s (17 %) | 1,580 GFLOP/s (15 %) | 1,542 GFLOP/s (14 %) |
| `cublas` | 6,438 GFLOP/s (100 %) | 8,947 GFLOP/s (100 %) | 10,432 GFLOP/s (100 %) | 10,670 GFLOP/s (100 %) |

_Percentages are relative to cuBLAS at the same size._

### Tile x coarsening grid, n = 2048 (GFLOP/s, % of cuBLAS)

| Tile \ coarse | 1 | 2 | 4 |
|---|---|---|---|
| 16 | 1,474 (14.1 %) | 1,544 (14.8 %) | 1,580 (15.1 %) |
| 32 | 1,420 (13.6 %) | 1,474 (14.1 %) | 1,478 (14.2 %) |

_Runs with stddev > 2 % of the median:_ `naive` n=512 (2.5 %), `tiled t16 c1` n=512 (3.0 %), `tiled t16 c1` n=4096 (6.0 %), `tiled coarsened t16 c1` n=512 (4.0 %), `tiled coarsened t16 c1` n=4096 (6.0 %), `tiled coarsened t16 c2` n=512 (3.5 %), `tiled coarsened t16 c2` n=4096 (5.2 %), `tiled coarsened t16 c4` n=512 (2.3 %), `tiled coarsened t16 c4` n=4096 (4.7 %), `tiled coarsened t32 c2` n=512 (5.5 %), `tiled coarsened t32 c4` n=512 (5.0 %), `cublas` n=512 (20.4 %), `cublas` n=1024 (3.7 %)
