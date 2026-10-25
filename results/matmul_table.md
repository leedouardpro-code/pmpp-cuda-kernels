### SGEMM, square M = N = K, FP32 (median of 30 runs)

| Kernel | n = 512 | n = 1024 | n = 2048 | n = 4096 |
|---|---|---|---|---|
| `naive` | 1,018 GFLOP/s (17 %) | 1,066 GFLOP/s (12 %) | 1,084 GFLOP/s (11 %) | 770 GFLOP/s (8 %) |
| `tiled t16 c1` | 1,369 GFLOP/s (22 %) | 1,462 GFLOP/s (17 %) | 1,482 GFLOP/s (15 %) | 1,445 GFLOP/s (14 %) |
| `tiled coarsened t16 c2` | 1,417 GFLOP/s (23 %) | 1,522 GFLOP/s (18 %) | 1,544 GFLOP/s (16 %) | 1,516 GFLOP/s (15 %) |
| `register tiled 1D t64 c8` | 3,370 GFLOP/s (55 %) | 4,486 GFLOP/s (52 %) | 4,640 GFLOP/s (47 %) | 4,426 GFLOP/s (44 %) |
| `register tiled 2D t64 c4` | 4,161 GFLOP/s (68 %) | 5,717 GFLOP/s (66 %) | 6,002 GFLOP/s (61 %) | 5,598 GFLOP/s (55 %) |
| `cublas` | 6,157 GFLOP/s (100 %) | 8,625 GFLOP/s (100 %) | 9,806 GFLOP/s (100 %) | 10,122 GFLOP/s (100 %) |

_Percentages are relative to cuBLAS at the same size._

### Tile x coarsening grid, n = 2048 (GFLOP/s, % of cuBLAS)

| Tile \ coarse | 1 | 2 | 4 |
|---|---|---|---|
| 16 | 1,474 (15.0 %) | 1,544 (15.7 %) | 1,580 (16.1 %) |
| 32 | 1,420 (14.5 %) | 1,474 (15.0 %) | 1,478 (15.1 %) |

_Runs with stddev > 2 % of the median:_ `naive` n=512 (4.1 %), `tiled t16 c1` n=4096 (4.0 %), `tiled coarsened t16 c1` n=4096 (5.2 %), `tiled coarsened t16 c2` n=512 (2.5 %), `tiled coarsened t16 c2` n=4096 (4.3 %), `tiled coarsened t16 c4` n=512 (12.0 %), `tiled coarsened t16 c4` n=4096 (7.1 %), `register tiled 1D t32 c4` n=512 (4.5 %), `register tiled 1D t32 c4` n=4096 (10.2 %), `register tiled 1D t32 c8` n=4096 (11.8 %), `register tiled 1D t64 c4` n=512 (3.1 %), `register tiled 1D t64 c4` n=4096 (4.1 %), `register tiled 1D t64 c8` n=512 (16.2 %), `register tiled 1D t64 c8` n=1024 (4.8 %), `register tiled 1D t64 c8` n=4096 (9.0 %), `register tiled 2D t32 c4` n=512 (9.8 %), `register tiled 2D t32 c4` n=4096 (13.7 %), `register tiled 2D t32 c8` n=512 (2.7 %), `register tiled 2D t64 c4` n=512 (5.9 %), `register tiled 2D t64 c4` n=1024 (3.0 %), `register tiled 2D t64 c4` n=4096 (15.8 %), `cublas` n=512 (29.1 %), `cublas` n=1024 (6.1 %)
