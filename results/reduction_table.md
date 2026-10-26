### Sum reduction, FP32 (median of 30 runs, effective bandwidth)

| Version (best configuration) | n = 2^20 (4 MiB) | n = 2^22 (16 MiB) | n = 2^24 (64 MiB) | n = 2^26 (256 MiB) |
|---|---|---|---|---|
| `simple b512 c1` | 104.9 GB/s (27 % of peak, 29 % of CUB) | 154.2 GB/s (40 % of peak, 19 % of CUB) | 146.3 GB/s (38 % of peak, 42 % of CUB) | 123.0 GB/s (32 % of peak, 34 % of CUB) |
| `reduction b512 c1` | 162.7 GB/s (42 % of peak, 45 % of CUB) | 184.2 GB/s (48 % of peak, 23 % of CUB) | 222.1 GB/s (58 % of peak, 63 % of CUB) | 233.5 GB/s (61 % of peak, 64 % of CUB) |
| `coarsened b256 c2` | 272.8 GB/s (71 % of peak, 75 % of CUB) | 340.7 GB/s (89 % of peak, 43 % of CUB) | 355.6 GB/s (93 % of peak, 102 % of CUB) | 367.8 GB/s (96 % of peak, 101 % of CUB) |
| `cub` | 361.6 GB/s (94 % of peak) | 795.0 GB/s (207 % of peak) | 350.2 GB/s (91 % of peak) | 365.9 GB/s (95 % of peak) |

_Peak = 384 GB/s (theoretical DRAM bandwidth). Sizes up to 32 MiB fit in the L2 cache._

### Coarsened kernel, n = 2^26 (GB/s, % of peak) by BLOCK_DIM x COARSE_FACTOR

| BLOCK_DIM \ COARSE_FACTOR | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 256 | 283.3 (74 %) | 367.8 (96 %) | 344.4 (90 %) | 367.6 (96 %) | 365.8 (95 %) |
| 512 | 241.6 (63 %) | 344.6 (90 %) | 363.8 (95 %) | 363.6 (95 %) | 342.5 (89 %) |
| 1024 | 140.1 (36 %) | 261.8 (68 %) | 331.9 (86 %) | 341.9 (89 %) | 363.7 (95 %) |

_Runs with stddev > 2 % of the median:_ `simple b512 c1` n=2^20 (22.0 %), `simple b512 c1` n=2^22 (16.9 %), `simple b512 c1` n=2^24 (2.8 %), `simple b512 c1` n=2^26 (8.3 %), `reduction b512 c1` n=2^20 (34.1 %), `reduction b512 c1` n=2^22 (9.1 %), `reduction b512 c1` n=2^24 (3.2 %), `reduction b512 c1` n=2^26 (2.7 %), `coarsened b256 c1` n=2^20 (36.0 %), `coarsened b256 c1` n=2^22 (14.2 %), `coarsened b256 c1` n=2^24 (6.0 %), `coarsened b256 c2` n=2^20 (72.1 %), `coarsened b256 c2` n=2^22 (19.1 %), `coarsened b256 c2` n=2^24 (7.4 %), `coarsened b256 c4` n=2^20 (27.0 %), `coarsened b256 c4` n=2^22 (32.2 %), `coarsened b256 c4` n=2^24 (5.9 %), `coarsened b256 c8` n=2^20 (63.2 %), `coarsened b256 c8` n=2^22 (66.7 %), `coarsened b256 c8` n=2^24 (4.0 %), `coarsened b256 c8` n=2^26 (3.7 %), `coarsened b256 c16` n=2^20 (11.2 %), `coarsened b256 c16` n=2^22 (49.2 %), `coarsened b256 c16` n=2^24 (4.2 %), `coarsened b512 c1` n=2^20 (40.4 %), `coarsened b512 c1` n=2^22 (10.3 %), `coarsened b512 c1` n=2^24 (3.7 %), `coarsened b512 c2` n=2^20 (79.3 %), `coarsened b512 c2` n=2^22 (17.2 %), `coarsened b512 c2` n=2^24 (4.2 %), `coarsened b512 c2` n=2^26 (2.0 %), `coarsened b512 c4` n=2^20 (99.2 %), `coarsened b512 c4` n=2^22 (28.8 %), `coarsened b512 c4` n=2^24 (4.1 %), `coarsened b512 c8` n=2^20 (78.1 %), `coarsened b512 c8` n=2^22 (50.2 %), `coarsened b512 c8` n=2^24 (7.0 %), `coarsened b512 c16` n=2^20 (60.9 %), `coarsened b512 c16` n=2^22 (59.0 %), `coarsened b512 c16` n=2^24 (6.4 %), `coarsened b1024 c1` n=2^20 (18.8 %), `coarsened b1024 c1` n=2^22 (9.8 %), `coarsened b1024 c1` n=2^24 (2.2 %), `coarsened b1024 c2` n=2^20 (38.2 %), `coarsened b1024 c2` n=2^22 (16.5 %), `coarsened b1024 c2` n=2^24 (6.8 %), `coarsened b1024 c2` n=2^26 (6.7 %), `coarsened b1024 c4` n=2^20 (44.4 %), `coarsened b1024 c4` n=2^22 (21.9 %), `coarsened b1024 c4` n=2^24 (4.5 %), `coarsened b1024 c8` n=2^20 (52.9 %), `coarsened b1024 c8` n=2^22 (21.6 %), `coarsened b1024 c8` n=2^24 (4.4 %), `coarsened b1024 c16` n=2^20 (50.0 %), `coarsened b1024 c16` n=2^22 (19.2 %), `coarsened b1024 c16` n=2^24 (19.1 %), `coarsened b1024 c16` n=2^26 (2.1 %), `cub` n=2^20 (91.4 %), `cub` n=2^22 (38.9 %), `cub` n=2^24 (4.2 %)
