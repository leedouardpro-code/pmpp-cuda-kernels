### Sum reduction, FP32 (median of 30 runs, effective bandwidth)

| Version (best configuration) | n = 2^20 (4 MiB) | n = 2^22 (16 MiB) | n = 2^24 (64 MiB) | n = 2^26 (256 MiB) |
|---|---|---|---|---|
| `simple b512 c1` | 124.2 GB/s (32 % of peak, 34 % of CUB) | 157.4 GB/s (41 % of peak, 19 % of CUB) | 148.5 GB/s (39 % of peak, 42 % of CUB) | 151.6 GB/s (39 % of peak, 45 % of CUB) |
| `shared b512 c1` | 102.6 GB/s (27 % of peak, 28 % of CUB) | 220.7 GB/s (57 % of peak, 27 % of CUB) | 218.3 GB/s (57 % of peak, 62 % of CUB) | 234.1 GB/s (61 % of peak, 70 % of CUB) |
| `coarsened b512 c16` | 388.9 GB/s (101 % of peak, 107 % of CUB) | 798.0 GB/s (208 % of peak, 98 % of CUB) | 350.5 GB/s (91 % of peak, 99 % of CUB) | 366.4 GB/s (95 % of peak, 110 % of CUB) |
| `cub` | 364.1 GB/s (95 % of peak) | 810.3 GB/s (211 % of peak) | 352.8 GB/s (92 % of peak) | 334.5 GB/s (87 % of peak) |

_Peak = 384 GB/s (theoretical DRAM bandwidth). Sizes up to 32 MiB fit in the L2 cache._

### Coarsened kernel, n = 2^26 (GB/s, % of peak) by BLOCK_DIM x COARSE_FACTOR

| BLOCK_DIM \ COARSE_FACTOR | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 256 | 284.2 (74 %) | 365.6 (95 %) | 366.3 (95 %) | 344.5 (90 %) | 365.9 (95 %) |
| 512 | 243.1 (63 %) | 343.4 (89 %) | 364.0 (95 %) | 344.3 (90 %) | 366.4 (95 %) |
| 1024 | 140.1 (36 %) | 261.9 (68 %) | 326.1 (85 %) | 340.8 (89 %) | 366.2 (95 %) |

_Runs with stddev > 2 % of the median:_ `simple b512 c1` n=2^20 (24.6 %), `simple b512 c1` n=2^22 (11.1 %), `simple b512 c1` n=2^24 (2.2 %), `simple b512 c1` n=2^26 (2.6 %), `shared b512 c1` n=2^20 (20.0 %), `shared b512 c1` n=2^22 (12.2 %), `shared b512 c1` n=2^24 (5.4 %), `coarsened b256 c1` n=2^20 (35.4 %), `coarsened b256 c1` n=2^22 (12.0 %), `coarsened b256 c1` n=2^24 (6.8 %), `coarsened b256 c2` n=2^20 (80.1 %), `coarsened b256 c2` n=2^22 (21.3 %), `coarsened b256 c2` n=2^24 (4.2 %), `coarsened b256 c2` n=2^26 (2.0 %), `coarsened b256 c4` n=2^20 (71.7 %), `coarsened b256 c4` n=2^22 (25.8 %), `coarsened b256 c4` n=2^24 (5.7 %), `coarsened b256 c4` n=2^26 (3.9 %), `coarsened b256 c8` n=2^20 (137.2 %), `coarsened b256 c8` n=2^22 (39.0 %), `coarsened b256 c8` n=2^24 (5.8 %), `coarsened b256 c16` n=2^20 (9.8 %), `coarsened b256 c16` n=2^22 (73.6 %), `coarsened b256 c16` n=2^24 (4.5 %), `coarsened b512 c1` n=2^20 (20.7 %), `coarsened b512 c1` n=2^22 (19.1 %), `coarsened b512 c1` n=2^24 (5.2 %), `coarsened b512 c2` n=2^20 (54.9 %), `coarsened b512 c2` n=2^22 (19.4 %), `coarsened b512 c2` n=2^24 (3.6 %), `coarsened b512 c2` n=2^26 (4.4 %), `coarsened b512 c4` n=2^20 (77.2 %), `coarsened b512 c4` n=2^22 (22.5 %), `coarsened b512 c4` n=2^24 (4.6 %), `coarsened b512 c8` n=2^20 (61.1 %), `coarsened b512 c8` n=2^22 (37.9 %), `coarsened b512 c8` n=2^24 (5.1 %), `coarsened b512 c16` n=2^20 (20.4 %), `coarsened b512 c16` n=2^22 (45.2 %), `coarsened b512 c16` n=2^24 (3.8 %), `coarsened b1024 c1` n=2^20 (21.9 %), `coarsened b1024 c1` n=2^22 (5.7 %), `coarsened b1024 c2` n=2^20 (26.2 %), `coarsened b1024 c2` n=2^22 (11.0 %), `coarsened b1024 c2` n=2^24 (3.1 %), `coarsened b1024 c2` n=2^26 (3.2 %), `coarsened b1024 c4` n=2^20 (109.2 %), `coarsened b1024 c4` n=2^22 (15.7 %), `coarsened b1024 c4` n=2^24 (5.0 %), `coarsened b1024 c8` n=2^20 (50.7 %), `coarsened b1024 c8` n=2^22 (27.1 %), `coarsened b1024 c8` n=2^24 (7.8 %), `coarsened b1024 c8` n=2^26 (9.1 %), `coarsened b1024 c16` n=2^20 (33.1 %), `coarsened b1024 c16` n=2^22 (60.2 %), `coarsened b1024 c16` n=2^24 (7.7 %), `cub` n=2^20 (35.7 %), `cub` n=2^22 (60.4 %), `cub` n=2^24 (6.7 %)
