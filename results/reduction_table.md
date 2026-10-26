### Sum reduction, FP32 (median of 30 runs, effective bandwidth)

| Version (best configuration) | n = 2^20 (4 MiB) | n = 2^22 (16 MiB) | n = 2^24 (64 MiB) | n = 2^26 (256 MiB) |
|---|---|---|---|---|
| `simple b512 c1` | 125.7 GB/s (33 % of peak, 38 % of CUB) | 135.8 GB/s (35 % of peak, 41 % of CUB) | 119.4 GB/s (31 % of peak, 44 % of CUB) | 119.0 GB/s (31 % of peak, 42 % of CUB) |
| `cub` | 329.7 GB/s (86 % of peak) | 328.8 GB/s (86 % of peak) | 269.9 GB/s (70 % of peak) | 281.1 GB/s (73 % of peak) |

_Peak = 384 GB/s (theoretical DRAM bandwidth). Sizes up to 32 MiB fit in the L2 cache._


_Runs with stddev > 2 % of the median:_ `simple b512 c1` n=2^20 (23.4 %), `simple b512 c1` n=2^22 (6.9 %), `simple b512 c1` n=2^24 (2.4 %), `cub` n=2^20 (107.1 %), `cub` n=2^22 (35.1 %), `cub` n=2^24 (3.5 %)
