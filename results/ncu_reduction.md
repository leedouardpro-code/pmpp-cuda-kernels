### Nsight Compute, sum reduction kernels (one launch, real clock)

| Metric | `simple` | `shared` | `coarsened` | `cub` |
|---|---|---|---|---|
| Duration (ms) | 1.711 | 1.142 | 0.728 | 0.777 |
| Memory SOL (%) | 80.0 | 80.8 | 96.5 | 98.6 |
| DRAM throughput (%) | 80.0 | 61.6 | 96.5 | 98.6 |
| Active threads per warp instruction (of 32) | 25.28 | 31.94 | 31.91 | 31.93 |
| Global load requests (M) | 16.7 | 2.1 | 2.1 | 1.0 |
| Global load sectors (M) | 83.8 | 8.4 | 8.4 | 8.4 |
| Global store sectors (M) | 50.3 | 0.0 | 0.0 | 0.0 |
| Shared-memory wavefronts (M) | 3.7 | 31.1 | 7.8 | 0.1 |
| Instructions executed (M) | 208.4 | 146.0 | 37.9 | 4.8 |
| Achieved occupancy (%) | 92.6 | 64.4 | 95.2 | 97.2 |
| Barrier stalls (cycles per instruction) | 5.34 | 3.30 | 6.68 | 6.73 |
| Global load sectors per request (4 = fully coalesced) | 5.02 | 4.00 | 4.00 | 8.00 |

### Top warp-stall reasons (cycles per issued instruction)

| Version | 1st | 2nd | 3rd | 4th |
|---|---|---|---|---|
| `simple` | long_scoreboard (10.3) | barrier (5.3) | wait (3.4) | not_selected (1.2) |
| `shared` | long_scoreboard (6.5) | short_scoreboard (4.8) | wait (3.6) | barrier (3.3) |
| `coarsened` | long_scoreboard (41.7) | barrier (6.7) | short_scoreboard (3.9) | wait (3.0) |
| `cub` | long_scoreboard (490.2) | barrier (6.7) | wait (2.0) | selected (1.0) |

Kernels: `simple` = Reduction(const float *, float *, float *, int); `shared` = Reduction(const float *, float *, int); `coarsened` = Reduction(const float *, float *, float *, int); `cub` = void DeviceReduceKernel<policy_selector_from_types<float, unsigned int, plus<void>>, const float *, unsigned int, plus<void>, float, identity>(T2, T5 *, T3, GridEvenShare<T3>, T4, T6)
