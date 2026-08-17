#include <cuda_runtime.h>
#include <stdio.h>

#include "cuda_check.h"
#include "device_query.h"

void print_device_props(int dev) {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));

    int maxBlocksPerSM = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&maxBlocksPerSM, cudaDevAttrMaxBlocksPerMultiprocessor, dev));
    int clockKHz = 0, memClockKHz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, dev));

    printf("===== Device %d: %s =====\n", dev, p.name);
    printf("Compute capability      : %d.%d\n", p.major, p.minor);
    printf("SM count                : %d\n", p.multiProcessorCount);
    printf("Clock rate (SM)         : %.2f GHz\n", clockKHz * 1e-6);

    printf("\n-- Execution limits --\n");
    printf("Warp size               : %d\n", p.warpSize);
    printf("Max threads / block     : %d\n", p.maxThreadsPerBlock);
    printf("Max threads per SM      : %d\n", p.maxThreadsPerMultiProcessor);
    printf("Max blocks per SM       : %d\n", p.maxBlocksPerMultiProcessor);
    printf("Max block dim           : (%d, %d, %d)\n", p.maxThreadsDim[0], p.maxThreadsDim[1],
           p.maxThreadsDim[2]);

    printf("\n-- Memory --\n");
    printf("Global memory           : %.2f GB\n", p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("L2 cache                : %d KB\n", p.l2CacheSize / 1024);
    printf("Memory bus width        : %d bits\n", p.memoryBusWidth);
    printf("Memory clock            : %.2f GHz\n", memClockKHz * 1e-6);
    printf("Shared memory per block : %f KB\n", p.sharedMemPerBlock * 1e-3);
    printf("Shared memory per SM    : %f KB\n", p.sharedMemPerMultiprocessor * 1e-3);
    printf("Registers per block     : %d \n", p.regsPerBlock);
    printf("Registers per SM        : %d \n", p.regsPerMultiprocessor);
    printf("Shared opt-in / block   : %.1f KB\n", p.sharedMemPerBlockOptin / 1024.0);

    printf("\n-- Derived --\n");
    double bw = 2.0 * memClockKHz * 1e3 * (p.memoryBusWidth / 8.0) / 1e9;
    double GFLOPS = 1e-6 * 128 * p.multiProcessorCount * clockKHz * 2;
    printf("Theoretical BW          : %.1f GB/s\n", bw);
    printf("Theoretical pic GFLOPS  : %.1f GFLOPS\n", GFLOPS);

    printf("\n");
}