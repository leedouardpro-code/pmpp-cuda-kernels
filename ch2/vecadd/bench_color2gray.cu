// bench_color2gray.cu
// Kernel-only timing of col2gray_kernel (RGB float image -> gray float image).
// Memory-bound kernel: effective bandwidth = pixels * (3*4 read + 4 written) bytes / time.
// Default 4096 x 4096 pixels: 192 MiB read + 64 MiB written, well above the 32 MiB L2.
//
// Usage: ./bench_color2gray [width] [height] [reps]   (defaults: 4096 4096 30)

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "bench.h"
#include "color2gray.cuh"
#include "cuda_check.h"

#define CSV_HEADER "CSV_HEADER,kernel,width,height,reps,median_ms,min_ms,max_ms,stddev_ms,gbps"

int main(int argc, char** argv) {
    const int width = (argc > 1) ? atoi(argv[1]) : 4096;
    const int height = (argc > 2) ? atoi(argv[2]) : 4096;
    const int reps = (argc > 3) ? atoi(argv[3]) : 30;
    if (width <= 0 || height <= 0 || reps <= 0) {
        fprintf(stderr, "usage: %s [width > 0] [height > 0] [reps > 0]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const size_t pixels = (size_t)width * height;
    const size_t size_in = pixels * 3 * sizeof(float);
    const size_t size_out = pixels * sizeof(float);
    const dim3 blocksize(32, 32);
    const dim3 gridsize((width + blocksize.x - 1) / blocksize.x,
                        (height + blocksize.y - 1) / blocksize.y);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::vector<float> Pin_h(pixels * 3);
    srand(42);
    for (float& x : Pin_h) x = rand() / (float)RAND_MAX;

    float *Pin_d, *Pout_d;
    CUDA_CHECK(cudaMalloc((void**)&Pin_d, size_in));
    CUDA_CHECK(cudaMalloc((void**)&Pout_d, size_out));
    CUDA_CHECK(cudaMemcpy(Pin_d, Pin_h.data(), size_in, cudaMemcpyHostToDevice));

    const BenchStats k = bench_kernel(
        [&] { col2gray_kernel<<<gridsize, blocksize>>>(Pin_d, Pout_d, width, height); }, reps);

    const double bytes = (double)size_in + (double)size_out;
    const double gbps = bytes / (k.median_ms * 1e-3) / 1e9;

    printf("GPU        : %s\n", prop.name);
    printf("Kernel     : color2gray, %d x %d pixels (%.0f MiB moved)\n", width, height,
           bytes / (1024.0 * 1024.0));
    printf("Kernel time: median %.4f ms | min %.4f | max %.4f | stddev %.4f (%.1f %%) | %d reps\n",
           k.median_ms, k.min_ms, k.max_ms, k.stddev_ms, 100.0 * k.stddev_ms / k.median_ms,
           k.reps);
    printf("Effective bandwidth: %.1f GB/s (pixels*16 bytes / median time)\n", gbps);
    printf("%s\n", CSV_HEADER);
    printf("CSV,color2gray,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.1f\n", width, height, k.reps,
           k.median_ms, k.min_ms, k.max_ms, k.stddev_ms, gbps);

    CUDA_CHECK(cudaFree(Pin_d));
    CUDA_CHECK(cudaFree(Pout_d));
    return 0;
}
