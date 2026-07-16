#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "color2gray.cuh"

static int check(int width, int height, const float tolerence) {
    const size_t size = sizeof(float) * width * height;

    // Host memory allocation
    float *Pin_h, *Pout_h_gpu,*Pout_h_cpu;
    Pin_h = (float*)malloc(size * 3);
    Pout_h_gpu = (float*)malloc(size);
    Pout_h_cpu = (float*)malloc(size);

    // Initialisation
    for (int i = 0; i < width * height; i++) {
        int j = i * 3;
        Pin_h[j] = (float)j;
        Pin_h[j+1] = (float)(j+1);
        Pin_h[j+2] = (float)(j+2);
    }

    // Kernel execution
    color2grayscale_gpu(Pin_h, Pout_h_gpu, width, height);

    // CPU execution
    color2grayscale_cpu(Pin_h, Pout_h_cpu, width, height);

    // Verification
    for (int i = 0; i < width*height; i++) {
        if (fabsf(Pout_h_gpu[i] - Pout_h_cpu[i]) > tolerence) {
            printf("[FAIL] test : width=%d, height=%d\n at index=%d:\n gpu:%f\n cpu:%f\n", width, height, i, Pout_h_gpu[i],
                   Pout_h_cpu[i]);
            free(Pin_h);
            free(Pout_h_gpu);
            free(Pout_h_cpu);
            return 1;
        }
    }
    printf("[PASS] test : width=%d, height=%d\n", width, height);

    // Free Host memory
    free(Pin_h);
    free(Pout_h_gpu);
    free(Pout_h_cpu);
    return 0;
}

int main(void) {
    int width[] = {1, 2 << 7, 527, 2 << 12};
    int height[] = {1, 523, 2 << 6, 2 << 15};
    int failures = 0;

    // test harness
    for (int i = 0; i < sizeof(width) / sizeof(width[0]); i++) {
        failures += check(width[i], height[i], 1e-5);
    }

    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}