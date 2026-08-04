#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "matmul.cuh"
#define TILE_WIDTH 32

static int check(int width, int height, const float tolerence, int visual = 0) {
    const size_t size = (size_t)width * height * sizeof(float);

    // Host memory allocation
    float *A_h, *B_h, *C_h_gpu, *C_h_cpu;
    A_h = (float*)malloc(size);
    B_h = (float*)malloc(size);
    C_h_gpu = (float*)malloc(size);
    C_h_cpu = (float*)malloc(size);

    // Initialisation
    for (int i = 0; i < width * height; i++) {
        A_h[i] = rand() / (float)RAND_MAX;
        B_h[i] = rand() / (float)RAND_MAX;
    }

    // Kernel execution
    matmul_gpu(A_h, B_h, C_h_gpu, width, height);

    // CPU execution
    matmul_cpu(A_h, B_h, C_h_cpu, width, height);

    // Verification
    if (visual){
        printf("\n===== GPU Visual =====\n")
        for (int row = 0; row < height; row++) {
            printf("|")
            for (int col = 0; col < width; col++) {
                printf("%f|",C_h_gpu[row*width+col])
            }
            printf("\n")
        }
        printf("\n===== CPU Visual =====\n")
        for (int row = 0; row < height; row++) {
            printf("|")
            for (int col = 0; col < width; col++) {
                printf("%f|",C_h_cpu[row*width+col])
            }
            printf("\n")
        }
    }

    for (int i = 0; i < width * height; i++) {
        if (fabsf(C_h_gpu[i] - C_h_cpu[i]) > tolerence) {
            printf("[FAIL] test : width=%d, height=%d\n at index=%d:\n gpu:%f\n cpu:%f\n", width,
                height, i, C_h_gpu[i], C_h_cpu[i]);
            free(A_h);
            free(B_h);
            free(C_h_gpu);
            free(C_h_cpu);
            return 1;
        }
    }
    printf("[PASS] test : width=%d, height=%d\n", width, height);


    // Free Host memory
    free(A_h);
    free(B_h);
    free(C_h_gpu);
    free(C_h_cpu);
    return 0;
}

static int visual_check

int main(void) {
    int width[] = {1, 33, 2 << 7, 2 << 9};
    int height[] = {1, 33, 2 << 7, 2 << 9};
    int failures = 0;

    // test harness
    for (int i = 0; i < sizeof(width) / sizeof(width[0]); i++) {
        failures +=
            check(width[i], height[i],
                  1e-4);  // tolerance is ajusted because cpu and gpu dont round off the same way
    }

    // visual test
    int w = 33;
    int h = 33;
    check(width[i], height[i], 1e-4, 1);
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}