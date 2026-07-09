#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vecadd.cuh"

static int check(float* C, float* expected, int N, const char* name) {
    for (int i = 0; i < N; i++) {
        if (C[i] != expected[i]) {
            printf("[FAIL] %s at i=%d: got %f, expected %f\n",
                   name, i, C[i], expected[i]);
            return 1;
        }
    }
    printf("[PASS] %s\n", name);
    return 0;
}

int main(void){
    int N = 1000;

    // Host memory allocation
    float* A_h, *B_h, *C_h;
    A_h = (float*)malloc(sizeof(float)*N);
    B_h = (float*)malloc(sizeof(float)*N);
    C_h = (float*)malloc(sizeof(float)*N);

    // Initialisation
    for(int i=0; i<N; i++){
        A_h[i]=1;
        B_h[i]=1;
    }

    // Kernel execution
    vec_add(A_h, B_h, C_h, N);

    // Free Host memory
    free(A_h);
    free(B_h);
    free(C_h);
}