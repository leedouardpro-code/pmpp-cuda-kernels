#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "vecadd.cuh"

static int check( int N, const float tolerence ) {
    const size_t size = sizeof(float) * N;
    
    // Host memory allocation
    float* A_h, *B_h, *C_h_gpu, *C_h_cpu;
    A_h = (float*)malloc(size);
    B_h = (float*)malloc(size);
    C_h_gpu = (float*)malloc(size);
    C_h_cpu = (float*)malloc(size);
    
    // Initialisation
    for(int i = 0; i < N; i++){
        A_h[i] = (float)i;
        B_h[i] = (float)i;
    }

    // Kernel execution
    vec_add_gpu(A_h, B_h, C_h_gpu, N);

    // CPU execution 
    vec_add_cpu(A_h, B_h, C_h_cpu, N);

    // Verification
    for (int i = 0; i < N; i++) {
        if (fabsf(C_h_gpu[i] - C_h_cpu[i]) > tolerence) {
            printf("[FAIL] test : N=%d\n at index=%d:\n gpu:%f\n cpu:%f\n", N, i, C_h_gpu[i], C_h_cpu[i]);
            free(A_h); free(B_h); free(C_h_gpu); free(C_h_cpu);
            return 1;
        }
    }
    printf("[PASS] test : N=%d\n", N);

    // Free Host memory
    free(A_h); free(B_h); free(C_h_gpu); free(C_h_cpu);
    return 0;
}

int main(void){
    int N[]= {1, 2<<7, 527, 2<<20};
    int failures = 0;

    // test harness
    for(int i=0; i < sizeof(N)/sizeof(N[0]); i++){
        failures += check(N[i], 1e-5);  
    }
    
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}