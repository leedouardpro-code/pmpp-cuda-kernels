#include <cuda_runtime.h>
#include "cuda_check.h"
#include "device_query.h"

int main(void) {
    int n;
    CUDA_CHECK(cudaGetDeviceCount(&n));
    for (int d = 0; d < n; d++) print_device_props(d);
    return 0;
}