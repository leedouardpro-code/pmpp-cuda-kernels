# ==== Config ====
NVCC := nvcc
ARCH ?= sm_120
NVCCFLAGS := -arch=$(ARCH) -O2 -I../utils

COMMON_HDRS := ../utils/cuda_check.h vecadd.cuh

# ==== Rules ====
all: test_vecadd bench_vecadd

vecadd.o: vecadd.cu $(COMMON_HDRS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

test_vecadd: test_vecadd.cu vecadd.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@

bench_vecadd: bench_vecadd.cu vecadd.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@

test: test_vecadd
	./test_vecadd

bench: bench_vecadd
	./bench_vecadd

clean:
	rm -f test_vecadd bench_vecadd *.o

.PHONY: all test bench clean