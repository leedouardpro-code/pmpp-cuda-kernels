# ==== Config ====
NVCC := nvcc
ARCH ?= sm_75
NVCCFLAGS := -arch=$(ARCH) -O2 -I../../utils

COMMON_HDRS := ../../utils/cuda_check.h matmul.cuh

# ==== Rules ====
all: test_matmul bench_matmul

matmul.o: matmul.cu $(COMMON_HDRS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

test_matmul: test_matmul.cu matmul.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@

bench_matmul: bench_matmul.cu matmul.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@

test: test_matmul
	./test_matmul

bench: bench_matmul
	./bench_matmul

clean:
	rm -f test_matmul bench_matmul *.o

.PHONY: all test bench clean