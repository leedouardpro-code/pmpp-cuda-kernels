# ==== Config ====
NVCC := nvcc
ARCH ?= sm_120
NVCCFLAGS := -arch=$(ARCH) -O2 -I../../utils

COMMON_HDRS := ../../utils/cuda_check.h color2gray.cuh

# ==== Rules ====
all: test_color2gray bench_color2gray

color2gray.o: color2gray.cu $(COMMON_HDRS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

test_color2gray: test_color2gray.cu color2gray.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@

bench_color2gray: bench_color2gray.cu color2gray.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@

test: test_color2gray
	./test_color2gray

bench: bench_color2gray
	./bench_color2gray

clean:
	rm -f test_color2gray bench_color2gray *.o

.PHONY: all test bench clean