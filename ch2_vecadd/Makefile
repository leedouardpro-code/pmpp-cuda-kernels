# ==== Config ====
NVCC      := nvcc
ARCH      ?= sm_75 #sm_87 for Orin
NVCCFLAGS := -arch=$(ARCH) -O2 -I../utils

COMMON_HDRS := ../utils/cuda_check.h vecadd.cuh

# ==== Rules ====
all: vecadd test_vecadd

vecadd.o: vecadd.cu $(COMMON_HDRS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

vecadd: main.cu vecadd.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@

test_vecadd: test_vecadd.cu vecadd.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@

run: vecadd
	./vecadd

test: test_vecadd
	./test_vecadd

clean:
	rm -f vecadd test_vecadd *.o

.PHONY: all run test clean