ROOT := ../..
include $(ROOT)/common.mk

TARGETS := test_matmul bench_matmul
HDRS    := $(UTILS_HDRS) matmul.cuh

all: $(TARGETS)

test_matmul: test_matmul.cu matmul.o $(HDRS)
	$(NVCC) $(NVCCFLAGS) $(filter %.cu %.o,$^) -o $@

bench_matmul: bench_matmul.cu matmul.o $(HDRS)
	$(NVCC) $(NVCCFLAGS) $(filter %.cu %.o,$^) -o $@

test: test_matmul
	./test_matmul

bench: bench_matmul
	./bench_matmul

.PHONY: all test bench