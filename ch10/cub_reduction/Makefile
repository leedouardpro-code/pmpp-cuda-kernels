ROOT := ../..
include $(ROOT)/common.mk

TARGETS := test_reduction bench_reduction
HDRS    := $(UTILS_HDRS) reduction.cuh

all: $(TARGETS)

test_reduction: test_reduction.cu reduction.o $(HDRS)
	$(NVCC) $(NVCCFLAGS) $(filter %.cu %.o,$^) -o $@

bench_reduction: bench_reduction.cu reduction.o $(HDRS)
	$(NVCC) $(NVCCFLAGS) $(filter %.cu %.o,$^) -o $@

test: test_reduction
	./test_reduction

bench: bench_reduction
	./bench_reduction

.PHONY: all test bench
