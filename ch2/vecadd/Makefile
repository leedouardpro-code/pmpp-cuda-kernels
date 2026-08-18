ROOT := ../..
include $(ROOT)/common.mk

TARGETS := test_vecadd bench_vecadd
HDRS    := $(UTILS_HDRS) vecadd.cuh

all: $(TARGETS)

test_vecadd: test_vecadd.cu vecadd.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@

bench_vecadd: bench_vecadd.cu vecadd.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@

test: test_vecadd
	./test_vecadd

bench: bench_vecadd
	./bench_vecadd

.PHONY: all test bench clean