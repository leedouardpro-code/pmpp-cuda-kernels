ROOT := ..
include $(ROOT)/common.mk

TARGETS := test_matmul bench_matmul
HDRS    := $(UTILS_HDRS) matmul.cuh
LDLIBS  := -lcublas

all: $(TARGETS)

# test_matmul.cu and bench_matmul.cu are the same files as in ch3/ch5/ch6
test_matmul: test_matmul.cu matmul.o $(HDRS)
	$(NVCC) $(NVCCFLAGS) $(filter %.cu %.o,$^) -o $@ $(LDLIBS)

bench_matmul: bench_matmul.cu matmul.o $(HDRS)
	$(NVCC) $(NVCCFLAGS) $(filter %.cu %.o,$^) -o $@ $(LDLIBS)

test: test_matmul
	./test_matmul

bench: bench_matmul
	./bench_matmul

.PHONY: all test bench
