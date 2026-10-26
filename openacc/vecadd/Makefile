ROOT := ../..

# OpenACC build: NVIDIA HPC SDK compiler (nvc++), not nvcc. common.mk is not used.
#   -acc          : enable OpenACC directives
#   -gpu=cc120    : generate code for compute capability 12.0 (RTX 5060 Laptop, Blackwell)
#   -Minfo=accel  : print how each loop is mapped to the GPU (gang = block, vector = thread)
#   -cuda         : CUDA runtime API available (cudaEvent_* in utils/bench.h)
NVCPP ?= nvc++
GPU   ?= cc120
ACCFLAGS := -acc -gpu=$(GPU) -Minfo=accel -cuda -O2 -std=c++17 -I$(ROOT)/utils

TARGETS := test_vecadd_acc bench_vecadd_acc
HDRS    := vecadd_acc.h $(ROOT)/utils/bench.h $(ROOT)/utils/cuda_check.h

all: $(TARGETS)

vecadd_acc.o: vecadd_acc.cpp $(HDRS)
	$(NVCPP) $(ACCFLAGS) -c $< -o $@

test_vecadd_acc: test_vecadd_acc.cpp vecadd_acc.o $(HDRS)
	$(NVCPP) $(ACCFLAGS) $(filter %.cpp %.o,$^) -o $@

bench_vecadd_acc: bench_vecadd_acc.cpp vecadd_acc.o $(HDRS)
	$(NVCPP) $(ACCFLAGS) $(filter %.cpp %.o,$^) -o $@

test: test_vecadd_acc
	./test_vecadd_acc

bench: bench_vecadd_acc
	./bench_vecadd_acc

clean:
	rm -f $(TARGETS) *.o

.PHONY: all test bench clean
