ROOT := ../..
include $(ROOT)/common.mk

TARGETS := test_color2gray bench_color2gray
HDRS    := $(UTILS_HDRS) color2gray.cuh

all: $(TARGETS)

test_color2gray: test_color2gray.cu color2gray.o
	$(NVCC) $(NVCCFLAGS) $(filter %.cu %.o,$^) -o $@

bench_color2gray: bench_color2gray.cu color2gray.o
	$(NVCC) $(NVCCFLAGS) $(filter %.cu %.o,$^) -o $@

test: test_color2gray
	./test_color2gray

bench: bench_color2gray
	./bench_color2gray

.PHONY: all test bench clean
