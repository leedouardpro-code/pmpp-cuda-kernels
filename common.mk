# ==== toolchain ====
NVCC  := nvcc
ARCH  ?= native
# -std=c++17 : lambdas and STL in bench.h / tests
# -lineinfo  : source line numbers in compute-sanitizer and Nsight Compute (no runtime cost)
# EXTRA: per-build defines, e.g. make EXTRA="-DTILE_WIDTH=16" (used by scripts/sweep_matmul.sh)
EXTRA ?=
NVCCFLAGS := -arch=$(ARCH) -O2 -std=c++17 -lineinfo -I$(ROOT)/utils --ptxas-options=-v $(EXTRA)

# ==== shared headers ====
UTILS_HDRS := $(ROOT)/utils/cuda_check.h $(ROOT)/utils/bench.h

# ==== common rules ====
# Secondary expansion: $$(HDRS) is expanded when the rule is used,
# i.e. after each chapter Makefile has defined HDRS (it is included before).
.SECONDEXPANSION:
%.o: %.cu $$(HDRS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# ==== common targets ====
clean:
	rm -f $(TARGETS) *.o

.PHONY: clean