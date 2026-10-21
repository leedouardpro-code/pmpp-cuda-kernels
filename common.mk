# ==== toolchain ====
NVCC  := nvcc
ARCH  ?= native         
NVCCFLAGS := -arch=$(ARCH) -O2 -I$(ROOT)/utils --ptxas-options=-v

# ==== shared headers ====
UTILS_HDRS := $(ROOT)/utils/cuda_check.h 

# ==== common rules ====
# Secondary expansion: $$(HDRS) is expanded when the rule is used,
# i.e. after each chapter Makefile has defined HDRS.
.SECONDEXPANSION:
%.o: %.cu $$(HDRS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# ==== common targets ====
clean:
	rm -f $(TARGETS) *.o

.PHONY: clean