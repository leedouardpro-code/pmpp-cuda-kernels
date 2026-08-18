# ==== toolchain ====
NVCC  := nvcc
ARCH  ?= native         
NVCCFLAGS := -arch=$(ARCH) -O2 -I$(ROOT)/utils --ptxas-options=-v

# ==== shared headers ====
UTILS_HDRS := $(ROOT)/utils/cuda_check.h 

# ==== common rules ====
%.o: %.cu $(HDRS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# ==== common targets ====
clean:
	rm -f $(TARGETS) *.o

.PHONY: clean