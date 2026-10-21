ROOT := ..
include $(ROOT)/common.mk

TARGETS := device_query clock_probe
HDRS    := $(UTILS_HDRS) device_query.h

all: $(TARGETS)

device_query: device_query_main.cu device_query.o $(HDRS)
	$(NVCC) $(NVCCFLAGS) $(filter %.cu %.o,$^) -o $@

clock_probe: clock_probe.cu $(HDRS)
	$(NVCC) $(NVCCFLAGS) $(filter %.cu %.o,$^) -o $@

query: device_query
	./device_query

probe: clock_probe
	./clock_probe

.PHONY: all query probe