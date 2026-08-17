ROOT := ..
include $(ROOT)/common.mk

TARGETS := device_query
HDRS    := cuda_check.h device_query.h

all: $(TARGETS)

device_query: device_query_main.cu device_query.o $(HDRS)
	$(NVCC) $(NVCCFLAGS) $(filter %.cu %.o,$^) -o $@

query: device_query
	./device_query

.PHONY: all query