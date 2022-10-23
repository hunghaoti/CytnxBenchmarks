#install root
CYTNX_ROOT=$(HOME)/libraries/Cytnx-dev
ITENSOR_ROOT:=$(HOME)/install_src/itensor
BENCHMARK_ROOT:=$(HOME)/install_src/benchmark

#cytnx flags
CYTNX_INC := $(shell python -c "exec(\"import sys\nsys.path.append(\'$(CYTNX_ROOT)\')\nimport cytnx\nprint(cytnx.__cpp_include__)\")")
CYTNX_LDFLAGS := $(shell python -c "exec(\"import sys\nsys.path.append(\'$(CYTNX_ROOT)\')\nimport cytnx\nprint(cytnx.__cpp_linkflags__)\")")
CYTNX_LIB := $(shell python -c "exec(\"import sys\nsys.path.append(\'$(CYTNX_ROOT)\')\nimport cytnx\nprint(cytnx.__cpp_lib__)\")")/libcytnx.a
CYTNX_CXXFLAGS := $(shell python -c "exec(\"import sys\nsys.path.append(\'$(CYTNX_ROOT)\')\nimport cytnx\nprint(cytnx.__cpp_flags__)\")")

#itensor flags
include $(ITENSOR_ROOT)/this_dir.mk
include $(ITENSOR_ROOT)/options.mk

#google benchmark
BENCHMARK_HEADER:=$(BENCHMARK_ROOT)/include
BENCHMARK_LIBDIR:=$(BENCHMARK_ROOT)/build/src

TARGETS:=basic_test.e
$(TARGETS): %.e:%.cpp
	$(CCCOM) $(CYTNX_CXXFLAGS) $(CCFLAGS) -I$(CYTNX_INC) $(ITENSOR_LIBS) $(ITENSOR_INCLUDEFLAGS) -I$(BENCHMARK_HEADER) $< -L$(ITENSOR_LIBDIR) $(ITENSOR_LIBFLAGS) $(CYTNX_LIB) -L$(BENCHMARK_LIBDIR) -lbenchmark -lrt -o $@

#clean
.phony: clean

clean:
	rm -f *.o *.e *.gcno *.gcda


