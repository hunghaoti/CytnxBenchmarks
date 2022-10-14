#install root
CYTNX_ROOT=/home/hunghaoti/install/CYTNX077_dev
ITENSOR_ROOT:=/home/hunghaoti/install/itensor

CC := g++

#cytnx flags
CYTNX_INC := $(shell python -c "exec(\"import sys\nsys.path.append(\'$(CYTNX_ROOT)\')\nimport cytnx\nprint(cytnx.__cpp_include__)\")")
CYTNX_LDFLAGS := $(shell python -c "exec(\"import sys\nsys.path.append(\'$(CYTNX_ROOT)\')\nimport cytnx\nprint(cytnx.__cpp_linkflags__)\")")
CYTNX_LIB := $(shell python -c "exec(\"import sys\nsys.path.append(\'$(CYTNX_ROOT)\')\nimport cytnx\nprint(cytnx.__cpp_lib__)\")")/libcytnx.a
CYTNX_CXXFLAGS := $(shell python -c "exec(\"import sys\nsys.path.append(\'$(CYTNX_ROOT)\')\nimport cytnx\nprint(cytnx.__cpp_flags__)\")")

#itensor flags
ITENSOR_CXXFLAGS:=-std=c++17
ITENSOR_HEADER:=$(ITENSOR_ROOT)
ITENSOR_LIBDIR:=$(ITENSOR_ROOT)/lib

TARGETS:=basic_test.e
$(TARGETS): %.e:%.cpp
	$(CC) $(ITENSOR_CXXFLAGS) $(CYTNX_CXXFLAGS) -I$(ITENSOR_HEADER) -I${CYTNX_INC}  $< -L$(ITENSOR_LIBDIR) ${CYTNX_LIB} ${CYTNX_LINK} -litensor -pthread -lbenchmark -lopenblas -o $@

#clean
.phony: clean

clean:
	rm -f *.o *.e *.gcno *.gcda


