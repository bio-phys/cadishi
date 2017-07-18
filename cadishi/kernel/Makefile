# Makefile to compile the CPU (pydh) and GPU (cudh) kernels as
# a standalone C library, independent of Python.

# flags common to GCC and NVCC
FLAGS=-DBUILD_C_LIBRARY -I./include/

# gcc flags
GCC_FLAGS=-g -fPIC -O3 -ffast-math -march=native -fopenmp -fopt-info-vec

# CUDA flags
# comment/uncomment/extend desired target architectures
NVCC_ARCHS+=--generate-code arch=compute_35,code=sm_35
NVCC_ARCHS+=--generate-code arch=compute_37,code=sm_37
NVCC_ARCHS+=--generate-code arch=compute_50,code=sm_50
NVCC_ARCHS+=--generate-code arch=compute_52,code=sm_52
NVCC_ARCHS+=--generate-code arch=compute_53,code=sm_53
# NVCC_ARCHS+=--generate-code arch=compute_60,code=sm_60
# NVCC_ARCHS+=--generate-code arch=compute_61,code=sm_61
NVCC_FLAGS=-O3 -use_fast_math


.PHONY: message
message:
	@echo ""
	@echo "Compile Cadishi CPU and GPU kernels as standalone C libraries:"
	@echo ""
	@echo "   $ make libpydh  # build CPU kernel"
	@echo "   $ make libcudh  # build GPU kernel"
	@echo "   $ make unit_tests  # build and run unit tests"
	@echo "   $ make all"
	@echo "   $ make clean"
	@echo ""


.PHONY: clean
clean:
	rm -f lib/libcudh.so lib/libpydh.so unit_tests unit_tests.o
	rm -rf lib

all: libpydh libcudh

libpydh: lib/libpydh.so
lib/libpydh.so: c_pydh.cc include/pydh.h
	@mkdir -p lib
	g++ $(FLAGS) $(GCC_FLAGS) -shared -o $@ $<
	@echo
	@echo "Compilation of PYDH CPU distance histogram kernel successful!"
	@echo "Use the following include file and shared object to compile and link your application:"
	@echo
	@ls -l $@ include/pydh.h
	@echo


libcudh: lib/libcudh.so
lib/libcudh.so: c_cudh.cu include/cudh.h
	@mkdir -p lib
	nvcc $(FLAGS) $(NVCC_FLAGS) $(NVCC_ARCHS) --compiler-options "$(GCC_FLAGS)" --linker-options "$(GCC_FLAGS)" -shared -o $@ $<
	@echo
	@echo "Compilation of CUDH GPU distance histogram kernel successful!"
	@echo "Use the following include file and shared object to compile and link your application:"
	@echo
	@ls -l $@ include/cudh.h
	@echo

unit_tests: unit_tests.cc libpydh
	g++ $(FLAGS) $(GCC_FLAGS) -c $<
	g++ -L$(realpath ./lib/) -Wl,-rpath,$(realpath ./lib/) -lpydh $(@).o -o $@
	./unit_tests
