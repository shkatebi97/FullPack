MKDIR=mkdir
TARGET_ISA=aarch64
RUY_LIB=ruy/bazel-bin/ruy
RUY_LIB_PROFILER=ruy/bazel-bin/ruy/profiler
RUY_INC=ruy
RUY_CCFLAGS := 	-Wall -Wextra -Wc++14-compat -Wundef -lpthread
RUY_LIB_LINK := -lcontext -lkernel_arm -lpack_arm -lfrontend -lprepacked_cache -lcontext_get_ctx -lctx -lallocator -lcpuinfo -lthread_pool -lprepare_packed_matrices -ltrmul -lblock_map -lapply_multiplier -lblocking_counter -ldenormal -lsystem_aligned_alloc -ltune -lwait
RUY_LIB_PROFILER_LINK := -linstrumentation
CPU_LIB=ruy/bazel-bin/external/cpuinfo
CPU_INC=ruy/third_party/cpuinfo/include
CPU_LIB_LINK := -lcpuinfo_impl -lclog

KERNELS_OBJS := kernels/Int8-Int4.o kernels/Int4-Int8.o kernels/Int4-Int4.o kernels/Int8-Ternary.o kernels/Ternary-Int8.o kernels/Ternary-Ternary.o kernels/Int8-Binary.o kernels/Binary-Int8.o kernels/Binary-Binary.o kernels/Binary-Binary-XOR.o kernels/Int8-Quaternary.o kernels/Int3-Int3.o kernels/ULPPACK.o kernels/ULPPACK/4x8-neon-multipack-type2.o kernels/ULPPACK/4x8-neon-multipack.o kernels/SelfDependent-kernels/W4A4.o kernels/SelfDependent-kernels/W4A8.o kernels/SelfDependent-kernels/W8A4.o kernels/SelfDependent-kernels/W2A2.o kernels/SelfDependent.o kernels/BarrelShiftMultiplier-kernels/W8A8.o kernels/BarrelShiftMultiplier.o

LDFLAGS :=

DEBUG ?= 0
ifeq ($(DEBUG), 1)
    CCFLAGS = -static -pthread -lstdc++ -g -march=armv8.2-a+fp16 -Wno-pointer-arith -Wno-narrowing -DIS_ARM -DIS_ARM64 -DTFLITE_BUILD -lm -flax-vector-conversions -fPIC
    SHARED_CCFLAGS = -pthread -lstdc++ -g -march=armv8.2-a+fp16 -Wno-pointer-arith -Wno-narrowing -DIS_ARM -DIS_ARM64 -DTFLITE_BUILD -lm -flax-vector-conversions -fPIC
else
    CCFLAGS = -static -pthread -lstdc++ -O3 -march=armv8.2-a+fp16 -Wno-pointer-arith -Wno-narrowing -DIS_ARM -DIS_ARM64 -DTFLITE_BUILD -lm -flax-vector-conversions -fPIC
    SHARED_CCFLAGS = -pthread -lstdc++ -O3 -march=armv8.2-a+fp16 -Wno-pointer-arith -Wno-narrowing -DIS_ARM -DIS_ARM64 -DTFLITE_BUILD -lm -flax-vector-conversions -fPIC
endif

ENABLE_RUY_PROFILER ?= 0

DISABLE_KERNELS_MEM_ACCESS ?= 0
ifeq ($(DISABLE_KERNELS_MEM_ACCESS), 1)
    KERNELS_MEM_ACCESS_FLAGS = -DDISABLE_KERNELS_MEM_ACCESS
else
    KERNELS_MEM_ACCESS_FLAGS = -UDISABLE_KERNELS_MEM_ACCESS
endif

CXX = /usr/bin/aarch64-linux-gnu-g++
CC = /usr/bin/aarch64-linux-gnu-gcc

all:												Build-Ruy \
													libfullpack.so \
													low_precision_fully_connected.o \
													ops-implementations/mul/LowPrecisionPacking.o \
													low_precision_fully_connected_test.o \
													test-16bit-2bit-packing \
													common/types.h \
													common/flags.h \
													common/half.hpp \
													common/asmutility.h \
													Makefile
	$(CXX) low_precision_fully_connected.o ops-implementations/mul/LowPrecisionPacking.o low_precision_fully_connected_test.o $(KERNELS_OBJS) -L$(RUY_LIB) $(RUY_LIB_LINK) -L$(RUY_LIB_PROFILER) $(RUY_LIB_PROFILER_LINK) -L$(CPU_LIB) $(CPU_LIB_LINK) $(RUY_CCFLAGS) $(CCFLAGS) ${LDFLAGS} -o low_precision_fully_connected_test

libfullpack.so:										low_precision_fully_connected.o \
													ops-implementations/mul/LowPrecisionPacking.o \
													common/types.h \
													common/flags.h \
													common/half.hpp \
													common/asmutility.h \
													Makefile
	$(CXX) -shared ops-implementations/mul/LowPrecisionPacking.o $(KERNELS_OBJS) $(SHARED_CCFLAGS) ${LDFLAGS} -o libfullpack.so

Build-Ruy:					
	$(MAKE) -C ruy ENABLE_RUY_PROFILER=$(ENABLE_RUY_PROFILER) DEBUG=$(DEBUG) DISABLE_KERNELS_MEM_ACCESS=$(DISABLE_KERNELS_MEM_ACCESS)

############################# Kernels Start #############################

# kernels/Int8-Int8.o:								kernels/Int8-Int8.cc \
# 													common/types.h \
# 													common/flags.h \
# 													low_precision_fully_connected.h \
# 													Makefile
# 	$(CXX) kernels/Int8-Int8.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) ${LDFLAGS} -o kernels/Int8-Int8.o -c

kernels/Int8-Int4.o:								kernels/Int8-Int4.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels/Int8-Int4.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) ${LDFLAGS} -o kernels/Int8-Int4.o -c

kernels/Int4-Int8.o:								kernels/Int4-Int8.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels/Int4-Int8.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) ${LDFLAGS} -o kernels/Int4-Int8.o -c

kernels/Int4-Int4.o:								kernels/Int4-Int4.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels/Int4-Int4.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) ${LDFLAGS} -o kernels/Int4-Int4.o -c

kernels/Int8-Ternary.o:								kernels/Int8-Ternary.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels/Int8-Ternary.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) ${LDFLAGS} -o kernels/Int8-Ternary.o -c

kernels/Ternary-Int8.o:								kernels/Ternary-Int8.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels/Ternary-Int8.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) ${LDFLAGS} -o kernels/Ternary-Int8.o -c

kernels/Ternary-Ternary.o:							kernels/Ternary-Ternary.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels/Ternary-Ternary.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) ${LDFLAGS} -o kernels/Ternary-Ternary.o -c

kernels/Int8-Binary.o:								kernels/Int8-Binary.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels/Int8-Binary.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) ${LDFLAGS} -o kernels/Int8-Binary.o -c

kernels/Binary-Int8.o:								kernels/Binary-Int8.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels/Binary-Int8.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) ${LDFLAGS} -o kernels/Binary-Int8.o -c

kernels/Binary-Binary.o:							kernels/Binary-Binary.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels/Binary-Binary.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) ${LDFLAGS} -o kernels/Binary-Binary.o -c

kernels/Binary-Binary-XOR.o:						kernels/Binary-Binary-XOR.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels/Binary-Binary-XOR.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) ${LDFLAGS} -o kernels/Binary-Binary-XOR.o -c

kernels/Int8-Quaternary.o:							kernels/Int8-Quaternary.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels/Int8-Quaternary.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) ${LDFLAGS} -o kernels/Int8-Quaternary.o -c

kernels/Int3-Int3.o:								kernels/Int3-Int3.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels/Int3-Int3.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) ${LDFLAGS} -o kernels/Int3-Int3.o -c

kernels/ULPPACK.o:									kernels/ULPPACK.cc \
													common/types.h \
													common/flags.h \
													kernels/ULPPACK/ULPPACK.h \
													kernels/ULPPACK/test.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels/ULPPACK.cc -flax-vector-conversions -lpthread -Wno-psabi -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) ${LDFLAGS} -o kernels/ULPPACK.o -c

kernels/ULPPACK/4x8-neon-multipack-type2.o:			kernels/ULPPACK.cc \
													common/types.h \
													common/flags.h \
													kernels/ULPPACK/ULPPACK.h \
													kernels/ULPPACK/test.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels/ULPPACK/4x8-neon-multipack-type2.cpp -flax-vector-conversions -lpthread -Wno-psabi -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) ${LDFLAGS} -o kernels/ULPPACK/4x8-neon-multipack-type2.o -c

kernels/ULPPACK/4x8-neon-multipack.o:				kernels/ULPPACK.cc \
													common/types.h \
													common/flags.h \
													kernels/ULPPACK/ULPPACK.h \
													kernels/ULPPACK/test.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels/ULPPACK/4x8-neon-multipack.cpp -flax-vector-conversions -lpthread -Wno-psabi -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) ${LDFLAGS} -o kernels/ULPPACK/4x8-neon-multipack.o -c

######  SelfDependent Kernels Start  ######

kernels/SelfDependent-kernels/W4A4.o:				kernels/SelfDependent-kernels/W4A4.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels/SelfDependent-kernels/W4A4.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) ${LDFLAGS} -o kernels/SelfDependent-kernels/W4A4.o -c

kernels/SelfDependent-kernels/W4A8.o:				kernels/SelfDependent-kernels/W4A8.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels/SelfDependent-kernels/W4A8.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) ${LDFLAGS} -o kernels/SelfDependent-kernels/W4A8.o -c

kernels/SelfDependent-kernels/W8A4.o:				kernels/SelfDependent-kernels/W8A4.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels/SelfDependent-kernels/W8A4.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) ${LDFLAGS} -o kernels/SelfDependent-kernels/W8A4.o -c

kernels/SelfDependent-kernels/W2A2.o:				kernels/SelfDependent-kernels/W2A2.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels/SelfDependent-kernels/W2A2.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) ${LDFLAGS} -o kernels/SelfDependent-kernels/W2A2.o -c

######  SelfDependent Kernels End  ######

kernels/SelfDependent.o:							kernels/SelfDependent.cc \
													common/types.h \
													common/flags.h \
													kernels/SelfDependent-kernels/W4A4.o \
													kernels/SelfDependent-kernels/W4A8.o \
													kernels/SelfDependent-kernels/W8A4.o \
													kernels/SelfDependent-kernels/W2A2.o \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels/SelfDependent.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) ${LDFLAGS} -o kernels/SelfDependent.o -c

######  BarrelShiftMultiplier Kernels Start  ######

kernels/BarrelShiftMultiplier-kernels/W8A8.o:		kernels/BarrelShiftMultiplier-kernels/W8A8.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels/BarrelShiftMultiplier-kernels/W8A8.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) ${LDFLAGS} -o kernels/BarrelShiftMultiplier-kernels/W8A8.o -c

######  BarrelShiftMultiplier Kernels End  ######

kernels/BarrelShiftMultiplier.o:					kernels/BarrelShiftMultiplier.cc \
													common/types.h \
													common/flags.h \
													kernels/BarrelShiftMultiplier-kernels/W8A8.o \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels/BarrelShiftMultiplier.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) ${LDFLAGS} -o kernels/BarrelShiftMultiplier.o -c

#############################  Kernels End  #############################

low_precision_fully_connected.o:					low_precision_fully_connected.cc \
													common/types.h \
													common/flags.h \
													$(KERNELS_OBJS) \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) low_precision_fully_connected.cc $(CCFLAGS) ${LDFLAGS} -o low_precision_fully_connected.o -c

ops-implementations/mul/LowPrecisionPacking.o:		ops-implementations/mul/LowPrecisionPacking.cc \
													common/types.h \
													common/flags.h \
													ops-implementations/mul/LowPrecisionPacking.h \
													Makefile
	$(CXX) ops-implementations/mul/LowPrecisionPacking.cc $(CCFLAGS) ${LDFLAGS} -o ops-implementations/mul/LowPrecisionPacking.o -c

low_precision_fully_connected_test.o:				low_precision_fully_connected_test.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected_benchmark.h \
													Makefile
	$(CXX) low_precision_fully_connected_test.cc  $(KERNELS_MEM_ACCESS_FLAGS) -I$(RUY_INC) $(CCFLAGS) ${LDFLAGS} -o low_precision_fully_connected_test.o -c

test-16bit-2bit-packing:							test-16bit-2bit-packing.o \
													Makefile
	$(CXX) test-16bit-2bit-packing.o $(CCFLAGS) ${LDFLAGS} -o test-16bit-2bit-packing

test-16bit-2bit-packing.o:							test-16bit-2bit-packing.cc \
													common/types.h \
													common/flags.h \
													Makefile
	$(CXX) test-16bit-2bit-packing.cc $(CCFLAGS) ${LDFLAGS} -o test-16bit-2bit-packing.o -c

clean:
	rm -f \
		low_precision_fully_connected_test.o \
		low_precision_fully_connected.o \
		ops-implementations/mul/LowPrecisionPacking.o \
		low_precision_fully_connected_test \
		$(KERNELS_OBJS)
	$(MAKE) -C ruy DEBUG=$(DEBUG) clean

