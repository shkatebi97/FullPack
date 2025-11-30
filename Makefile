MKDIR=mkdir
TARGET_ISA ?= aarch64
RUY_LIB=-Lruy/bazel-bin/ruy
RUY_LIB_PROFILER=-Lruy/bazel-bin/ruy/profiler
RUY_INC=-Iruy
RUY_CCFLAGS := 	-Wall -Wextra -Wc++14-compat -Wundef -lpthread
RUY_LIB_PROFILER_LINK := -linstrumentation
CPU_LIB=-Lruy/bazel-bin/external/cpuinfo
CPU_INC=-Iruy/third_party/cpuinfo/include
CPU_LIB_LINK := -lcpuinfo_impl -lclog
ifeq ($(TARGET_ISA), aarch64)
	RUY_LIB_LINK := -lcontext -lkernel_arm -lpack_arm -lfrontend -lprepacked_cache -lcontext_get_ctx -lctx -lallocator -lcpuinfo -lthread_pool -lprepare_packed_matrices -ltrmul -lblock_map -lapply_multiplier -lblocking_counter -ldenormal -lsystem_aligned_alloc -ltune -lwait
	ARCH_MODIFIER_FLAGS := -march=armv8.2-a+fp16
	ARCH_DEFINES := -DIS_ARM -DIS_ARM64
	CXX := /usr/bin/aarch64-linux-gnu-g++
	CC := /usr/bin/aarch64-linux-gnu-gcc
	BUILD_DIR?=build-aarch64
else ifeq ($(TARGET_ISA), x86_64-avx512)
	RUY_LIB_LINK := -lallocator -lapply_multiplier -lcontext -lcontext_get_ctx -lcpuinfo -lctx -lfrontend -lhave_built_path_for_avx2_fma -lhave_built_path_for_avx512 -lhave_built_path_for_avx -lkernel_arm -lkernel_avx2_fma -lkernel_avx512 -lkernel_avx -lpack_arm -lpack_avx2_fma -lpack_avx512 -lpack_avx -lprepacked_cache -lprepare_packed_matrices -lsystem_aligned_alloc -lthread_pool -lblocking_counter -ltrmul -lblock_map -ldenormal -ltune -lwait
	ARCH_MODIFIER_FLAGS := -mavx512f -mavx512dq -mavx512cd -mavx512bw -mavx512vl -mavx512vbmi2
	ARCH_DEFINES := -DIS_X86 -DIS_X86_64 -DHAS_AVX512
	CXX := /usr/bin/g++
	CC := /usr/bin/gcc
	BUILD_DIR?=build-x86_64-avx512
else ifeq ($(TARGET_ISA), x86_64-avx2)
	RUY_LIB_LINK := -lallocator -lapply_multiplier -lcontext -lcontext_get_ctx -lcpuinfo -lctx -lfrontend -lhave_built_path_for_avx2_fma -lhave_built_path_for_avx512 -lhave_built_path_for_avx -lkernel_arm -lkernel_avx2_fma -lkernel_avx512 -lkernel_avx -lpack_arm -lpack_avx2_fma -lpack_avx512 -lpack_avx -lprepacked_cache -lprepare_packed_matrices -lsystem_aligned_alloc -lthread_pool -lblocking_counter -ltrmul -lblock_map -ldenormal -ltune -lwait
	ARCH_MODIFIER_FLAGS := -mavx2 -mfma
	ARCH_DEFINES := -DIS_X86 -DIS_X86_64 -DHAS_AVX2
	CXX := /usr/bin/g++
	CC := /usr/bin/gcc
	BUILD_DIR?=build-x86_64-avx2
else ifeq ($(TARGET_ISA), x86_64-avx)
	RUY_LIB_LINK := -lallocator -lapply_multiplier -lcontext -lcontext_get_ctx -lcpuinfo -lctx -lfrontend -lhave_built_path_for_avx2_fma -lhave_built_path_for_avx512 -lhave_built_path_for_avx -lkernel_arm -lkernel_avx2_fma -lkernel_avx512 -lkernel_avx -lpack_arm -lpack_avx2_fma -lpack_avx512 -lpack_avx -lprepacked_cache -lprepare_packed_matrices -lsystem_aligned_alloc -lthread_pool -lblocking_counter -ltrmul -lblock_map -ldenormal -ltune -lwait
	ARCH_MODIFIER_FLAGS := -mavx
	ARCH_DEFINES := -DIS_X86 -DIS_X86_64 -DHAS_AVX
	CXX := /usr/bin/g++
	CC := /usr/bin/gcc
	BUILD_DIR?=build-x86_64-avx
else ifeq ($(TARGET_ISA), riscv64-vector-internal)
	RUY_LIB_LINK:=
	RUY_LIB:=
	RUY_LIB_PROFILER:=
	RUY_INC:=
	RUY_CCFLAGS:=
	RUY_LIB_PROFILER_LINK:=
	CPU_LIB:=
	CPU_INC:=
	CPU_LIB_LINK:=
	ARCH_MODIFIER_FLAGS :=
	SHARED_CCFLAGS = -lstdc++ -std=c++20 $(OPTIMIZATION_FLAG) -Wno-pointer-arith -Wno-narrowing $(ARCH_DEFINES) -DTFLITE_BUILD -lm -flax-vector-conversions -fPIC
	ARCH_DEFINES := -DIS_RISCV -DIS_RISCV64 -DHAS_VEXTENSION
	TOOLCHAIN_INSTALL_PREFIX?=/riscv/_install
# 	Can be riscv64-unknown-linux-gnu or riscv64-unknown-elf
	ifeq ($(TARGET_TYPE), linux)
		TARGET:=riscv64-unknown-linux-gnu
	else ifeq ($(TARGET_TYPE), newlib)
		TARGET:=riscv64-unknown-elf
	endif
	CXX:=$(TOOLCHAIN_INSTALL_PREFIX)/bin/$(TARGET)-g++
	CC:=$(TOOLCHAIN_INSTALL_PREFIX)/bin/$(TARGET)-gcc
	SYSROOT?=$(INSTALL_PREFIX)/$(TARGET)
	SUB_PARAM?=
  BUILD_DIR?=build-rvv
else ifeq ($(TARGET_ISA), riscv64-vector)
# 	Can be gem5/llvm-rv64gcv-newlib or gem5/llvm-rv64gcv-linux
	TARGET_TYPE?=linux
	ifeq ($(TARGET_TYPE), linux)
		RISCV_BUILDER_IMAGE:=gem5/llvm-rv64gcv-linux
	else ifeq ($(TARGET_TYPE), newlib)
		RISCV_BUILDER_IMAGE:=gem5/llvm-rv64gcv-newlib
	endif
endif

KERNELS_OBJS := $(BUILD_DIR)/kernels-impl/Int8-Int4.o $(BUILD_DIR)/kernels-impl/Int4-Int8.o $(BUILD_DIR)/kernels-impl/Int4-Int4.o $(BUILD_DIR)/kernels-impl/Int8-Ternary.o $(BUILD_DIR)/kernels-impl/Ternary-Int8.o $(BUILD_DIR)/kernels-impl/Ternary-Ternary.o $(BUILD_DIR)/kernels-impl/Int8-Binary.o $(BUILD_DIR)/kernels-impl/Binary-Int8.o $(BUILD_DIR)/kernels-impl/Binary-Binary.o $(BUILD_DIR)/kernels-impl/Binary-Binary-XOR.o $(BUILD_DIR)/kernels-impl/Int8-Quaternary.o $(BUILD_DIR)/kernels-impl/Int3-Int3.o $(BUILD_DIR)/kernels-impl/ULPPACK.o $(BUILD_DIR)/kernels-impl/ULPPACK/4x8-neon-multipack-type2.o $(BUILD_DIR)/kernels-impl/ULPPACK/4x8-neon-multipack.o $(BUILD_DIR)/kernels-impl/SelfDependent-kernels/W4A4.o $(BUILD_DIR)/kernels-impl/SelfDependent-kernels/W4A8.o $(BUILD_DIR)/kernels-impl/SelfDependent-kernels/W8A4.o $(BUILD_DIR)/kernels-impl/SelfDependent-kernels/W2A2.o $(BUILD_DIR)/kernels-impl/SelfDependent.o $(BUILD_DIR)/kernels-impl/BarrelShiftMultiplier-kernels/W8A8.o $(BUILD_DIR)/kernels-impl/BarrelShiftMultiplier-kernels/W4A4.o $(BUILD_DIR)/kernels-impl/BarrelShiftMultiplier.o $(BUILD_DIR)/kernels-impl/Float32.o

LDFLAGS :=

DEBUG ?= 0
ifeq ($(DEBUG), 1)
    OPTIMIZATION_FLAG = -g
else
    OPTIMIZATION_FLAG = -O3
endif

SHARED_CCFLAGS ?= -pthread -lstdc++ $(OPTIMIZATION_FLAG) -Wno-pointer-arith -Wno-narrowing $(ARCH_DEFINES) -DTFLITE_BUILD -lm -flax-vector-conversions -fPIC
CCFLAGS = -static $(SHARED_CCFLAGS)

ENABLE_RUY_PROFILER ?= 0

DISABLE_KERNELS_MEM_ACCESS ?= 0
ifeq ($(DISABLE_KERNELS_MEM_ACCESS), 1)
    KERNELS_MEM_ACCESS_FLAGS = -DDISABLE_KERNELS_MEM_ACCESS
else
    KERNELS_MEM_ACCESS_FLAGS = -UDISABLE_KERNELS_MEM_ACCESS
endif

ifeq ($(TARGET_ISA), riscv64-vector)
all: 												docker-build

docker-build:
	docker run --rm -v $(PWD):/workspace -w /workspace $(RISCV_BUILDER_IMAGE) /bin/bash -c "make TARGET_ISA=riscv64-vector-internal DEBUG=$(DEBUG) DISABLE_KERNELS_MEM_ACCESS=$(DISABLE_KERNELS_MEM_ACCESS) TARGET_TYPE=$(TARGET_TYPE) $(SUB_PARAM)"
else
all: 												link
endif

ifeq ($(TARGET_ISA), aarch64 x86_64-avx512 x86_64-avx2 x86_64-avx)
link:												Build-Ruy
endif

link:												$(BUILD_DIR)/libfullpack.so \
													$(BUILD_DIR)/low_precision_fully_connected.o \
													$(BUILD_DIR)/ops-implementations/mul/LowPrecisionPacking.o \
													$(BUILD_DIR)/low_precision_fully_connected_test.o
													common/types.h \
													common/flags.h \
													common/half.hpp \
													common/asmutility.h \
													Makefile
	$(CXX) $(BUILD_DIR)/low_precision_fully_connected.o $(BUILD_DIR)/ops-implementations/mul/LowPrecisionPacking.o $(BUILD_DIR)/low_precision_fully_connected_test.o $(KERNELS_OBJS) $(RUY_LIB) $(RUY_LIB_LINK) $(RUY_LIB_PROFILER) $(RUY_LIB_PROFILER_LINK) $(CPU_LIB) $(CPU_LIB_LINK) $(RUY_CCFLAGS) $(CCFLAGS) ${LDFLAGS} -o $(BUILD_DIR)/low_precision_fully_connected_test

$(BUILD_DIR)/libfullpack.so:						Create-Build-Directory \
													$(BUILD_DIR)/low_precision_fully_connected.o \
													$(BUILD_DIR)/ops-implementations/mul/LowPrecisionPacking.o \
													common/types.h \
													common/flags.h \
													common/half.hpp \
													common/asmutility.h \
													Makefile
	$(CXX) -shared $(BUILD_DIR)/ops-implementations/mul/LowPrecisionPacking.o $(KERNELS_OBJS) $(SHARED_CCFLAGS) ${LDFLAGS} -o $(BUILD_DIR)/libfullpack.so

Create-Build-Directory:
	mkdir -p $(BUILD_DIR) \
			 $(BUILD_DIR)/kernels-impl \
			 $(BUILD_DIR)/kernels-impl/ULPPACK \
			 $(BUILD_DIR)/kernels-impl/SelfDependent-kernels \
			 $(BUILD_DIR)/kernels-impl/BarrelShiftMultiplier-kernels \
			 $(BUILD_DIR)/kernels-impl/Float32-kernels \
			 $(BUILD_DIR)/ops-implementations/mul

ifneq ($(TARGET_ISA), riscv-vector riscv64-vector-internal)
Build-Ruy:					
	$(MAKE) -C ruy ENABLE_RUY_PROFILER=$(ENABLE_RUY_PROFILER) DEBUG=$(DEBUG) DISABLE_KERNELS_MEM_ACCESS=$(DISABLE_KERNELS_MEM_ACCESS) TARGET_ISA=$(TARGET_ISA)
else
Build-Ruy:
	@echo "Ruy does not support RISC-V Vector target ISA."
	@false
endif

############################# Kernels Start #############################

# $(BUILD_DIR)/kernels-impl/Int8-Int8.o:								kernels-impl/Int8-Int8.cc \
# 													common/types.h \
# 													common/flags.h \
# 													low_precision_fully_connected.h \
# 													Makefile
# 	$(CXX) kernels-impl/Int8-Int8.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) $(ARCH_MODIFIER_FLAGS) ${LDFLAGS} -o $(BUILD_DIR)/kernels-impl/Int8-Int8.o -c

$(BUILD_DIR)/kernels-impl/Int8-Int4.o:								kernels-impl/Int8-Int4.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels-impl/Int8-Int4.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) $(ARCH_MODIFIER_FLAGS) ${LDFLAGS} -o $(BUILD_DIR)/kernels-impl/Int8-Int4.o -c

$(BUILD_DIR)/kernels-impl/Int4-Int8.o:								kernels-impl/Int4-Int8.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels-impl/Int4-Int8.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) $(ARCH_MODIFIER_FLAGS) ${LDFLAGS} -o $(BUILD_DIR)/kernels-impl/Int4-Int8.o -c

$(BUILD_DIR)/kernels-impl/Int4-Int4.o:								kernels-impl/Int4-Int4.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels-impl/Int4-Int4.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) $(ARCH_MODIFIER_FLAGS) ${LDFLAGS} -o $(BUILD_DIR)/kernels-impl/Int4-Int4.o -c

$(BUILD_DIR)/kernels-impl/Int8-Ternary.o:								kernels-impl/Int8-Ternary.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels-impl/Int8-Ternary.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) $(ARCH_MODIFIER_FLAGS) ${LDFLAGS} -o $(BUILD_DIR)/kernels-impl/Int8-Ternary.o -c

$(BUILD_DIR)/kernels-impl/Ternary-Int8.o:								kernels-impl/Ternary-Int8.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels-impl/Ternary-Int8.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) $(ARCH_MODIFIER_FLAGS) ${LDFLAGS} -o $(BUILD_DIR)/kernels-impl/Ternary-Int8.o -c

$(BUILD_DIR)/kernels-impl/Ternary-Ternary.o:							kernels-impl/Ternary-Ternary.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels-impl/Ternary-Ternary.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) $(ARCH_MODIFIER_FLAGS) ${LDFLAGS} -o $(BUILD_DIR)/kernels-impl/Ternary-Ternary.o -c

$(BUILD_DIR)/kernels-impl/Int8-Binary.o:								kernels-impl/Int8-Binary.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels-impl/Int8-Binary.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) $(ARCH_MODIFIER_FLAGS) ${LDFLAGS} -o $(BUILD_DIR)/kernels-impl/Int8-Binary.o -c

$(BUILD_DIR)/kernels-impl/Binary-Int8.o:								kernels-impl/Binary-Int8.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels-impl/Binary-Int8.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) $(ARCH_MODIFIER_FLAGS) ${LDFLAGS} -o $(BUILD_DIR)/kernels-impl/Binary-Int8.o -c

$(BUILD_DIR)/kernels-impl/Binary-Binary.o:							kernels-impl/Binary-Binary.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels-impl/Binary-Binary.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) $(ARCH_MODIFIER_FLAGS) ${LDFLAGS} -o $(BUILD_DIR)/kernels-impl/Binary-Binary.o -c

$(BUILD_DIR)/kernels-impl/Binary-Binary-XOR.o:						kernels-impl/Binary-Binary-XOR.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels-impl/Binary-Binary-XOR.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) $(ARCH_MODIFIER_FLAGS) ${LDFLAGS} -o $(BUILD_DIR)/kernels-impl/Binary-Binary-XOR.o -c

$(BUILD_DIR)/kernels-impl/Int8-Quaternary.o:							kernels-impl/Int8-Quaternary.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels-impl/Int8-Quaternary.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) $(ARCH_MODIFIER_FLAGS) ${LDFLAGS} -o $(BUILD_DIR)/kernels-impl/Int8-Quaternary.o -c

$(BUILD_DIR)/kernels-impl/Int3-Int3.o:								kernels-impl/Int3-Int3.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels-impl/Int3-Int3.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) $(ARCH_MODIFIER_FLAGS) ${LDFLAGS} -o $(BUILD_DIR)/kernels-impl/Int3-Int3.o -c

$(BUILD_DIR)/kernels-impl/ULPPACK.o:									kernels-impl/ULPPACK.cc \
													common/types.h \
													common/flags.h \
													kernels-impl/ULPPACK/ULPPACK.h \
													kernels-impl/ULPPACK/test.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels-impl/ULPPACK.cc -flax-vector-conversions -lpthread -Wno-psabi -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) $(ARCH_MODIFIER_FLAGS) ${LDFLAGS} -o $(BUILD_DIR)/kernels-impl/ULPPACK.o -c

$(BUILD_DIR)/kernels-impl/ULPPACK/4x8-neon-multipack-type2.o:			kernels-impl/ULPPACK.cc \
													common/types.h \
													common/flags.h \
													kernels-impl/ULPPACK/ULPPACK.h \
													kernels-impl/ULPPACK/test.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels-impl/ULPPACK/4x8-neon-multipack-type2.cpp -flax-vector-conversions -lpthread -Wno-psabi -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) $(ARCH_MODIFIER_FLAGS) ${LDFLAGS} -o $(BUILD_DIR)/kernels-impl/ULPPACK/4x8-neon-multipack-type2.o -c

$(BUILD_DIR)/kernels-impl/ULPPACK/4x8-neon-multipack.o:				kernels-impl/ULPPACK.cc \
													common/types.h \
													common/flags.h \
													kernels-impl/ULPPACK/ULPPACK.h \
													kernels-impl/ULPPACK/test.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels-impl/ULPPACK/4x8-neon-multipack.cpp -flax-vector-conversions -lpthread -Wno-psabi -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) $(ARCH_MODIFIER_FLAGS) ${LDFLAGS} -o $(BUILD_DIR)/kernels-impl/ULPPACK/4x8-neon-multipack.o -c

######  SelfDependent Kernels Start  ######

$(BUILD_DIR)/kernels-impl/SelfDependent-kernels/W4A4.o:				kernels-impl/SelfDependent-kernels/W4A4.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels-impl/SelfDependent-kernels/W4A4.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) $(ARCH_MODIFIER_FLAGS) ${LDFLAGS} -o $(BUILD_DIR)/kernels-impl/SelfDependent-kernels/W4A4.o -c

$(BUILD_DIR)/kernels-impl/SelfDependent-kernels/W4A8.o:				kernels-impl/SelfDependent-kernels/W4A8.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels-impl/SelfDependent-kernels/W4A8.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) $(ARCH_MODIFIER_FLAGS) ${LDFLAGS} -o $(BUILD_DIR)/kernels-impl/SelfDependent-kernels/W4A8.o -c

$(BUILD_DIR)/kernels-impl/SelfDependent-kernels/W8A4.o:				kernels-impl/SelfDependent-kernels/W8A4.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels-impl/SelfDependent-kernels/W8A4.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) $(ARCH_MODIFIER_FLAGS) ${LDFLAGS} -o $(BUILD_DIR)/kernels-impl/SelfDependent-kernels/W8A4.o -c

$(BUILD_DIR)/kernels-impl/SelfDependent-kernels/W2A2.o:				kernels-impl/SelfDependent-kernels/W2A2.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels-impl/SelfDependent-kernels/W2A2.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) $(ARCH_MODIFIER_FLAGS) ${LDFLAGS} -o $(BUILD_DIR)/kernels-impl/SelfDependent-kernels/W2A2.o -c

######  SelfDependent Kernels End  ######

$(BUILD_DIR)/kernels-impl/SelfDependent.o:							kernels-impl/SelfDependent.cc \
													common/types.h \
													common/flags.h \
													$(BUILD_DIR)/kernels-impl/SelfDependent-kernels/W4A4.o \
													$(BUILD_DIR)/kernels-impl/SelfDependent-kernels/W4A8.o \
													$(BUILD_DIR)/kernels-impl/SelfDependent-kernels/W8A4.o \
													$(BUILD_DIR)/kernels-impl/SelfDependent-kernels/W2A2.o \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels-impl/SelfDependent.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) $(ARCH_MODIFIER_FLAGS) ${LDFLAGS} -o $(BUILD_DIR)/kernels-impl/SelfDependent.o -c

######  BarrelShiftMultiplier Kernels Start  ######

$(BUILD_DIR)/kernels-impl/BarrelShiftMultiplier-kernels/W8A8.o:		kernels-impl/BarrelShiftMultiplier-kernels/W8A8.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels-impl/BarrelShiftMultiplier-kernels/W8A8.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) $(ARCH_MODIFIER_FLAGS) ${LDFLAGS} -o $(BUILD_DIR)/kernels-impl/BarrelShiftMultiplier-kernels/W8A8.o -c

$(BUILD_DIR)/kernels-impl/BarrelShiftMultiplier-kernels/W4A4.o:		kernels-impl/BarrelShiftMultiplier-kernels/W4A4.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels-impl/BarrelShiftMultiplier-kernels/W4A4.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) $(ARCH_MODIFIER_FLAGS) ${LDFLAGS} -o $(BUILD_DIR)/kernels-impl/BarrelShiftMultiplier-kernels/W4A4.o -c

######  BarrelShiftMultiplier Kernels End  ######

$(BUILD_DIR)/kernels-impl/BarrelShiftMultiplier.o:					kernels-impl/BarrelShiftMultiplier.cc \
													common/types.h \
													common/flags.h \
													$(BUILD_DIR)/kernels-impl/BarrelShiftMultiplier-kernels/W8A8.o \
													$(BUILD_DIR)/kernels-impl/BarrelShiftMultiplier-kernels/W4A4.o \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels-impl/BarrelShiftMultiplier.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) $(ARCH_MODIFIER_FLAGS) ${LDFLAGS} -o $(BUILD_DIR)/kernels-impl/BarrelShiftMultiplier.o -c

######  Float32 Kernels Start  ######

######  Float32 Kernels End  ######

$(BUILD_DIR)/kernels-impl/Float32.o:								kernels-impl/Float32.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) kernels-impl/Float32.cc -Wno-return-type $(KERNELS_MEM_ACCESS_FLAGS) $(CCFLAGS) $(ARCH_MODIFIER_FLAGS) ${LDFLAGS} -o $(BUILD_DIR)/kernels-impl/Float32.o -c



#############################  Kernels End  #############################

$(BUILD_DIR)/low_precision_fully_connected.o:					low_precision_fully_connected.cc \
													common/types.h \
													common/flags.h \
													$(KERNELS_OBJS) \
													low_precision_fully_connected.h \
													Makefile
	$(CXX) low_precision_fully_connected.cc $(CCFLAGS) ${LDFLAGS} -o $(BUILD_DIR)/low_precision_fully_connected.o -c

$(BUILD_DIR)/ops-implementations/mul/LowPrecisionPacking.o:		ops-implementations/mul/LowPrecisionPacking.cc \
													common/types.h \
													common/flags.h \
													ops-implementations/mul/LowPrecisionPacking.h \
													Makefile
	$(CXX) ops-implementations/mul/LowPrecisionPacking.cc $(CCFLAGS) ${LDFLAGS} -o $(BUILD_DIR)/ops-implementations/mul/LowPrecisionPacking.o -c

$(BUILD_DIR)/low_precision_fully_connected_test.o:				low_precision_fully_connected_test.cc \
													common/types.h \
													common/flags.h \
													low_precision_fully_connected_benchmark.h \
													Makefile
	$(CXX) low_precision_fully_connected_test.cc  $(KERNELS_MEM_ACCESS_FLAGS) -I$(RUY_INC) $(CCFLAGS) ${LDFLAGS} -o $(BUILD_DIR)/low_precision_fully_connected_test.o -c

clean:
	$(RM) -r $(BUILD_DIR) build-*
	$(MAKE) -C ruy DEBUG=$(DEBUG) clean

