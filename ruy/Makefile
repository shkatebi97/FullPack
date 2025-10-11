BAZEL=bazel

DEBUG ?= 1
ifeq ($(DEBUG), 1)
    BUILD_TYPE = dbg
else
    BUILD_TYPE = opt
endif

ENABLE_RUY_PROFILER ?= 0
ifeq ($(ENABLE_RUY_PROFILER), 1)
    RUY_PROFILER_FLAGS = --define ruy_profiler=true
else
    RUY_PROFILER_FLAGS = --define ruy_profiler=false
endif

DISABLE_KERNELS_MEM_ACCESS ?= 1
ifeq ($(DISABLE_KERNELS_MEM_ACCESS), 1)
    KERNELS_MEM_ACCESS_FLAGS = -DDISABLE_KERNELS_MEM_ACCESS
else
    KERNELS_MEM_ACCESS_FLAGS = -UDISABLE_KERNELS_MEM_ACCESS
endif

TARGET_ISA ?= aarch64
ifeq ($(TARGET_ISA), aarch64)
    RUY_TARGET_ISA := aarch64
    RUY_TARGET_EXT := aarch64
else ifeq ($(TARGET_ISA), x86_64-avx512)
    RUY_TARGET_ISA := linux_x86_64
    RUY_TARGET_EXT := avx512
else ifeq ($(TARGET_ISA), x86_64-avx2)
    RUY_TARGET_ISA := linux_x86_64
    RUY_TARGET_EXT := avx2
else ifeq ($(TARGET_ISA), x86_64-avx)
    RUY_TARGET_ISA := linux_x86_64
    RUY_TARGET_EXT := avx
endif

all:
	@echo ========== Ran $(value BAZEL) build --jobs `nproc` --config=$(value RUY_TARGET_ISA) --config=$(value RUY_TARGET_EXT) $(value RUY_PROFILER_FLAGS) --copt=$(value KERNELS_MEM_ACCESS_FLAGS) --cxxopt=$(value KERNELS_MEM_ACCESS_FLAGS) -c $(value BUILD_TYPE) //example:multiplication-example ==========
	$(shell $(BAZEL) build --jobs `nproc` --config=$(RUY_TARGET_ISA) --config=$(RUY_TARGET_EXT) $(RUY_PROFILER_FLAGS) --copt=$(KERNELS_MEM_ACCESS_FLAGS) --cxxopt=$(KERNELS_MEM_ACCESS_FLAGS) -c $(BUILD_TYPE) //example:multiplication-example)

clean:
	$(shell $(BAZEL) clean)