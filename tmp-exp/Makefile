CXX = /usr/bin/aarch64-linux-gnu-g++
CC = /usr/bin/aarch64-linux-gnu-gcc

DEBUG ?= 0
ifeq ($(DEBUG), 1)
    CCFLAGS = -static -lstdc++ -g -march=armv8.2-a+fp16 -DIS_ARM -DIS_ARM64 -lm -flax-vector-conversions
else
    CCFLAGS = -static -lstdc++ -O3 -march=armv8.2-a+fp16 -DIS_ARM -DIS_ARM64 -lm -flax-vector-conversions
endif
ASM ?= 0
ifeq ($(ASM), 1)
    CCFLAGS += -S
else
    CCFLAGS +=
endif

SRC := $(wildcard *.cc)
OBJ := $(SRC:%.cc=%.o)
ASM := $(SRC:%.cc=%.S)

.PHONY: all clean

all: $(OBJ) $(ASM)

%.o: %.cc
	$(CC) -o $@ $< $(CCFLAGS)
	chmod +x $@

%.S: %.cc
	$(CC) -o $@ $< $(CCFLAGS) -S

clean:
	@$(RM) -rv $(OBJ) $(ASM)
