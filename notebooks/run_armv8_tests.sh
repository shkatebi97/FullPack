#!/usr/bin/env bash
# Build & run ARMv8 harness under qemu-aarch64 (uses existing make objects).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REL=build-aarch64  # make targets are relative to ROOT (see Makefile BUILD_DIR)
BUILD="$ROOT/$REL"
OUT="$BUILD/armv8_layer_select_harness"
QEMU="${QEMU:-qemu-aarch64-static}"
CXX=aarch64-linux-gnu-g++
FLAGS="-O2 -std=c++14 -static -pthread -lstdc++ -lm -flax-vector-conversions -Wno-pointer-arith -Wno-narrowing -DIS_ARM -DIS_ARM64 -DTFLITE_BUILD -march=armv8.2-a+fp16 -I$ROOT"

cd "$ROOT"
make -j"$(nproc)" TARGET_ISA=aarch64 \
  "$REL/libfullpack.so" \
  "$REL/low_precision_fully_connected.o" \
  "$REL/ops-implementations/mul/LowPrecisionPacking.o" \
  "$REL/kernels-impl/Float32.o" >/tmp/fullpack_make.log 2>&1 || {
  tail -50 /tmp/fullpack_make.log; exit 1;
}

OBJS=(
  "$BUILD/low_precision_fully_connected.o"
  "$BUILD/ops-implementations/mul/LowPrecisionPacking.o"
  "$BUILD/kernels-impl/Float32.o"
  "$BUILD/kernels-impl/Int8-Int4.o"
  "$BUILD/kernels-impl/Int4-Int8.o"
  "$BUILD/kernels-impl/Int4-Int4.o"
  "$BUILD/kernels-impl/Int8-Ternary.o"
  "$BUILD/kernels-impl/Ternary-Int8.o"
  "$BUILD/kernels-impl/Ternary-Ternary.o"
  "$BUILD/kernels-impl/Int8-Binary.o"
  "$BUILD/kernels-impl/Binary-Int8.o"
  "$BUILD/kernels-impl/Binary-Binary.o"
  "$BUILD/kernels-impl/Binary-Binary-XOR.o"
  "$BUILD/kernels-impl/Int8-Quaternary.o"
  "$BUILD/kernels-impl/Int3-Int3.o"
  "$BUILD/kernels-impl/ULPPACK.o"
  "$BUILD/kernels-impl/ULPPACK/4x8-neon-multipack.o"
  "$BUILD/kernels-impl/ULPPACK/4x8-neon-multipack-type2.o"
  "$BUILD/kernels-impl/SelfDependent.o"
  "$BUILD/kernels-impl/SelfDependent-kernels/W4A4.o"
  "$BUILD/kernels-impl/SelfDependent-kernels/W4A8.o"
  "$BUILD/kernels-impl/SelfDependent-kernels/W8A4.o"
  "$BUILD/kernels-impl/SelfDependent-kernels/W2A2.o"
  "$BUILD/kernels-impl/BarrelShiftMultiplier.o"
  "$BUILD/kernels-impl/BarrelShiftMultiplier-kernels/W8A8.o"
  "$BUILD/kernels-impl/BarrelShiftMultiplier-kernels/W4A4.o"
)

NEED_LINK=1
if [[ -x "$OUT" ]]; then
  newest_src=$(find "$ROOT/notebooks/armv8_layer_select_harness.cc" "$ROOT/low_precision_fully_connected.cc" "$ROOT/kernels-impl/Float32.cc" -printf '%T@
' | sort -n | tail -1)
  bin_t=$(stat -c '%Y' "$OUT")
  # compare as integers
  if python3 -c "import sys; sys.exit(0 if float('$newest_src') <= $bin_t else 1)"; then
    NEED_LINK=0
  fi
fi
if [[ "$NEED_LINK" == 1 ]]; then
  echo "Linking $OUT ..."
  $CXX $FLAGS "$ROOT/notebooks/armv8_layer_select_harness.cc" "${OBJS[@]}" -o "$OUT"
else
  echo "Reusing existing $OUT"
fi
echo "Running under $QEMU ..."
"$QEMU" "$OUT"
