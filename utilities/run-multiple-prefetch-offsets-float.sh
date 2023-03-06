#!/bin/bash
aarch64-linux-gnu-g++ multiplication.cc -g -o multiplication -static -O3
echo "[+] Executable Have Been Built"
adb push multiplication /data/local/tmp/ > /dev/null
TIMEFORMAT="%3R"

for prefetch_offset in {1..128}; do
    echo -n "$prefetch_offset - "
    adb shell FLOAT_PREFETCH_OFFSET=$prefetch_offset /data/local/tmp/multiplication float 100 | grep "float Multiplication Done."
done
