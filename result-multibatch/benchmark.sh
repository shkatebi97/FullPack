#! /bin/bash
echo -n "Running Int8... "
adb shell /data/local/tmp/simpleperf stat -e cpu-cycles:u,instructions:u,L1-dcache-load-misses,L1-dcache-loads taskset f0 /data/local/tmp/low_precision_fully_connected_test benchmark-multi-batch Int8 > simpleperf-cpu-instruction-stat-Int8-multibatch.log
echo -e "\rDone Int8.                          "
echo -n "Running Int4... "
adb shell /data/local/tmp/simpleperf stat -e cpu-cycles:u,instructions:u,L1-dcache-load-misses,L1-dcache-loads taskset f0 /data/local/tmp/low_precision_fully_connected_test benchmark-multi-batch Int4 > simpleperf-cpu-instruction-stat-Int4-multibatch.log
echo -e "\rDone Int4.                          "
echo -n "Running Binary... "
adb shell /data/local/tmp/simpleperf stat -e cpu-cycles:u,instructions:u,L1-dcache-load-misses,L1-dcache-loads taskset f0 /data/local/tmp/low_precision_fully_connected_test benchmark-multi-batch Binary > simpleperf-cpu-instruction-stat-Binary-multibatch.log
echo -e "\rDone Binary.                          "
echo -n "Running Ternary... "
adb shell /data/local/tmp/simpleperf stat -e cpu-cycles:u,instructions:u,L1-dcache-load-misses,L1-dcache-loads taskset f0 /data/local/tmp/low_precision_fully_connected_test benchmark-multi-batch Ternary > simpleperf-cpu-instruction-stat-Ternary-multibatch.log
echo -e "\rDone Ternary.                          "
echo -n "Running Int4InputsInt8Weights... "
adb shell /data/local/tmp/simpleperf stat -e cpu-cycles:u,instructions:u,L1-dcache-load-misses,L1-dcache-loads taskset f0 /data/local/tmp/low_precision_fully_connected_test benchmark-multi-batch Int4InputsInt8Weights > simpleperf-cpu-instruction-stat-Int4InputsInt8Weights-multibatch.log
echo -e "\rDone Int4InputsInt8Weights.                          "
echo -n "Running Int4InputsInt4Weights... "
adb shell /data/local/tmp/simpleperf stat -e cpu-cycles:u,instructions:u,L1-dcache-load-misses,L1-dcache-loads taskset f0 /data/local/tmp/low_precision_fully_connected_test benchmark-multi-batch Int4InputsInt4Weights > simpleperf-cpu-instruction-stat-Int4InputsInt4Weights-multibatch.log
echo -e "\rDone Int4InputsInt4Weights.                          "
echo -n "Running TernaryInputsInt8Weights... "
adb shell /data/local/tmp/simpleperf stat -e cpu-cycles:u,instructions:u,L1-dcache-load-misses,L1-dcache-loads taskset f0 /data/local/tmp/low_precision_fully_connected_test benchmark-multi-batch TernaryInputsInt8Weights > simpleperf-cpu-instruction-stat-TernaryInputsInt8Weights-multibatch.log
echo -e "\rDone TernaryInputsInt8Weights.                          "
echo -n "Running TernaryInputsTernaryWeights... "
adb shell /data/local/tmp/simpleperf stat -e cpu-cycles:u,instructions:u,L1-dcache-load-misses,L1-dcache-loads taskset f0 /data/local/tmp/low_precision_fully_connected_test benchmark-multi-batch TernaryInputsTernaryWeights > simpleperf-cpu-instruction-stat-TernaryInputsTernaryWeights-multibatch.log
echo -e "\rDone TernaryInputsTernaryWeights.                          "