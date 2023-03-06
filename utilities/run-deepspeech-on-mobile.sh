#!/usr/bin/env bash

II_MODEL="deepspeech-0.9.3-default-none-INT8.tflite"
FF_MODEL="deepspeech-0.9.3-none-none-none.tflite"

adb push $II_MODEL /data/local/tmp/
adb push $FF_MODEL /data/local/tmp/

adb push benchmark_model_non_ruy /data/local/tmp/benchmark_model_non_ruy
adb push benchmark_model /data/local/tmp/benchmark_model
adb push simpleperf /data/local/tmp/simpleperf

adb shell chmod +x /data/local/tmp/benchmark_model_non_ruy
adb shell chmod +x /data/local/tmp/benchmark_model
adb shell chmod +x /data/local/tmp/simpleperf

bash run-template-tflite.sh \
    --num-iterations 100 --num-warmup-iterations 10 \
    --min-secs 0.000000001 --warmup-min-secs 0.000000001 \
    --models-dir /data/local/tmp --single-model deepspeech-0.9.3-default-none-INT8 \
    --fp32-models-dir /data/local/tmp --single-model-fp32 deepspeech-0.9.3-none-none-none \
    --clear-methods \
    --add-method I8-I8 --add-method I4-I4 \
    --add-method Ternary-Ternary --add-method Binary-Binary \
    --record-report-stat logs/detailed --sleep-multiply-coeff 3 \
    --sleep-between-runs --disbale-perf \
    --enable_XNNPACK_W8A8 --enable_TFLITE_W8A8 \
    --enable_GEMMLOWP --enable_RUY_FP32 \
    --enable_XNNPACK_FP32 --enable_TFLITE_FP32 \
    --enable_EIGEN --enable_ULPPACK1 \
    --enable_ULPPACK2 --enable_ULPPACK3 \
    --model-runner benchmark_model \
    --model-runner-no-ruy benchmark_model_non_ruy \
    .