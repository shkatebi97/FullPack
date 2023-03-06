#!/usr/bin/env bash

II_DIR="single-mul-single-batch-all-sizes-i8i8"
FF_DIR="single-mul-single-batch-all-sizes-f32f32"

if [[ ! -d $II_DIR ]]; then
    mkdir -p $II_DIR
    for i in 128 256 512 1024 2048 8192; do
        for j in 128 256 512 1024 2048 8192; do
            python3 generate-template-tflite.py -z $i -r 1 -i $j -o $II_DIR/model-1-batch-"$j"x"$i".tflite --quantize-activations
        done
    done
fi
if [[ ! -d $FF_DIR ]]; then
    mkdir -p $FF_DIR
    for i in 128 256 512 1024 2048 8192; do
        for j in 128 256 512 1024 2048 8192; do
            python3 generate-template-tflite.py -z $i -r 1 -i $j -o $FF_DIR/model-1-batch-"$j"x"$i".tflite --no-optimization
        done
    done
fi

adb push $II_DIR /data/local/tmp/$II_DIR
adb push $FF_DIR /data/local/tmp/$FF_DIR

adb push benchmark_model_non_ruy /data/local/tmp/benchmark_model_non_ruy
adb push benchmark_model /data/local/tmp/benchmark_model
adb push simpleperf /data/local/tmp/simpleperf

adb shell chmod +x /data/local/tmp/benchmark_model_non_ruy
adb shell chmod +x /data/local/tmp/benchmark_model
adb shell chmod +x /data/local/tmp/simpleperf

bash run-template-tflite.sh \
    --num-iterations 200 --num-warmup-iterations 50 \
    --min-secs 0.000000001 --warmup-min-secs 0.000000001 \
    --models-dir /data/local/tmp/$II_DIR \
    --fp32-models-dir /data/local/tmp/$FF_DIR \
    --clear-methods --add-method I8-I8 --add-method I4-I4 \
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
    $II_DIR