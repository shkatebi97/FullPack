#!/usr/bin/env bash

REMOTE_PREFIX_DIR="/home/pi/Desktop/run-CNNs"
REMOTE_II_DIR="/home/pi/Desktop/run-CNNs/models/i8i8"
REMOTE_MODEL_RUNNER_PATH="$REMOTE_PREFIX_DIR/benchmark_model"
REMOTE_MODEL_RUNNER_NON_RUY_PATH="$REMOTE_PREFIX_DIR/benchmark_model_non_ruy"

II_DIR="/home/user/Project/models/CNNs/i8i8"
MODEL_RUNNER_PATH="/home/user/Project/Experiments/tensorflow-fullpack/bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model"
MODEL_RUNNER_NON_RUY_PATH="/home/user/Project/Experiments/tensorflow-fullpack/bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model_non_ruy"

run-on-raspberry mkdir -p $REMOTE_PREFIX_DIR
run-on-raspberry mkdir -p $REMOTE_II_DIR

# if [[ `ls $II_DIR | wc -l` != `run-on-raspberry ls $REMOTE_II_DIR | wc -l` ]]; then
#     push-to-raspberry $II_DIR $REMOTE_II_DIR
# else
#     echo "[+] Models exist on remote device, not sending."
# fi
if [[ -z $DISCARD_PUSHING_TOOLS || $DISCARD_PUSHING_TOOLS = 1 ]]; then
    push-to-raspberry $MODEL_RUNNER_PATH $REMOTE_MODEL_RUNNER_PATH
    push-to-raspberry $MODEL_RUNNER_NON_RUY_PATH $REMOTE_MODEL_RUNNER_NON_RUY_PATH
else
    echo "[!] Discarding sending tools."
fi

bash run-template-tflite.sh \
    --num-iterations 20 \
    --num-warmup-iterations 10 \
    --min-secs 0.000000001 \
    --warmup-min-secs 0.000000001 \
    --models-dir $REMOTE_II_DIR \
    --clear-methods \
    --add-method I8-I8 \
    --add-method I4-I4 \
    --add-method Ternary-Ternary \
    --add-method Binary-Binary \
    --enable_TFLITE_W8A8 \
    --enable_GEMMLOWP \
    --enable_ULPPACK1 \
    --enable_ULPPACK2 \
    --enable_ULPPACK3 \
    --record-report-stat logs \
    --sleep-constant-secs 20 \
    --sleep-between-runs \
    --model-runner $REMOTE_MODEL_RUNNER_PATH \
    --model-runner-no-ruy $REMOTE_MODEL_RUNNER_NON_RUY_PATH \
    --add-taskset-mode f \
    --use-less-storage-space \
    --run-on-remote run-on-raspberry push-to-raspberry pull-from-raspberry \
    --save-per-operation-profile \
    $II_DIR

# --use-less-storage-space \
# --enable_XNNPACK_W8A8 \
# --enable_TFLITE_W8A8 \
# --enable_GEMMLOWP \
# --enable_RUY_FP32 \
# --enable_XNNPACK_FP32 \
# --enable_TFLITE_FP32 \
# --enable_EIGEN \
# --enable_ULPPACK1 \
# --enable_ULPPACK2 \
# --enable_ULPPACK3 \
