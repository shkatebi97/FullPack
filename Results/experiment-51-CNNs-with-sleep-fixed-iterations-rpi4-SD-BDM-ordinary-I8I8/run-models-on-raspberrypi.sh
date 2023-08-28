#!/usr/bin/env bash

REMOTE_PREFIX_DIR="/home/pi/Desktop/CNNs-ResNet101-DenseNet201-InceptionV3"
REMOTE_II_DIR="/home/pi/Desktop/CNNs-ResNet101-DenseNet201-InceptionV3/models/i8i8"
REMOTE_FF_DIR="/home/pi/Desktop/CNNs-ResNet101-DenseNet201-InceptionV3/models/f32f32"
REMOTE_MODEL_RUNNER_PATH="$REMOTE_PREFIX_DIR/benchmark_model"
REMOTE_MODEL_RUNNER_NON_RUY_PATH="$REMOTE_PREFIX_DIR/benchmark_model_non_ruy"

II_DIR="/home/user/Project/models/CNNs-ResNet101-DenseNet201-InceptionV3/i8i8"
FF_DIR="/home/user/Project/models/CNNs-ResNet101-DenseNet201-InceptionV3/f32f32"
MODEL_RUNNER_PATH="/home/user/Project/Experiments/tensorflow-fullpack/bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model"
MODEL_RUNNER_NON_RUY_PATH="/home/user/Project/Experiments/tensorflow-fullpack/bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model_non_ruy"

run-on-raspberry mkdir -p $REMOTE_PREFIX_DIR
run-on-raspberry mkdir -p $REMOTE_II_DIR
run-on-raspberry mkdir -p $REMOTE_FF_DIR

# if [[ `ls $II_DIR | wc -l` != `run-on-raspberry ls $REMOTE_II_DIR | wc -l` ]]; then
#     push-to-raspberry $II_DIR $REMOTE_II_DIR
# else
#     echo "[+] Models exist on remote device, not sending."
# fi
if [[ -z $DISCARD_PUSHING_TOOLS || $DISCARD_PUSHING_TOOLS = 0 ]]; then
    push-to-raspberry $MODEL_RUNNER_PATH $REMOTE_MODEL_RUNNER_PATH
    push-to-raspberry $MODEL_RUNNER_NON_RUY_PATH $REMOTE_MODEL_RUNNER_NON_RUY_PATH
else
    echo "[!] Discarding sending tools."
fi

bash run-template-tflite.sh \
    --num-iterations 100 \
    --num-warmup-iterations 2 \
    --min-secs 0.000000001 \
    --warmup-min-secs 0.000000001 \
    --models-dir $REMOTE_II_DIR \
    --fp32-models-dir $REMOTE_FF_DIR \
    --clear-methods \
    --add-method I8-I8 \
    --add-method BSM-W8A8 \
    --add-method SelfDependent-W4A4 \
    --add-method SelfDependent-W4A8 \
    --add-method SelfDependent-W8A4 \
    --add-method ULPPACK-W4A4 \
    --enable_TFLITE_W8A8 \
    --enable_GEMMLOWP \
    --enable_RUY_FP32 \
    --enable_TFLITE_FP32 \
    --enable_EIGEN \
    --record-report-stat logs \
    --sleep-constant-secs 10 \
    --sleep-between-runs \
    --model-runner $REMOTE_MODEL_RUNNER_PATH \
    --model-runner-no-ruy $REMOTE_MODEL_RUNNER_NON_RUY_PATH \
    --add-taskset-mode f \
    --use-less-storage-space \
    --run-on-remote run-on-raspberry push-to-raspberry pull-from-raspberry \
    --save-per-operation-profile \
    $II_DIR $FF_DIR

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

# --add-method I4-I4 \

run-on-raspberry rm -r $REMOTE_PREFIX_DIR
