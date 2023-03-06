#!/bin/bash
ITERATIONS=1000
BENCHMARK=multiplication
BE_VERBOSE=0
VERBOSE=2
USE_SHARED_KERNEL=TRUE
USE_FUSED_LOG=FALSE
POSITIONAL=()

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -v|--be-verbose)
            BE_VERBOSE=1
            shift
            ;;
        -n|--num-iterations)
            ITERATIONS=$2
            shift
            shift
            ;;
        -l|--benchmark-verbose-level)
            VERBOSE=$2
            shift
            shift
            ;;
        -k|--disable-use-shared-kernel)
            USE_SHARED_KERNEL=FALSE
            shift
            ;;
        -u|--use-fused-log)
            USE_FUSED_LOG=TRUE
            shift
            shift
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL[@]}"
if [[ ! -z $1 ]]; then
        echo "Running $1 Benchmark"
        BENCHMARK=$1
fi

function exit_on_fail {
        eval $@
        if [ ! $? -eq 0 ]; then
                ret=$?
                echo "[x] Failed"
                exit
        fi
}
if [[ $BE_VERBOSE = 1 ]]; then
    echo "Running Make with $(nproc) workers and DEBUG set to 0"
fi
make -j$(nproc) DEBUG=0 > /dev/null 2>&1
if [ ! $? -eq 0 ]; then
    echo "Make Failed"
    exit
fi

if [[ $BE_VERBOSE = 1 ]]; then
    echo "Make Done"
    echo "Pushing the new built executables to remote device"
fi
adb push build/$BENCHMARK-example /data/local/tmp/ > /dev/null
if [[ $BE_VERBOSE = 1 ]]; then
    echo "Saving Report to $PWD/run-report.txt"
fi
exec &> >(tee  $PWD/run-report.txt)

if [[ $BE_VERBOSE = 1 ]]; then
    echo -e "Running Each Experiment for $ITERATIONS iterations"
fi

echo -e "Non-Fused Log Multiplication Kernels, Small Weight Matrixes, 3 Layers ( 2048 , 2048 , 2048 )"
if [[ $BE_VERBOSE = 1 ]]; then
    echo adb shell VERBOSE=$VERBOSE USE_SHARED_KERNEL=$USE_SHARED_KERNEL USE_FUSED_LOG=$USE_FUSED_LOG OPERATION_SIZE=0 /data/local/tmp/$BENCHMARK-example $ITERATIONS
fi
exit_on_fail adb shell VERBOSE=$VERBOSE USE_SHARED_KERNEL=$USE_SHARED_KERNEL USE_FUSED_LOG=$USE_FUSED_LOG OPERATION_SIZE=0 /data/local/tmp/$BENCHMARK-example $ITERATIONS

echo -e "Non-Fused Log Multiplication Kernels, Medium Weight Matrixes, 3 Layers ( 2048 , 4096 , 4096 )"
if [[ $BE_VERBOSE = 1 ]]; then
    echo adb shell VERBOSE=$VERBOSE USE_SHARED_KERNEL=$USE_SHARED_KERNEL USE_FUSED_LOG=$USE_FUSED_LOG OPERATION_SIZE=1 /data/local/tmp/$BENCHMARK-example $ITERATIONS
fi
exit_on_fail adb shell VERBOSE=$VERBOSE USE_SHARED_KERNEL=$USE_SHARED_KERNEL USE_FUSED_LOG=$USE_FUSED_LOG OPERATION_SIZE=1 /data/local/tmp/$BENCHMARK-example $ITERATIONS

echo -e "Non-Fused Log Multiplication Kernels, Large Weight Matrixes, 3 Layers ( 2048 , 4096 , 8192 )"
if [[ $BE_VERBOSE = 1 ]]; then
    echo adb shell VERBOSE=$VERBOSE USE_SHARED_KERNEL=$USE_SHARED_KERNEL USE_FUSED_LOG=$USE_FUSED_LOG OPERATION_SIZE=2 /data/local/tmp/$BENCHMARK-example $ITERATIONS
fi
exit_on_fail adb shell VERBOSE=$VERBOSE USE_SHARED_KERNEL=$USE_SHARED_KERNEL USE_FUSED_LOG=$USE_FUSED_LOG OPERATION_SIZE=2 /data/local/tmp/$BENCHMARK-example $ITERATIONS

