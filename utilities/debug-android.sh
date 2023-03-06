#!/bin/bash
ADB=$(which adb)
DEBUG=0
FILE=example-multiplication
NUM_ITER=1000
VERBOSE_LEVEL=3
MODES="float,hybrid,int8,i8-shift,i8-binary,f32-binary,i8-i4,i8-4shift"
CAPTURE_LOG=0
OPERATION_SIZE=1
CLEAR_SCREEN=0
USE_SHARED_KERNEL=TRUE
USE_FUSED_LOG=FALSE
NUM_LAYER=3
DO_MAKE=1
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -d|--debug)
            DEBUG=1
            shift
            ;;
        -c|--capture-log)
            CAPTURE_LOG=1
            shift
            ;;
        -s|--clear-screen)
            CLEAR_SCREEN=1
            shift
            ;;
        -f|--file)
            FILE=$2
            shift
            shift
            ;;
        -e|--no-method)
            MODES=
            shift
            ;;
        -p|--operation-size)
            OPERATION_SIZE=$2
            shift
            shift
            ;;
        -n|--num-iteration)
            NUM_ITER=$2
            shift
            shift
            ;;
        -l|--num-layers)
            NUM_LAYER=$2
            shift
            shift
            ;;
        -v|--verbose-level)
            VERBOSE_LEVEL=$2
            shift
            shift
            ;;
        -m|--no-make)
            DO_MAKE=0
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
    if [[ $VERBOSE_LEVEL = 3 ]]; then
        echo "Set Operation Modes to $1"
    fi
    MODES=$1
fi

if [[ $CLEAR_SCREEN = 1 ]];then
    clear
fi
$ADB forward tcp:1234 tcp:1234 > /dev/null
$ADB shell pkill -9 -f gdbserver

if [[ ! -f build/$FILE ]]; then
    FILE=multiplication-example
fi

if [[ $DO_MAKE = 1 ]]; then
    if [[ $DEBUG = 1 ]]; then
        make -j$(nproc) DEBUG=1
    else
        make -j$(nproc) DEBUG=0
    fi
    if [ ! $? -eq 0 ]; then
        echo "Make Failed"
        exit
fi
fi
if [[ $VERBOSE_LEVEL = 3 ]]; then
    $ADB push build/$FILE /data/local/tmp/ > /dev/null
else
    $ADB push build/$FILE /data/local/tmp/ > /dev/null
fi
if [[ $DEBUG = 1 ]]; then
    if [[ $CAPTURE_LOG = 1 ]]; then
        $ADB logcat > tmp/ruy-float-restart-logcat.log &
        LOG_PID=$(ps axf | grep "adb logcat" | grep -v grep | awk '{print $1}')
        echo "Started the log capturer process with pid of $LOG_PID"
    fi
    if [[ $VERBOSE_LEVEL = 3 ]]; then
        echo "Executing:"
        echo time $ADB shell VERBOSE=$VERBOSE_LEVEL USE_SHARED_KERNEL=$USE_SHARED_KERNEL USE_FUSED_LOG=$USE_FUSED_LOG OPERATION_SIZE=$OPERATION_SIZE NUM_LAYER=$NUM_LAYER taskset f0 /data/local/tmp/gdbserver :1234 /data/local/tmp/$FILE $MODES $NUM_ITER
    fi
    time $ADB shell VERBOSE=$VERBOSE_LEVEL USE_SHARED_KERNEL=$USE_SHARED_KERNEL USE_FUSED_LOG=$USE_FUSED_LOG OPERATION_SIZE=$OPERATION_SIZE NUM_LAYER=$NUM_LAYER taskset f0 /data/local/tmp/gdbserver :1234 /data/local/tmp/$FILE $MODES $NUM_ITER
    if [[ $CAPTURE_LOG = 1 ]]; then
        echo "Stopping log capturing."
        kill -9 $LOG_PID
        echo "The Log Capture process stoped. The logs are written in $PWD/tmp/ruy-float-restart-logcat.log"
    fi
else
    if [[ $VERBOSE_LEVEL = 3 ]]; then
        echo "Executing:"
        echo time $ADB shell VERBOSE=$VERBOSE_LEVEL USE_SHARED_KERNEL=$USE_SHARED_KERNEL USE_FUSED_LOG=$USE_FUSED_LOG OPERATION_SIZE=$OPERATION_SIZE NUM_LAYER=$NUM_LAYER taskset f0 /data/local/tmp/$FILE $MODES $NUM_ITER
    fi
    if [[ $VERBOSE_LEVEL = 3 ]]; then
        time $ADB shell VERBOSE=$VERBOSE_LEVEL USE_SHARED_KERNEL=$USE_SHARED_KERNEL USE_FUSED_LOG=$USE_FUSED_LOG OPERATION_SIZE=$OPERATION_SIZE NUM_LAYER=$NUM_LAYER taskset f0 /data/local/tmp/$FILE $MODES $NUM_ITER
    else
        $ADB shell VERBOSE=$VERBOSE_LEVEL USE_SHARED_KERNEL=$USE_SHARED_KERNEL USE_FUSED_LOG=$USE_FUSED_LOG OPERATION_SIZE=$OPERATION_SIZE NUM_LAYER=$NUM_LAYER taskset f0 /data/local/tmp/$FILE $MODES $NUM_ITER
    fi
fi