#!/bin/bash
ITERATIONS=1000
BENCHMARK=multiplication
BE_VERBOSE=0
VERBOSE=2
USE_SHARED_KERNEL=TRUE
USE_FUSED_LOG=FALSE
NUM_LAYER=3
OPERATION_SIZE=2
REPORT_DIR=execution-reports
POSITIONAL=()

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -v|--be-verbose)
            BE_VERBOSE=1
            shift
            ;;
        -n|--num-iteration)
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
        -y|--num-layers)
            NUM_LAYER=$2
            shift
            shift
            ;;
        -p|--operation-size)
            OPERATION_SIZE=$2
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
if [ ! -d $REPORT_DIR ]; then
    mkdir -p $REPORT_DIR
fi
adb push build/$BENCHMARK-example /data/local/tmp/ > /dev/null
# if [[ $BE_VERBOSE = 1 ]]; then
#     echo "Saving Report to $PWD/run-report.txt"
# fi
# exec &> >(tee  $PWD/run-report.txt)

if [[ $BE_VERBOSE = 1 ]]; then
    echo -e "Running Each Experiment for $ITERATIONS iterations"
fi

echo -e "Recording Float Multiplication"
exit_on_fail adb shell \
                VERBOSE=$VERBOSE \
                USE_SHARED_KERNEL=$USE_SHARED_KERNEL \
                USE_FUSED_LOG=$USE_FUSED_LOG \
                OPERATION_SIZE=$OPERATION_SIZE \
                NUM_LAYER=$NUM_LAYER \
                /data/local/tmp/simpleperf \
                record -e cpu-cycles --call-graph fp \
                --symfs /data/local/tmp/ \
                -o /data/local/tmp/simpleperf.record -- \
                    /data/local/tmp/$BENCHMARK-example float $ITERATIONS
echo -e "Reporting Float Multiplication"
exit_on_fail adb shell \
                /data/local/tmp/simpleperf \
                report -n -g caller \
                --full-callgraph \
                --dsos /data/local/tmp/$BENCHMARK-example \
                --symfs /data/local/tmp/ \
                -i /data/local/tmp/simpleperf.record \
                -o /data/local/tmp/simpleperf.report
echo -e "Pulling Report File"
exit_on_fail adb pull \
                /data/local/tmp/simpleperf.report \
                $REPORT_DIR/simpleperf-$BENCHMARK-float-$ITERATIONS-NL-$NUM_LAYER-OS-$OPERATION_SIZE-cpu-caller.report > /dev/null

echo -e "Recording Int8 Multiplication"
exit_on_fail adb shell \
                VERBOSE=$VERBOSE \
                USE_SHARED_KERNEL=$USE_SHARED_KERNEL \
                USE_FUSED_LOG=$USE_FUSED_LOG \
                OPERATION_SIZE=$OPERATION_SIZE \
                NUM_LAYER=$NUM_LAYER \
                /data/local/tmp/simpleperf \
                record -e cpu-cycles --call-graph fp \
                --symfs /data/local/tmp/ \
                -o /data/local/tmp/simpleperf.record -- \
                    /data/local/tmp/$BENCHMARK-example int8 $ITERATIONS
echo -e "Reporting Int8 Multiplication"
exit_on_fail adb shell \
                /data/local/tmp/simpleperf \
                report -n -g caller \
                --full-callgraph \
                --dsos /data/local/tmp/$BENCHMARK-example \
                --symfs /data/local/tmp/ \
                -i /data/local/tmp/simpleperf.record \
                -o /data/local/tmp/simpleperf.report
echo -e "Pulling Report File"
exit_on_fail adb pull \
                /data/local/tmp/simpleperf.report \
                $REPORT_DIR/simpleperf-$BENCHMARK-int8-$ITERATIONS-NL-$NUM_LAYER-OS-$OPERATION_SIZE-cpu-caller.report > /dev/null

echo -e "Recording Log Multiplication"
exit_on_fail adb shell \
                VERBOSE=$VERBOSE \
                USE_SHARED_KERNEL=$USE_SHARED_KERNEL \
                USE_FUSED_LOG=$USE_FUSED_LOG \
                OPERATION_SIZE=$OPERATION_SIZE \
                NUM_LAYER=$NUM_LAYER \
                /data/local/tmp/simpleperf \
                record -e cpu-cycles --call-graph fp \
                --symfs /data/local/tmp/ \
                -o /data/local/tmp/simpleperf.record -- \
                    /data/local/tmp/$BENCHMARK-example log $ITERATIONS
echo -e "Reporting Log Multiplication"
exit_on_fail adb shell \
                /data/local/tmp/simpleperf \
                report -n -g caller \
                --full-callgraph \
                --dsos /data/local/tmp/$BENCHMARK-example \
                --symfs /data/local/tmp/ \
                -i /data/local/tmp/simpleperf.record \
                -o /data/local/tmp/simpleperf.report
echo -e "Pulling Report File"
exit_on_fail adb pull \
                /data/local/tmp/simpleperf.report \
                $REPORT_DIR/simpleperf-$BENCHMARK-log-$ITERATIONS-NL-$NUM_LAYER-OS-$OPERATION_SIZE-cpu-caller.report > /dev/null

echo -e "Recording Hybrid Multiplication"
exit_on_fail adb shell \
                VERBOSE=$VERBOSE \
                USE_SHARED_KERNEL=$USE_SHARED_KERNEL \
                USE_FUSED_LOG=$USE_FUSED_LOG \
                OPERATION_SIZE=$OPERATION_SIZE \
                NUM_LAYER=$NUM_LAYER \
                /data/local/tmp/simpleperf \
                record -e cpu-cycles --call-graph fp \
                --symfs /data/local/tmp/ \
                -o /data/local/tmp/simpleperf.record -- \
                    /data/local/tmp/$BENCHMARK-example hybrid $ITERATIONS
echo -e "Reporting Hybrid Multiplication"
exit_on_fail adb shell \
                /data/local/tmp/simpleperf \
                report -n -g caller \
                --full-callgraph \
                --dsos /data/local/tmp/$BENCHMARK-example \
                --symfs /data/local/tmp/ \
                -i /data/local/tmp/simpleperf.record \
                -o /data/local/tmp/simpleperf.report
echo -e "Pulling Report File"
exit_on_fail adb pull \
                /data/local/tmp/simpleperf.report \
                $REPORT_DIR/simpleperf-$BENCHMARK-hybrid-$ITERATIONS-NL-$NUM_LAYER-OS-$OPERATION_SIZE-cpu-caller.report > /dev/null
