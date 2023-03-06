#!/bin/bash

function list_contains() {
        local list=("$@")
        local value=${list[-1]}
        unset list[-1]
        for v in ${list[@]}; do
                if [[ "$v" = "$value" ]]; then
                        return 1
                fi
        done
        return 0
}

function print_help {
        echo -e "$0 [options] -- command args"
        echo -e "OPTIONS:"
        echo -e "\t-o\t--output"
        echo -e "\t\tSpecify a path to store output report.\n"
        echo -e "\t-e\t--event"
        echo -e "\t\tSpecify record event.\n"
        echo -e "\t-t\t--report-type"
        echo -e "\t\tSpecify report type. Can be 'caller' or 'callee'.\n"
        echo -e "\t-v\t--enviroment-varible"
        echo -e "\t\tAdd enviroment variable to the list. Should be used like 'var=val'.\n"
        echo -e "\t-c\t--record-flags"
        echo -e "\t\tAdd record phase flags to the record flags list.\n"
        echo -e "\t-p\t--report-flags"
        echo -e "\t\tAdd report phase flags to the report flags list.\n"
        echo -e "\t-h\t--help"
        echo -e "\t\tPrint this help.\n"
}

function exit_on_fail {
        eval $@
        if [ ! $? -eq 0 ]; then
                ret=$?
                echo "[x] Failed"
                exit
        fi
}

REC_FLAGS=( --call-graph fp )
REP_FLAGS=()
event=cpu-cycles:u
group=
report_type=caller
callgraph_flag=--full-callgraph
generate_stat=
stat_events=cpu-cycles:u
COMMAND=()
ENV=()
adb_options=()
output=/tmp/output.report
device_string=$(adb devices | head -2 | tail -1 | cut -f 1)
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -a|--add-adb-options)
        adb_options+=($2)
        shift
        shift
        ;;
    -i|--device-name)
        device_string=$2
        shift
        shift
        ;;
    -o|--output)
        output=$2
        shift
        shift
        ;;
    -e|--event)
        event=$2
        shift
        shift
        ;;
    -g|--group)
        group=$2
        shift
        shift
        ;;
    -d|--disable-callgraph)
        callgraph_flag=
        shift
        ;;
    -s|--generate-stat-csv)
        generate_stat=$2
        stat_events=$3
        shift
        shift
        shift
        ;;
    -t|--report-type)
        report_type=$2
        shift
        shift
        ;;
    -v|--enviroment-varible)
        if `list_contains "${ENV[@]}" "$2"`; then
            ENV+=($2)
        else
            echo "Enviroment Variable $2 is already in the list"
        fi
        shift
        shift
        ;;
    -c|--record-flags)
        if `list_contains "${REC_FLAGS[@]}" "$2"`; then
            REC_FLAGS+=($2)
        else
            echo "Record Flag $2 is already in the list"
        fi
        shift
        shift
        ;;
    -p|--report-flags)
        if `list_contains "${REP_FLAGS[@]}" "$2"`; then
            REP_FLAGS+=($2)
        else
            echo "Report Flag $2 is already in the list"
        fi
        shift
        shift
        ;;
    -h|--help)
        print_help $@
        exit
        ;;
    --)
        shift
        while [[ $# -gt 0 ]]; do
            COMMAND+=("$1")
            shift
        done
        ;;
    *)
        shift
        ;;
  esac
done
if [[ -z $COMMAND ]]; then
        echo "You must pass a command to execute"
        exit
fi

if [[ -z $group ]]; then
    exit_on_fail adb -s $device_string ${adb_options[@]} shell ${ENV[@]} /data/local/tmp/simpleperf record -e $event -o /data/local/tmp/simpleperf.record ${REC_FLAGS[@]} -- ${COMMAND[@]}
else
    exit_on_fail adb -s $device_string ${adb_options[@]} shell ${ENV[@]} /data/local/tmp/simpleperf record --group $group -o /data/local/tmp/simpleperf.record ${REC_FLAGS[@]} -- ${COMMAND[@]}
fi
exit_on_fail adb -s $device_string ${adb_options[@]} shell /data/local/tmp/simpleperf report $callgraph_flag -g $report_type -i /data/local/tmp/simpleperf.record -o /data/local/tmp/simpleperf.report ${REP_FLAGS[@]}
exit_on_fail adb -s $device_string ${adb_options[@]} pull /data/local/tmp/simpleperf.report $output
if [[ ! -z $generate_stat ]]; then
    exit_on_fail adb -s $device_string ${adb_options[@]} shell ${ENV[@]} /data/local/tmp/simpleperf stat --csv -e $stat_events -o /data/local/tmp/simpleperf.csv ${COMMAND[@]}
    exit_on_fail adb -s $device_string ${adb_options[@]} pull /data/local/tmp/simpleperf.csv $generate_stat
fi


