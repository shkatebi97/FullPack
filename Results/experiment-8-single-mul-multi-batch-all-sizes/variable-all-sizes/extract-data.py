#!/usr/bin/env python
from locale import currency
from os.path import isfile, join, splitext, isdir
from os import listdir, times
from pprint import pprint
import re
import optparse

parser = optparse.OptionParser()

parser.add_option('-b', '--batch-size',
    action="store", dest="batch_size",
    help="Set batch size", default=16)

options, _ = parser.parse_args()

batch_size = int(options.batch_size)

methods_order = ['I8-I8', 'I8-I4', 'I4-I8', 'I4-I4', 'I8-Ternary', 'Ternary-I8', 'Ternary-Ternary', 'I8-Binary']
multibatch_method_kernels = {
    "I8-I8": "ruy::Kernel8bitNeon",
    "I8-I4": "LowPrecision::FullyConnected::Int4::MultiplyInt8MultiBatched",
    "I4-I8": "LowPrecision::FullyConnected::Int4InputsInt8Weights::MultiplyInt8MultiBatched",
    "I4-I4": "LowPrecision::FullyConnected::Int4InputsInt4Weights::MultiplyInt8MultiBatched",
    "I8-Ternary": "LowPrecision::FullyConnected::Ternary::MultiplyInt8MultiBatched",
    "Ternary-I8": "LowPrecision::FullyConnected::TernaryInputsInt8Weights::MultiplyInt8MultiBatched",
    "Ternary-Ternary": "LowPrecision::FullyConnected::TernaryInputsTernaryWeights::MultiplyInt8MultiBatched",
    "I8-Binary": "LowPrecision::FullyConnected::Binary::MultiplyInt8MultiBatched",
    "Binary-I8": "LowPrecision::FullyConnected::BinaryInputsInt8Weights::MultiplyInt8MultiBatched",
    "Binary-Binary": "LowPrecision::FullyConnected::BinaryInputsBinaryWeights::MultiplyInt8MultiBatched",
}
singlebatch_method_kernels = {
    "I8-I8": "tflite::tensor_utils::NeonMatrixBatchVectorMultiplyAccumulateImpl",
    "I8-I4": "LowPrecision::FullyConnected::Int4::MultiplyInt8SingleBatch",
    "I4-I8": "LowPrecision::FullyConnected::Int4InputsInt8Weights::MultiplyInt8SingleBatch",
    "I4-I4": "LowPrecision::FullyConnected::Int4InputsInt4Weights::MultiplyInt8SingleBatch",
    "I8-Ternary": "LowPrecision::FullyConnected::Ternary::MultiplyInt8SingleBatch",
    "Ternary-I8": "LowPrecision::FullyConnected::TernaryInputsInt8Weights::MultiplyInt8SingleBatch",
    "Ternary-Ternary": "LowPrecision::FullyConnected::TernaryInputsTernaryWeights::MultiplyInt8SingleBatch",
    "I8-Binary": "LowPrecision::FullyConnected::Binary::MultiplyInt8SingleBatch",
    "Binary-I8": "LowPrecision::FullyConnected::BinaryInputsInt8Weights::MultiplyInt8SingleBatch",
    "Binary-Binary": "LowPrecision::FullyConnected::BinaryInputsBinaryWeights::MultiplyInt8SingleBatch",
}

def print_as_csv(title: str, rows_name: str, column_names: list[str], row_names: list[str], data: dict[dict[str]]):
    print("".join(['-']*100))
    print(title)
    print("".join(['-']*100))
    print(rows_name, end=",")
    for column in column_names:
        print(column, end=",")
    print()

    for row in row_names:
        print(row[0], end=",")
        for column in column_names:
            print(data[row[0]][column], end=",")
        print()

def print_as_each_method_as_square_csv(title: str, rows_name: str, column_names: list[str], x_values: list[str], y_values: list[str], c_value: int, data: dict[dict[str]]):
    for column_name in sorted(column_names, key=lambda column: methods_order.index(column)):
        print("".join(['-']*100))
        print(title.format(column_name))
        print("".join(['-']*100))
        print(rows_name, end=",")
        for x in x_values:
            print(x, end=",")
        print()

        for y in y_values:
            print(y, end=",")
            for x in x_values:
                print("{:.1f}".format(data[f"{c_value}x{x}x{y}"][column_name]), end=",")
            print()

iterations = 100
result_dirs = listdir(".")
result_dirs = list(filter(isdir, result_dirs))
result_dirs = list(filter(lambda x: isfile(join('.', x, f"output-{100}.log")), result_dirs))
result_methods_dirs = list(map(lambda x: (x, join('.', x, f"output-{100}.log"), join('.', x)), result_dirs))
results = {}
methods = []
for result_method_dir in result_methods_dirs:
    method, file_name, _ = result_method_dir
    methods.append(method)
    with open(file_name) as file:
        lines = list(map(lambda x: x[:-1], file.readlines()))
        models_size = list(map(lambda line: "{}x{}x{}".format(line.split("-")[0], int(line.split("batch-")[1].split("x")[0]), line.split("x")[1].split(".tflite")[0]), lines[0::2]))
        models_size = list(filter(lambda x: batch_size == int(x.split('x')[0]), models_size))
        models_time = list(map(lambda line: float(line.split("avg=")[1]), lines[1::2]))
        models_size_time = zip(models_size, models_time)
    for model_size_time in models_size_time:
        size, time = model_size_time
        if size in results:
            results[size][method] = time
        else:
            results[size] = { method: time }

sizes = results.keys()
sizes = list(map(lambda x: (x, "{}-batch-{}x{}".format(x.split('x')[0], int(x.split('x')[1]), x.split('x')[2])), sizes))
batch_size = int(sizes[0][1].split("-batch")[0])
input_sizes = sorted(list(set(list(map(lambda x: int(x[1].split("batch-")[1].split("x")[0]), sizes)))))
output_sizes = sorted(list(set(list(map(lambda x: int(x[1].split("batch-")[1].split("x")[1]), sizes)))))

print_as_each_method_as_square_csv(
    "Average Run of each model for {} with " + f"{batch_size} batch size.",
    "Output Size", methods, input_sizes, output_sizes,
    batch_size, results
)

exit(0)

cpu_cycles_kernel_share = {}
instructions_kernel_share = {}
l1d_loads_kernel_share = {}
l1d_misses_kernel_share = {}

cpu_cycles = {}
instructions = {}
l1d_loads = {}
l1d_misses = {}

total_times = {}

for result_method_dir in result_methods_dirs:
    method, file_name, dir_name = result_method_dir
    for size_tuple in sizes:
        size, size_string = size_tuple
        report_file_name = f"simpleperf-{iterations}-{size_string}.report"
        stat_file_name = f"simpleperf-{iterations}-{size_string}.csv"
        with open(join(dir_name, report_file_name)) as report_f:
            # Remove First 2 lines containing nothing important
            report_lines = report_f.readlines()[2:]
            # Remove the end of line form each line
            report_lines = list(map(lambda x: x[:-1], report_lines))
            # Remove lines that start with " " or are empty
            report_lines = list(filter(lambda x: x and x[0] != " ", report_lines))
            # Filter out only kernel lines
            report_lines = list(filter(lambda x: multibatch_method_kernels[method] in x or singlebatch_method_kernels[method] in x, report_lines))
            # Filter out only kernel share percentage
            report_lines = list(map(lambda x: x.split()[1][:-1], report_lines))
            # Convert each percentage to float number
            report_lines = list(map(float, report_lines))
            # Convert each percentage to between 0 and 1 share
            report_lines = list(map(lambda x: x / 100, report_lines))
            # Unpack 4 values of kernel share to corresponding values if exists
            if len(report_lines) == 4:
                cpu_share, instruction_share, miss_share, load_share = report_lines
            else:
                instruction_share = miss_share = 0
                cpu_share         = load_share = 0.00001
        
        with open(join(dir_name, stat_file_name)) as stat_f:
            # Remove First line containing nothing important
            stat_lines = stat_f.readlines()[1:]
            # Seperating and removing test time from other lines
            stat_total_time, stat_lines = stat_lines[-1], stat_lines[:-1]
            # Remove the end of line form each line
            stat_lines = list(map(lambda x: x[:-1], stat_lines))
            # Remove extra non important data
            stat_lines = list(map(lambda x: x.split(",")[0], stat_lines))
            # Converting each data to integer and Unpack each to corresponding value
            cpu_total, instruction_total, miss_total, load_total = list(map(int, stat_lines))
            # Extracting total time and converting to float number 
            total_time = float(stat_total_time.split(",")[1])
        
        if size in cpu_cycles_kernel_share:
            cpu_cycles_kernel_share[size][method]   = cpu_share
            instructions_kernel_share[size][method] = instruction_share
            l1d_misses_kernel_share[size][method]   = miss_share
            l1d_loads_kernel_share[size][method]    = load_share

            cpu_cycles[size][method]   = cpu_total
            instructions[size][method] = instruction_total
            l1d_misses[size][method]   = miss_total
            l1d_loads[size][method]    = load_total

            total_times[size][method]    = total_time
        else:
            cpu_cycles_kernel_share[size]   = { method: cpu_share }
            instructions_kernel_share[size] = { method: instruction_share }
            l1d_misses_kernel_share[size]   = { method: miss_share }
            l1d_loads_kernel_share[size]    = { method: load_share }

            cpu_cycles[size]   = { method: cpu_total }
            instructions[size] = { method: instruction_total }
            l1d_misses[size]   = { method: miss_total }
            l1d_loads[size]    = { method: load_total }

            total_times[size]    = { method: total_time }

print_as_csv(
    "CPU Cycles",
    "Sizes", methods, sizes, cpu_cycles
)
print_as_csv(
    "Kernels CPU Cycles Share",
    "Sizes", methods, sizes, cpu_cycles_kernel_share
)

print_as_csv(
    "Instructions",
    "Sizes", methods, sizes, instructions
)
print_as_csv(
    "Kernels Instructions Share",
    "Sizes", methods, sizes, instructions_kernel_share
)

print_as_csv(
    "L1 Data Cache Loads",
    "Sizes", methods, sizes, l1d_loads
)
print_as_csv(
    "Kernels L1 Data Cache Loads Share",
    "Sizes", methods, sizes, l1d_loads_kernel_share
)

print_as_csv(
    "L1 Data Cache Misses",
    "Sizes", methods, sizes, l1d_misses
)
print_as_csv(
    "Kernels L1 Data Cache Misses Share",
    "Sizes", methods, sizes, l1d_misses_kernel_share
)

print_as_csv(
    "Total Time",
    "Sizes", methods, sizes, total_times
)

        






