#!/usr/bin/env python
"""

"""

from os.path import isfile, join, splitext, isdir
from os import listdir, times
import optparse
from texttable import Texttable
import numpy as np
from termcolor import colored
import matplotlib.pyplot as plt

parser = optparse.OptionParser()

parser.add_option('-o', '--output',
    action="store", dest="output",
    help="Put output table in output file.", default="")
parser.add_option('-b', '--batch-num',
    action="append", dest="batch_nums",
    help="Add number of batches to process", default=[])
parser.add_option('-z', '--size-num',
    action="append", dest="size_nums",
    help="Add number of sizes to process", default=[])
parser.add_option('-m', '--method-name-kernels',
    action="append", dest="methods_name_kernel",
    help="Add a method for processing", default=[])
parser.add_option('-p', '--no-ipc',
    action="store_false", dest="process_ipc",
    help="Disable processing for IPC", default=True)
parser.add_option('-c', '--no-cache-misses',
    action="store_false", dest="process_cmr",
    help="Disable processing for Cache Misses", default=True)
parser.add_option('-t', '--table',
    action="store_true", dest="table",
    help="Output the table of results", default=False)

options, args = parser.parse_args()
color_map = {
    "BLUE": '\033[94m',
    "HEADER": '\033[95m',
    "CYAN": '\033[96m',
    "GREEN": '\033[92m',
    "WARNING": '\033[93m',
    "RED": '\033[91m',
    "END": '\033[0m',
    "BOLD": '\033[1m',
    "UNDERLINE": '\033[4m',
}
method_to_color_map = {
    "I8I8": "CYAN",
    "I8I4": "GREEN",
}

if len(args) == 0:
    print("You must pass a folder to search inside")
    exit(-1)

parent_dir = args[0]
if len(options.batch_nums) == 0:
    print(f"You have not specified any batch number.")
    exit(-2)
if len(options.size_nums) == 0:
    print(f"You have not specified any size number.")
    exit(-3)
if len(options.methods_name_kernel) == 0:
    print(f"You have not specified any method/kernel.")
    exit(-3)
else:
    methods_name_kernels = list(map(lambda x: (x.split(':')[1], x.split(':')[0], x[len(x.split(':')[0]) + len(x.split(':')[1]) + 2:]), options.methods_name_kernel))
    print(methods_name_kernels)
if options.process_ipc:
    if not isdir(parent_dir):
        print(f"The '{parent_dir}' is not a directory.")
        exit(-4)
    if len(listdir(parent_dir)) == 0:
        print(f"The '{parent_dir}' is an empty directory")
        exit(-5)
    discarded_files = []
    for _num_batch in options.batch_nums:
        num_batch = int(_num_batch)
        for _num_size in options.size_nums:
            num_size = int(_num_size)
            for method_kernel in methods_name_kernels:
                method, _, _ = method_kernel
                if not isdir(join(parent_dir, method.upper())):
                    print(f"The '{join(parent_dir, method.upper())}' does not exist.")
                    exit(-6)
                if len(listdir(join(parent_dir, method.upper()))) == 0:
                    print(f"The '{join(parent_dir, method.upper())}' is empty.")
                    exit(-7)
                cpu_file_name  = join(method.upper(), f"simpleperf-multi-batch-{method}-{num_batch}x{num_size}x{num_size}-cpu.report")
                ins_file_name  = join(method.upper(), f"simpleperf-multi-batch-{method}-{num_batch}x{num_size}x{num_size}-instruction.report")
                stat_file_name = join(method.upper(), f"simpleperf-multi-batch-{method}-{num_batch}x{num_size}x{num_size}-stats.csv")
                if not isfile(join(parent_dir, cpu_file_name)):
                    print(f"Cannot open file '{join(parent_dir, cpu_file_name)}', Discarding.")
                    discarded_files.append(f"{num_batch}-{num_size}-{num_size}")
                    continue
                if not isfile(join(parent_dir, ins_file_name)):
                    print(f"Cannot open file '{join(parent_dir, ins_file_name)}', Discarding.")
                    discarded_files.append(f"{num_batch}-{num_size}-{num_size}")
                    continue
                if not isfile(join(parent_dir, stat_file_name)):
                    print(f"Cannot open file '{join(parent_dir, stat_file_name)}', Discarding.")
                    discarded_files.append(f"{num_batch}-{num_size}-{num_size}")
                    continue

    print(f"Files check successfully")
    if options.table:
        ipc_output_table = [["size", "method", "instruction_kernel_share", "instruction", "cpu-cycles_kernel_share", "cpu-cycles", "IPC", "time"]]
    else:
        output_string = ""
    for _num_batch in options.batch_nums:
        num_batch = int(_num_batch)
        for _num_size in options.size_nums:
            num_size = int(_num_size)
            if f"{num_batch}-{num_size}-{num_size}" in discarded_files:
                continue
            if not options.table:
                output_string += f"{num_batch}x{num_size}x{num_size}\n\n\tIPC:\t\t= (kernel_percent * instructions) / (kernel_percent * cpu-cycles)\n"
            for i, method_kernel in enumerate(methods_name_kernels):
                method, name, kernel = method_kernel
                if f"{method}-{num_batch}-{num_size}-{num_size}" in discarded_files:
                    continue
                cpu_file_name  = join(method.upper(), f"simpleperf-multi-batch-{method}-{num_batch}x{num_size}x{num_size}-cpu.report")
                ins_file_name  = join(method.upper(), f"simpleperf-multi-batch-{method}-{num_batch}x{num_size}x{num_size}-instruction.report")
                stat_file_name = join(method.upper(), f"simpleperf-multi-batch-{method}-{num_batch}x{num_size}x{num_size}-stats.csv")
                with open(join(parent_dir, cpu_file_name), 'r') as cpu_f:
                    cpu_functions    = cpu_f.readlines()[7:]
                    cpu_functions    = list(map(lambda function: function[:-1], cpu_functions))
                    cpu_functions    = list(filter(lambda function_line: function_line[0] != " ", cpu_functions))
                    cpu_kernel_line  = list(filter(lambda function_line: kernel in function_line, cpu_functions))[0]
                    cpu_kernel_line  = list(filter(lambda part: part, cpu_kernel_line.split(" ")))
                    cpu_kernel_share = float(cpu_kernel_line[1].split("%")[0]) / 100.0
                with open(join(parent_dir, ins_file_name), 'r') as ins_f:
                    ins_functions    = ins_f.readlines()[7:]
                    ins_functions    = list(map(lambda function: function[:-1], ins_functions))
                    ins_functions    = list(filter(lambda function_line: function_line[0] != " ", ins_functions))
                    ins_kernel_line  = list(filter(lambda function_line: kernel in function_line, ins_functions))[0]
                    ins_kernel_line  = list(filter(lambda part: part, ins_kernel_line.split(" ")))
                    ins_kernel_share = float(ins_kernel_line[1].split("%")[0]) / 100.0
                with open(join(parent_dir, stat_file_name), 'r') as stat_f:
                    stats            = stat_f.readlines()[1:]
                    cpu_total        = int(list(filter(lambda stat: "cpu-cycles:u" in stat, stats))[0].split(",")[0])
                    ins_total        = int(list(filter(lambda stat: "instructions:u" in stat, stats))[0].split(",")[0])
                    time_total       = float(list(filter(lambda stat: "Total test time" in stat, stats))[0].split(",")[1])
                if options.table:
                    ipc_output_table.append([
                        f"{num_batch}x{num_size}x{num_size}" if i == 0 else f"", 
                        f"{name}",
                        f"{ins_kernel_share}",
                        f"{ins_total}",
                        f"{cpu_kernel_share}",
                        f"{cpu_total}",
                        f"{(ins_kernel_share * ins_total) / (cpu_kernel_share * cpu_total)}",
                        f"{(cpu_kernel_share * time_total)}"
                    ])
                else:
                    output_string += f"\t\t{name}:\t({ins_kernel_share:.4f} * {ins_total}) / ({cpu_kernel_share:.4f} * {cpu_total}) = {((ins_kernel_share * ins_total) / (cpu_kernel_share * cpu_total)):.2f} ({(cpu_kernel_share * time_total):.2f})\n"
            if not options.table:
                output_string += f"\n\n"
    if options.table:
        ipc_table = Texttable(max_width=200)
        ipc_table.add_rows(ipc_output_table)
        if options.output:
            with open(options.output, 'w') as output_file:
                output_file.write(ipc_table.draw())
            print(f"File saved in '{options.output}'")
        else:
            print(ipc_table.draw())
    else:
        if options.output:
            with open(options.output, 'w') as output_file:
                output_file.write(output_string)
            print(f"File saved in '{options.output}'")
        else:
            print(output_string)

if options.process_cmr:
    if not isdir(parent_dir):
        print(f"The '{parent_dir}' is not a directory.")
        exit(-4)
    if len(listdir(parent_dir)) == 0:
        print(f"The '{parent_dir}' is an empty directory")
        exit(-5)
    discarded_files = []
    for _num_batch in options.batch_nums:
        num_batch = int(_num_batch)
        for _num_size in options.size_nums:
            num_size = int(_num_size)
            for method_kernel in methods_name_kernels:
                method, _, _ = method_kernel
                if not isdir(join(parent_dir, method.upper())):
                    print(f"The '{join(parent_dir, method.upper())}' does not exist.")
                    exit(-6)
                if len(listdir(join(parent_dir, method.upper()))) == 0:
                    print(f"The '{join(parent_dir, method.upper())}' is empty.")
                    exit(-7)
                load_file_name = join(method.upper(), f"simpleperf-multi-batch-{method}-{num_batch}x{num_size}x{num_size}-l1d-loads.report")
                miss_file_name = join(method.upper(), f"simpleperf-multi-batch-{method}-{num_batch}x{num_size}x{num_size}-l1d-misses.report")
                stat_file_name = join(method.upper(), f"simpleperf-multi-batch-{method}-{num_batch}x{num_size}x{num_size}-stats.csv")
                if not isfile(join(parent_dir, load_file_name)):
                    print(f"Cannot open file '{join(parent_dir, load_file_name)}', Discarding.")
                    discarded_files.append(f"{num_batch}-{num_size}-{num_size}")
                    continue
                if not isfile(join(parent_dir, miss_file_name)):
                    print(f"Cannot open file '{join(parent_dir, miss_file_name)}', Discarding.")
                    discarded_files.append(f"{num_batch}-{num_size}-{num_size}")
                    continue
                if not isfile(join(parent_dir, stat_file_name)):
                    print(f"Cannot open file '{join(parent_dir, stat_file_name)}', Discarding.")
                    discarded_files.append(f"{num_batch}-{num_size}-{num_size}")
                    continue

    print(f"Files check successfully")
    if options.table:
        cmr_output_table = [["size", "method", "miss_kernel_share", "L1d-dcache-misses", "load_kernel_share", "L1d-dcache-loads", "cmr"]]
    else:
        output_string = ""
    for _num_batch in options.batch_nums:
        num_batch = int(_num_batch)
        for _num_size in options.size_nums:
            num_size = int(_num_size)
            if f"{num_batch}-{num_size}-{num_size}" in discarded_files:
                continue
            if not options.table:
                output_string += f"{num_batch}x{num_size}x{num_size}\n\n\tCMR:\t\t= (kernel_percent * cache-misses) / (kernel_percent * cache-loads) * 100\n"
            for i, method_kernel in enumerate(methods_name_kernels):
                method, name, kernel = method_kernel
                load_file_name = join(method.upper(), f"simpleperf-multi-batch-{method}-{num_batch}x{num_size}x{num_size}-l1d-loads.report")
                miss_file_name = join(method.upper(), f"simpleperf-multi-batch-{method}-{num_batch}x{num_size}x{num_size}-l1d-misses.report")
                stat_file_name = join(method.upper(), f"simpleperf-multi-batch-{method}-{num_batch}x{num_size}x{num_size}-stats.csv")
                with open(join(parent_dir, load_file_name), 'r') as load_f:
                    load_functions    = load_f.readlines()[7:]
                    load_functions    = list(map(lambda function: function[:-1], load_functions))
                    load_functions    = list(filter(lambda function_line: function_line[0] != " ", load_functions))
                    load_kernel_line  = list(filter(lambda function_line: kernel in function_line, load_functions))[0]
                    load_kernel_line  = list(filter(lambda part: part, load_kernel_line.split(" ")))
                    load_kernel_share = float(load_kernel_line[1].split("%")[0]) / 100.0
                with open(join(parent_dir, miss_file_name), 'r') as miss_f:
                    miss_functions    = miss_f.readlines()[7:]
                    miss_functions    = list(map(lambda function: function[:-1], miss_functions))
                    miss_functions    = list(filter(lambda function_line: function_line[0] != " ", miss_functions))
                    miss_kernel_line  = list(filter(lambda function_line: kernel in function_line, miss_functions))[0]
                    miss_kernel_line  = list(filter(lambda part: part, miss_kernel_line.split(" ")))
                    miss_kernel_share = float(miss_kernel_line[1].split("%")[0]) / 100.0
                with open(join(parent_dir, stat_file_name), 'r') as stat_f:
                    stats            = stat_f.readlines()[1:]
                    load_total       = int(list(filter(lambda stat: "L1-dcache-loads" in stat, stats))[0].split(",")[0])
                    miss_total       = int(list(filter(lambda stat: "L1-dcache-load-misses" in stat, stats))[0].split(",")[0])
                if options.table:
                    cmr_output_table.append([
                        f"{num_batch}x{num_size}x{num_size}" if i == 0 else f"", 
                        f"{name}",
                        f"{miss_kernel_share}",
                        f"{miss_total}",
                        f"{load_kernel_share}",
                        f"{load_total}",
                        f"{((miss_kernel_share * miss_total) / (load_kernel_share * load_total)) * 100}",
                    ])
                else:
                    output_string += f"\t\t{name}:\t({miss_kernel_share:.4f} * {miss_total}) / ({load_kernel_share:.4f} * {load_total}) * 100 = {(((miss_kernel_share * miss_total) / (load_kernel_share * load_total)) * 100):.2f} %\n"
            if not options.table:
                output_string += f"\n\n"

    if options.table:
        cmr_table = Texttable(max_width=200)
        cmr_table.add_rows(cmr_output_table)
        if options.output:
            with open(options.output, 'w') as output_file:
                output_file.write(cmr_table.draw())
            print(f"File saved in '{options.output}'")
        else:
            print(cmr_table.draw())
    else:
        if options.output:
            with open(options.output, 'a' if options.process_ipc else 'w') as output_file:
                output_file.write(output_string)
            print(f"File saved in '{options.output}'")
        else:
            print(output_string)
                












