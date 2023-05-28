#!/usr/bin/env python
from pathlib import Path
from os.path import isfile, join, isdir, splitext
from os import listdir
from types import FunctionType
import xlsxwriter
import optparse
import json
import re
import os, sys
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common_config import *
from contextlib import redirect_stdout


parser = optparse.OptionParser()

parser.add_option('-b', '--batch-size',
    action="store", dest="batch_size",
    help="Set batch size", default=1)
parser.add_option('-n', '--iterations',
    action="store", dest="iterations",
    help="Set iteration count", default=50)
parser.add_option('-w', '--warmup-iterations',
    action="store", dest="warmup_iterations",
    help="Set iteration count", default=10)
parser.add_option('-o', '--output',
    action="store", dest="output",
    help="Set the output", default=sys.stdout)
parser.add_option('-s', '--speedups',
    action="store", dest="speedups",
    help="Enable Speedup Mode", default=None)

options, _ = parser.parse_args()

batch_size = int(options.batch_size)
iterations = 1
warmup_iterations = 1
results = {}
methods = []

def ignore_ignored_sizes(size):
    for ignored_size in options.ignore_sizes:
        if ignored_size in size[1].split("batch-")[1]:
            return False
    return True

def print_as_csv(title: str, rows_name: str, column_names: list[str], row_names: list[str], data: dict[dict[str]], output_file = sys.stdout):
    # output_file.write("".join(['-']*100) + "\n")
    output_file.write("   " + title + "   " + "\n")
    # output_file.write("".join(['-']*100) + "\n")
    output_file.write(rows_name + ",")
    for column in column_names:
        output_file.write(column + ",")
    output_file.write("\n")

    for row in row_names:
        output_file.write(row + ",")
        for column in column_names:
            try:
                output_file.write(str(data[row][column]) + ",")
            except KeyError:
                output_file.write("-,")
        output_file.write("\n")

def print_as_csv_wrt_baseline(title: str, rows_name: str, column_names: list[str], row_names: list[str], data: dict[dict[str]], baseline: str, output_file = sys.stdout):
    output_file.write("   " + title + "   " + "\n")
    output_file.write(rows_name + ",")
    for column in column_names:
        output_file.write(column + ",")
    output_file.write("\n")

    for row in row_names:
        output_file.write(row + ",")
        for column in column_names:
            try:
                output_file.write("{:.2f}".format(data[row][baseline] / data[row][column]) + ",")
            except KeyError:
                output_file.write("-,")
        output_file.write("\n")

iterations = options.iterations
warmup_iterations = options.warmup_iterations

result_dirs = listdir(".")
result_dirs = list(filter(isdir, result_dirs))
result_dirs = list(filter(lambda x: isfile(join(x, f"output-{iterations}-{warmup_iterations}.log")), result_dirs))
result_methods_dirs = list(map(lambda x: (x, join(x, f"output-{iterations}-{warmup_iterations}.log"), join('.', x)), result_dirs))

for result_method_dir in result_methods_dirs:
    method, file_name, _ = result_method_dir
    methods.append(method)
    with open(file_name) as file:
        lines = list(map(lambda x: x[:-1], file.readlines()))
        models_names = lines[0::2]
        models_time = list(map(lambda line: float(line), lines[1::2]))
        models_name_time = zip(models_names, models_time)
    for model_name_time in models_name_time:
        name, time = model_name_time
        if name in results:
            results[name][method] = time
        else:
            results[name] = { method: time }

methods = sorted(methods, key=lambda column: methods_order.index(column))
models_names = list(results.keys())
if options.output != sys.stdout:
    deepspeech_csv_path = splitext(options.output)[0] + "-deepspeech" + splitext(options.output)[1]
    print_as_csv("DeepSpeech", "Models", methods, models_names, results, open(deepspeech_csv_path, "w"))
    if options.speedups:
        deepspeech_csv_path = splitext(options.speedups)[0] + "-deepspeech" + splitext(options.speedups)[1]
        print_as_csv_wrt_baseline("DeepSpeech", "Models", methods, models_names, results, "I8-I8", open(deepspeech_csv_path, "w"))
else:
    print_as_csv("DeepSpeech", "Models", methods, models_names, results)
    print_as_csv_wrt_baseline("DeepSpeech", "Models", methods, models_names, results, "I8-I8", open(join("CSVs", "speedups-imagenet-.csv"), "w"))
exit(0)
