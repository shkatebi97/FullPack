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
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common_config import *
from contextlib import redirect_stdout
import subprocess
import glob


parser = optparse.OptionParser()

parser.add_option('-n', '--iterations',
    action="store", dest="iterations",
    help="Set iteration count", default=100)
parser.add_option('-w', '--warmup-iterations',
    action="store", dest="warmup_iterations",
    help="Set iteration count", default=2)
parser.add_option('-o', '--output',
    action="store", dest="output",
    help="Set the output", default=sys.stdout)
parser.add_option('-s', '--speedups',
    action="store", dest="speedups",
    help="Enable Speedup Mode", default=None)
parser.add_option('--speedups-against',
    action="append", dest="speedups_against",
    help="Add the method name to baselines", default=["I8-I8"])
parser.add_option('-p', '--per-layer-breakdown',
    action="store", dest="per_layer_breakdown",
    help="Set Per Layer Breakdown saving path", default=None)
parser.add_option('-d', '--detailed-breakdown',
    action="store", dest="detailed_breakdown",
    help="Set Detailed Breakdown saving path", default=None)
parser.add_option('-S', '--detailed-speedup',
    action="store", dest="detailed_speedup",
    help="Set Detailed Speedups saving path", default=None)
parser.add_option('-c', '--detailed-comparison',
    action="store", dest="detailed_comparison",
    help="Set Detailed Comparison saving path", default=None)

options, _ = parser.parse_args()

iterations = 1
warmup_iterations = 1
results = {}
functions_results = {}
per_layer_results = {}
methods = []
functions = {
    "HasChild": True,
    "Overall": {
        "I8-I8": "tflite::Subgraph::OpInvoke",
        "GEMMLOWP": "tflite::Subgraph::OpInvoke",
        "No-Caching": "tflite::Subgraph::OpInvoke",
        "I4-I4": "tflite::Subgraph::OpInvoke",
        "BSM-W8A8": "tflite::Subgraph::OpInvoke",
        "SelfDependent-W4A4": "tflite::Subgraph::OpInvoke",
        "Ternary-Ternary": "tflite::Subgraph::OpInvoke",
        "Binary-Binary": "tflite::Subgraph::OpInvoke",
        "ULPPACK-W1A1": "tflite::Subgraph::OpInvoke",
        "ULPPACK-W2A2": "tflite::Subgraph::OpInvoke",
        "ULPPACK-W3A3": "tflite::Subgraph::OpInvoke",
        "ULPPACK-W4A4": "tflite::Subgraph::OpInvoke",
    },
    "Conv": {
        "HasChild": True,
        "Overall": {
            "I8-I8": "tflite::ops::builtin::conv::Eval<(tflite::ops::builtin::conv::KernelType)1>",
            "GEMMLOWP": "tflite::ops::builtin::conv::Eval<(tflite::ops::builtin::conv::KernelType)2>",
            "No-Caching": "tflite::ops::builtin::conv::Eval<(tflite::ops::builtin::conv::KernelType)1>",
            "I4-I4": "tflite::ops::builtin::conv::Eval<(tflite::ops::builtin::conv::KernelType)1>",
            "BSM-W8A8": "tflite::ops::builtin::conv::Eval<(tflite::ops::builtin::conv::KernelType)1>",
            "SelfDependent-W4A4": "tflite::ops::builtin::conv::Eval<(tflite::ops::builtin::conv::KernelType)1>",
            "Ternary-Ternary": "tflite::ops::builtin::conv::Eval<(tflite::ops::builtin::conv::KernelType)1>",
            "Binary-Binary": "tflite::ops::builtin::conv::Eval<(tflite::ops::builtin::conv::KernelType)1>",
            "ULPPACK-W1A1": "tflite::ops::builtin::conv::Eval<(tflite::ops::builtin::conv::KernelType)1>",
            "ULPPACK-W2A2": "tflite::ops::builtin::conv::Eval<(tflite::ops::builtin::conv::KernelType)1>",
            "ULPPACK-W3A3": "tflite::ops::builtin::conv::Eval<(tflite::ops::builtin::conv::KernelType)1>",
            "ULPPACK-W4A4": "tflite::ops::builtin::conv::Eval<(tflite::ops::builtin::conv::KernelType)1>",
        },
        "GEMM": {
            "HasChild": True,
            "Overall": {
                "I8-I8": "tflite::ops::builtin::conv::EvalQuantizedPerChannel<(tflite::ops::builtin::conv::KernelType)1>",
                "GEMMLOWP": "tflite::ops::builtin::conv::EvalQuantizedPerChannel<(tflite::ops::builtin::conv::KernelType)1>",
                "No-Caching": "tflite::ops::builtin::conv::EvalQuantizedPerChannel<(tflite::ops::builtin::conv::KernelType)1>",
                "I4-I4": "tflite::ops::builtin::conv::EvalQuantizedPerChannel<(tflite::ops::builtin::conv::KernelType)1>",
                "BSM-W8A8": "tflite::ops::builtin::conv::EvalQuantizedPerChannel<(tflite::ops::builtin::conv::KernelType)1>",
                "SelfDependent-W4A4": "tflite::ops::builtin::conv::EvalQuantizedPerChannel<(tflite::ops::builtin::conv::KernelType)1>",
                "Ternary-Ternary": "tflite::ops::builtin::conv::EvalQuantizedPerChannel<(tflite::ops::builtin::conv::KernelType)1>",
                "Binary-Binary": "tflite::ops::builtin::conv::EvalQuantizedPerChannel<(tflite::ops::builtin::conv::KernelType)1>",
                "ULPPACK-W1A1": "tflite::ops::builtin::conv::EvalQuantizedPerChannel<(tflite::ops::builtin::conv::KernelType)1>",
                "ULPPACK-W2A2": "tflite::ops::builtin::conv::EvalQuantizedPerChannel<(tflite::ops::builtin::conv::KernelType)1>",
                "ULPPACK-W3A3": "tflite::ops::builtin::conv::EvalQuantizedPerChannel<(tflite::ops::builtin::conv::KernelType)1>",
                "ULPPACK-W4A4": "tflite::ops::builtin::conv::EvalQuantizedPerChannel<(tflite::ops::builtin::conv::KernelType)1>",
            },
            # "Overall": {
            #     "I8-I8": "ruy::Mul<(ruy::Path)16, signed char, signed char, int, signed char>",
            #     "GEMMLOWP": "tflite::cpu_backend_gemm::detail::GemmImplUsingGemmlowp<signed char, signed char, int, signed char, (tflite::cpu_backend_gemm::QuantizationFlavor)2>::Run",
            #     "No-Caching": "ruy::Mul<(ruy::Path)16, signed char, signed char, int, signed char>",
            #     "I4-I4": "LowPrecision::FullyConnected::Mul",
            #     "BSM-W8A8": "LowPrecision::FullyConnected::Mul",
            #     "SelfDependent-W4A4": "LowPrecision::FullyConnected::Mul",
            #     "Ternary-Ternary": "LowPrecision::FullyConnected::Mul",
            #     "Binary-Binary": "LowPrecision::FullyConnected::Mul",
            #     "ULPPACK-W1A1": "LowPrecision::FullyConnected::Mul",
            #     "ULPPACK-W2A2": "LowPrecision::FullyConnected::Mul",
            #     "ULPPACK-W3A3": "LowPrecision::FullyConnected::Mul",
            #     "ULPPACK-W4A4": "LowPrecision::FullyConnected::Mul",
            # },
            "Kernel": {
                "HasChild": False,
                "Overall": {
                    "I8-I8": "ruy::TrMulParams::RunKernel",
                    "GEMMLOWP": "tflite::cpu_backend_gemm::detail::GemmImplUsingGemmlowp<signed char, signed char, int, signed char, (tflite::cpu_backend_gemm::QuantizationFlavor)2>::Run",
                    "No-Caching": "ruy::TrMulParams::RunKernel",
                    "I4-I4": "LowPrecision::GEMM",
                    "BSM-W8A8": "LowPrecision::GEMM",
                    "SelfDependent-W4A4": "LowPrecision::GEMM",
                    "Ternary-Ternary": "LowPrecision::GEMM",
                    "Binary-Binary": "LowPrecision::GEMM",
                    "ULPPACK-W1A1": "LowPrecision::GEMM",
                    "ULPPACK-W2A2": "LowPrecision::GEMM",
                    "ULPPACK-W3A3": "LowPrecision::GEMM",
                    "ULPPACK-W4A4": "LowPrecision::GEMM",
                },
            },
            "Packing": {
                "HasChild": False,
                "Overall": {
                    "I8-I8": "ruy::(anonymous namespace)::TrMulTask::EnsurePacked",
                    "GEMMLOWP": "tflite::cpu_backend_gemm::detail::GemmImplUsingGemmlowp<signed char, signed char, int, signed char, (tflite::cpu_backend_gemm::QuantizationFlavor)2>::Run",
                    "No-Caching": "ruy::(anonymous namespace)::TrMulTask::EnsurePacked",
                    "I4-I4": "LowPrecision::PrepareMatrixAsInputForMethod",
                    "BSM-W8A8": "LowPrecision::PrepareMatrixAsInputForMethod",
                    "SelfDependent-W4A4": "LowPrecision::PrepareMatrixAsInputForMethod",
                    "Ternary-Ternary": "LowPrecision::PrepareMatrixAsInputForMethod",
                    "Binary-Binary": "LowPrecision::PrepareMatrixAsInputForMethod",
                    "ULPPACK-W1A1": "LowPrecision::PrepareMatrixAsInputForMethod",
                    "ULPPACK-W2A2": "LowPrecision::PrepareMatrixAsInputForMethod",
                    "ULPPACK-W3A3": "LowPrecision::PrepareMatrixAsInputForMethod",
                    "ULPPACK-W4A4": "LowPrecision::PrepareMatrixAsInputForMethod",
                },
            },
        },
        "Others": {
            "HasChild": False,
            "Overall": {
                "I8-I8": "tflite::optimized_ops::Im2col<signed char>",
                "GEMMLOWP": "tflite::optimized_ops::Im2col<signed char>",
                "No-Caching": "tflite::optimized_ops::Im2col<signed char>",
                "I4-I4": "tflite::optimized_ops::Im2col<signed char>",
                "BSM-W8A8": "tflite::optimized_ops::Im2col<signed char>",
                "SelfDependent-W4A4": "tflite::optimized_ops::Im2col<signed char>",
                "Ternary-Ternary": "tflite::optimized_ops::Im2col<signed char>",
                "Binary-Binary": "tflite::optimized_ops::Im2col<signed char>",
                "ULPPACK-W1A1": "tflite::optimized_ops::Im2col<signed char>",
                "ULPPACK-W2A2": "tflite::optimized_ops::Im2col<signed char>",
                "ULPPACK-W3A3": "tflite::optimized_ops::Im2col<signed char>",
                "ULPPACK-W4A4": "tflite::optimized_ops::Im2col<signed char>",
            },
        },
    },
    
}

figure_tex_template = """
\\documentclass{{article}}
\\usepackage{{pgfplots}}
\\usepackage{{pgfplotstable}}
\\usepackage{{tikz}}
\\pgfplotsset{{compat=1.8}}
\\begin{{document}}
    \\begin{{figure*}}[t]
        \\begin{{tikzpicture}}
            \\begin{{axis}}[
                xbar=0pt,
                xmin=0.6,
                xmax=3,
                width=0.995\\linewidth,
                height=2*\\axisdefaultheight,
                tick label style={{font=\\footnotesize}},
                legend style={{font=\\tiny}},
                label style={{font=\\footnotesize}},
                symbolic y coords={{{METHODS}}},
                ytick=data,
                xtick={{0.60,0.80,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0}},
                align={{center}},
                bar width={WIDTH}ex,
                legend columns=7,
                legend style={{at={{(0.49,-0.2)}},anchor=north,font=\\footnotesize}},
                title=\\textbf{{{TITLE}}},
                xmajorgrids,
                yminorgrids = true,
                minor tick num=1
                ]
                {PLOTS}
                \\legend{{
                    {LEGENDS}
                }};
            \\end{{axis}}
        \\end{{tikzpicture}}
    \\end{{figure*}}
\\end{{document}}
"""

figure_stacked_ybar_tex_template = """
\\documentclass{{article}}
\\usepackage{{pgfplots}}
\\usepackage{{pgfplotstable}}
\\usepackage{{tikz}}
\\pgfplotsset{{compat=1.8}}
\\begin{{document}}
    \\begin{{figure*}}[t]
        \\begin{{tikzpicture}}
            \\begin{{axis}}[
                ybar stacked,
                ymin=0.0,
                width=\\linewidth,
                height=\\axisdefaultheight,
                tick label style={{font=\\footnotesize}},
                legend style={{font=\\tiny}},
                label style={{font=\\footnotesize}},
                symbolic x coords={{{METHODS}}},
                xtick=data,
                % ytick={{0.60,0.80,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0}},
                align={{center}},
                bar width={WIDTH}ex,
                legend columns=7,
                legend style={{at={{(0.49,-0.2)}},anchor=north,font=\\footnotesize}},
                title=\\textbf{{{TITLE}}},
                ymajorgrids,
                xminorgrids = true,
                minor tick num=1
                ]
                {PLOTS}
                \\legend{{
                    {LEGENDS}
                }};
            \\end{{axis}}
        \\end{{tikzpicture}}
    \\end{{figure*}}
\\end{{document}}
"""

single_plot_tex_template = """\\addplot coordinates {{ {COORDS} }};"""


def is_float(x: float):
    try:
        float(x)
        return True
    except ValueError:
        return False

def print_as_csv(title: str, rows_name: str, column_names: list[str], row_names: list[str], data: dict[dict[str]], output_file = sys.stdout, close_output_at_finish = False):
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
    if close_output_at_finish:
        output_file.close()

def print_as_csv_wrt_baseline(title: str, rows_name: str, column_names: list[str], row_names: list[str], data: dict[dict[str]], baseline: str, output_file = sys.stdout, close_output_at_finish = False, reverse_order = False, do_print = False, data_is_callable = False):
    if title is not None:
        output_file.write("   " + title + "   " + "\n")
    output_file.write(rows_name + ",")
    for column in column_names:
        if not reverse_order and baseline == column:
                continue
        output_file.write(column + ",")
    output_file.write("\n")

    for row in row_names:
        if reverse_order and baseline == row:
            continue
        output_file.write(row + ",")
        for column in column_names:
            if not reverse_order and baseline == column:
                continue
            try:
                if data_is_callable:
                    if data(row, column) is not None and data(row, column) != 0:
                        output_file.write("{:.2f}".format(data(row, baseline) / data(row, column)) + ",")
                    else:
                        output_file.write("{:.2f}".format(0) + ",")
                elif reverse_order:
                    if do_print:
                        print(f"reverse_order: {reverse_order} -> data[{baseline}][{column}] / data[{row}][{column}]")
                    output_file.write("{:.2f}".format(data[baseline][column] / data[row][column]) + ",")
                else:
                    if do_print:
                        print(f"data[{row}][{baseline}] / data[{row}][{column}]")
                    if data[row][column] is not None and data[row][column] != 0:
                        output_file.write("{:.2f}".format(data[row][baseline] / data[row][column]) + ",")
                    else:
                        output_file.write("{:.2f}".format(0) + ",")
            except KeyError as e:
                output_file.write("-,")
        output_file.write("\n")
    if close_output_at_finish:
        output_file.close()

iterations = options.iterations
warmup_iterations = options.warmup_iterations

result_dirs = listdir(".")
# print(result_dirs)
methods = list(filter(lambda dir: isdir(join(".", dir)) and dir in methods_order, result_dirs))
# print(methods)
# print(list(filter(lambda x: isdir(join('.', methods[0], x)) and isfile(join('.', methods[0], x, f"output-{iterations}-{warmup_iterations}.log")), listdir(join('.', methods[0])))))
models = [ list(filter(lambda x: isdir(join('.', methods[0], x)) and isfile(join('.', method, x, f"output-{iterations}-{warmup_iterations}.log")), listdir(join('.', method)))) for method in methods ]
models = list(set([ item for sublist in models for item in sublist ]))
# print(models)
# print(f"output-{iterations}-{warmup_iterations}.log")
models_layer = {}
models_layer_figure_tex = {}
temp_models = [ *models ]
for model in temp_models:
    results[model] = {}
    per_layer_results[model] = {}
    functions_results[model] = {}
    dismiss_model = False
    conv_2d_operations = []
    fully_connected_operations = []
    for method in methods:
        # File Pathes
        output_file_path = join('.', method, model, f'output-{iterations}-{warmup_iterations}.log')
        report_file_path = join('.', method, model, f'cpu-cycles-{iterations}-{warmup_iterations}.report')
        
        # Check if this model execution logs exist for this method
        if not isfile(output_file_path):
            results[model][method] = 0.0
            continue

        # Loading output file
        if not isfile(output_file_path):
            results[model][method] = 0.0
            continue
        with open(output_file_path) as output_file:
            output_file_lines = list(filter(lambda line: line, map(lambda line: line[:-1], output_file.readlines())))

        # Parsing end-to-end total time in seconds
        try:
            selected_lines = list(filter(lambda line: "Inference timings in us:" in line, output_file_lines))
            if len(selected_lines) > 0:
                results[model][method] = float(selected_lines[0].split()[-1]) / 1000
            else:
                results[model][method] = 0.0
        except ValueError:
            results[model][method] = 0.0

        # Parsing per layer profile
        try:
            start_idx = output_file_lines.index("Operator-wise Profiling Info for Regular Benchmark Runs:") + 3
        except ValueError:
            continue
            # models.remove(model)
            # dismiss_model = True
            # break
        end_idx = output_file_lines[start_idx:].index("============================== Top by Computation Time ==============================") + start_idx
        selected_lines = output_file_lines[start_idx:end_idx]
        selected_lines = list(map(lambda line: line.split(), selected_lines))
        conv_2d_operations = list(map(lambda operation: (float(operation[3]), float(operation[4][:-1])), filter(lambda line: line[0] == "CONV_2D", selected_lines)))
        fully_connected_operations = list(map(lambda operation: (float(operation[3]), float(operation[4][:-1])), filter(lambda line: line[0] == "FULLY_CONNECTED", selected_lines)))
        per_layer_results[model][method] = dict(
            [ 
                *list(map(lambda x: (f"CONV-{x[0] + 1}", x[1][0]), enumerate(conv_2d_operations))), 
                *list(map(lambda x: (f"FC-{x[0] + 1}", x[1][0]), enumerate(fully_connected_operations))), 
            ]
        )

        # Parsing report
        if not isfile(report_file_path):
            continue
        with open(report_file_path) as report_file:
            report_file_lines = list(filter(lambda line: line, map(lambda line: line[:-1], report_file.readlines())))
        reported_functions_w_percent = list(filter(lambda line: '[.] ' in line, report_file_lines))
        reported_functions_w_percent = dict(map(lambda func: (
            ' '.join(func.split()[5:]),
            (
                float(func.split()[0][:-1]) / 100,
                float(func.split()[1][:-1]) / 100,
            ),
        ), reported_functions_w_percent))
        functions_results[model][method] = reported_functions_w_percent
    
    if len(conv_2d_operations) == 0 or len(fully_connected_operations) == 0:
        models.remove(model)
        continue

    models_layer[model] = [ 
        *list(map(lambda x: f"CONV-{x[0] + 1}", enumerate(conv_2d_operations))), 
        *list(map(lambda x: f"FC-{x[0] + 1}", enumerate(fully_connected_operations))), 
    ]
    models_layer_figure_tex[model] = figure_tex_template.format(
        METHODS=", ".join(methods[1:]),
        TITLE=model,
        LEGENDS="\n                    ".join(list(map(lambda layer: f"{layer},", models_layer[model]))),
        PLOTS="\n                ".join(list(map(
            lambda layer: single_plot_tex_template.format(
                COORDS=" ".join(
                    list(map(lambda method: f"({per_layer_results[model]['I8-I8'][layer] / per_layer_results[model][method][layer]:.2f},{method})" if method in per_layer_results[model] else f"({0:.2f},{method})", methods[1:]))
                )
            ),
            models_layer[model]
        ))),
        WIDTH=float(f"{30 / len(models_layer[model]):.3f}")
    )

methods = sorted(methods, key=lambda column: methods_order.index(column))
models_names = models
models_names_1K_imagenet = list(sorted(filter(lambda x: "-21K" not in x, models_names)))
models_names_21K_imagenet = list(sorted(filter(lambda x: "-21K" in x, models_names)))

print("Generating Execution Time and Speedup CSVs")

if options.output != sys.stdout:
    if len(models_names_21K_imagenet) == 0:
        imagenet_1K_csv_path = splitext(options.output)[0] + splitext(options.output)[1]
        print_as_csv("CNNs", "Models", methods, models_names_1K_imagenet, results, open(imagenet_1K_csv_path, "w"), close_output_at_finish=True)
    else:
        imagenet_1K_csv_path = splitext(options.output)[0] + "-imagenet-1K" + splitext(options.output)[1]
        imagenet_21K_csv_path = splitext(options.output)[0] + "-imagenet-21K" + splitext(options.output)[1]
        print_as_csv("CNNs-ImageNet-1K", "Models", methods, models_names_1K_imagenet, results, open(imagenet_1K_csv_path, "w"), close_output_at_finish=True)
        print_as_csv("CNNs-ImageNet-21K", "Models", methods, models_names_21K_imagenet, results, open(imagenet_21K_csv_path, "w"), close_output_at_finish=True)
    if options.speedups:
        for baseline in options.speedups_against:
            if len(models_names_21K_imagenet) == 0:
                speedup_imagenet_1K_csv_path = splitext(options.speedups)[0] + "-wrt-" + baseline + splitext(options.speedups)[1]
                print_as_csv_wrt_baseline("CNNs", "Models", methods, models_names_1K_imagenet, results, baseline, open(speedup_imagenet_1K_csv_path, "w"), close_output_at_finish=True)
            else:
                speedup_imagenet_1K_csv_path = splitext(options.speedups)[0] + "-wrt-" + baseline + "-imagenet-1K" + splitext(options.speedups)[1]
                speedup_imagenet_21K_csv_path = splitext(options.speedups)[0] + "-wrt-" + baseline + "-imagenet-21K" + splitext(options.speedups)[1]
                print_as_csv_wrt_baseline("CNNs-ImageNet-1K", "Models", methods, models_names_1K_imagenet, results, baseline, open(speedup_imagenet_1K_csv_path, "w"), close_output_at_finish=True)
                print_as_csv_wrt_baseline("CNNs-ImageNet-21K", "Models", methods, models_names_21K_imagenet, results, baseline, open(speedup_imagenet_21K_csv_path, "w"), close_output_at_finish=True)
            
else:
    if len(models_names_21K_imagenet) > 0:
        print_as_csv("CNNs-ImageNet-1K", "Models", methods, models_names_1K_imagenet, results)
        print_as_csv("CNNs-ImageNet-21K", "Models", methods, models_names_21K_imagenet, results)
    else:
        print_as_csv("CNNs", "Models", methods, models_names_1K_imagenet, results)
    Path("CSVs").mkdir(exist_ok=True, parents=True)
    for baseline in options.speedups_against:
        if len(models_names_21K_imagenet) > 0:
            print_as_csv_wrt_baseline("CNNs-ImageNet-1K", "Models", methods, models_names_1K_imagenet, results, baseline, open(join("CSVs", f"speedups-wrt-{baseline}-imagenet-1K.csv"), "w"), close_output_at_finish=True)
            print_as_csv_wrt_baseline("CNNs-ImageNet-21K", "Models", methods, models_names_21K_imagenet, results, baseline, open(join("CSVs", f"speedups-wrt-{baseline}-imagenet-21K.csv"), "w"), close_output_at_finish=True)
        else:
            print_as_csv_wrt_baseline("CNNs", "Models", methods, models_names_1K_imagenet, results, baseline, open(join("CSVs", f"speedups-wrt-{baseline}.csv"), "w"), close_output_at_finish=True)

print("Generating Figures TeXs")

per_layer_breakdown_parent_path = Path(options.per_layer_breakdown if options.per_layer_breakdown else "PerLayerBreakdown")
per_layer_breakdown_parent_path.mkdir(exist_ok=True, parents=True)
per_layer_breakdown_figure_tex_parent_path = Path(join(per_layer_breakdown_parent_path, "Figures-Texs"))
per_layer_breakdown_figure_tex_parent_path.mkdir(exist_ok=True, parents=True)
per_layer_breakdown_figure_parent_path = Path(join(per_layer_breakdown_parent_path, "Figures"))
per_layer_breakdown_figure_parent_path.mkdir(exist_ok=True, parents=True)

for model in models_names:
    print_as_csv_wrt_baseline(
        # f"{model} Per Layer Breakdown for each method",
        None, "Methods",
        models_layer[model], methods, per_layer_results[model], "I8-I8",
        output_file = open(join(per_layer_breakdown_parent_path, f"{model}.csv"), "w"),
        reverse_order = True, close_output_at_finish = True
    )
    with open(join(per_layer_breakdown_figure_tex_parent_path, f"{model}.tex"), "w") as f:
        f.write(models_layer_figure_tex[model])

print("Generating Figures PDFs")
for model in models_names:
    print("Running", " ".join(["pdflatex", "-shell-escape", "-interaction=nonstopmode", "-file-line-error", f"-output-directory={str(per_layer_breakdown_figure_parent_path)}", join(per_layer_breakdown_figure_tex_parent_path, f"{model}.tex")]))
    subprocess.run(["pdflatex", "-shell-escape", "-interaction=nonstopmode", "-file-line-error", f"-output-directory={str(per_layer_breakdown_figure_parent_path)}", join(per_layer_breakdown_figure_tex_parent_path, f"{model}.tex")], check=True, capture_output=True)

for texAuxFile in glob.iglob(os.path.join(per_layer_breakdown_figure_parent_path, '*.aux')):
    os.remove(texAuxFile)
for texLogFile in glob.iglob(os.path.join(per_layer_breakdown_figure_parent_path, '*.log')):
    os.remove(texLogFile)

exit(0)

print("Generating Detailed Breakdown YAMLs")
detailed_breakdown = {}
for model in models:
    detailed_breakdown[model] = ""
    for method in methods:
        if method not in functions_results[model]:
            continue
        model_method_result = functions_results[model][method]
        try:
            detailed_breakdown[model] += f"""{method}:
    Overall: '{model_method_result[functions['Overall'][method]][0] * results[model][method]:.2f}'
    Conv:
        Overall: '{model_method_result[functions['Conv']['Overall'][method]][0] * results[model][method]:.2f}'
        GEMM: 
            Overall: '{model_method_result[functions['Conv']['GEMM']['Overall'][method]][0] * results[model][method]:.2f}'
            Kernel:  '{model_method_result[functions['Conv']['GEMM']['Kernel']['Overall'][method]][0] * results[model][method]:.2f}'
            Packing: '{model_method_result[functions['Conv']['GEMM']['Packing']['Overall'][method]][0] * results[model][method]:.2f}'
        Others (Im2Col):
            Overall: '{model_method_result[functions['Conv']['Others']['Overall'][method]][0] * results[model][method]:.2f}
'
"""
        except KeyError:
            continue

detailed_breakdown_parent_path = Path(options.detailed_breakdown if options.detailed_breakdown else "DetailedBreakdown")
detailed_breakdown_parent_path.mkdir(exist_ok=True, parents=True)

for model in models:
    with open(join(detailed_breakdown_parent_path, f'{model}.yaml'), 'w') as f:
        f.write(detailed_breakdown[model])

print("Generating Detailed Speedup YAMLs")
detailed_speedups = {}
for model in models:
    detailed_speedups[model] = ""
    for method in methods:
        if method == 'I8-I8':
            continue
        if method not in functions_results[model]:
            continue
        model_method_result = functions_results[model][method]

        try:
            detailed_speedups[model] += f"""W8A8 / {method}:
    Overall: '{(functions_results[model]['I8-I8'][functions['Overall']['I8-I8']][0] * results[model]['I8-I8']) / (model_method_result[functions['Overall'][method]][0] * results[model][method]):.2f}'
    Conv:
        Overall: '{(functions_results[model]['I8-I8'][functions['Conv']['Overall']['I8-I8']][0] * results[model]['I8-I8']) / (model_method_result[functions['Conv']['Overall'][method]][0] * results[model][method]):.2f}'
        GEMM: 
            Overall: '{(functions_results[model]['I8-I8'][functions['Conv']['GEMM']['Overall']['I8-I8']][0] * results[model]['I8-I8']) / (model_method_result[functions['Conv']['GEMM']['Overall'][method]][0] * results[model][method]):.2f}'
            Kernel:  '{(functions_results[model]['I8-I8'][functions['Conv']['GEMM']['Kernel']['Overall']['I8-I8']][0] * results[model]['I8-I8']) / (model_method_result[functions['Conv']['GEMM']['Kernel']['Overall'][method]][0] * results[model][method]):.2f}'
            Packing: '{(functions_results[model]['I8-I8'][functions['Conv']['GEMM']['Packing']['Overall']['I8-I8']][0] * results[model]['I8-I8']) / (model_method_result[functions['Conv']['GEMM']['Packing']['Overall'][method]][0] * results[model][method]):.2f}'

"""
        except KeyError:
            print(f"[x] Skipping {model} for {method}")
            continue

detailed_breakdown_parent_path = Path(options.detailed_speedup if options.detailed_speedup else "DetailedSpeedups")
detailed_breakdown_parent_path.mkdir(exist_ok=True, parents=True)

for model in models:
    with open(join(detailed_breakdown_parent_path, f'{model}.yaml'), 'w') as f:
        f.write(detailed_speedups[model])

detailed_comparison_parent_path = Path(options.detailed_comparison if options.detailed_comparison else "DetailedComparison")
detailed_comparison_parent_path.mkdir(exist_ok=True, parents=True)

print("Generating Detailed Comparison CSVs")

detailed_comparison_data_generator_lambdas = {
    "Overall": lambda model, method: functions_results[model][method][functions['Overall'][method]][0] * results[model][method],
    "Conv": lambda model, method: functions_results[model][method][functions['Conv']['Overall'][method]][0] * results[model][method],
    "GEMM": lambda model, method: functions_results[model][method][functions['Conv']['GEMM']['Overall'][method]][0] * results[model][method],
    "Kernel": lambda model, method: functions_results[model][method][functions['Conv']['GEMM']['Kernel']['Overall'][method]][0] * results[model][method],
    "Packing": lambda model, method: functions_results[model][method][functions['Conv']['GEMM']['Packing']['Overall'][method]][0] * results[model][method],
}

for key in detailed_comparison_data_generator_lambdas.keys():
    print_as_csv_wrt_baseline(
        None, "CNN models", methods, models, 
        detailed_comparison_data_generator_lambdas[key], 'I8-I8', 
        open(join(detailed_comparison_parent_path, f"speedup-{key}.csv"), 'w'), 
        close_output_at_finish=True, data_is_callable=True
    )


print("Generating Detiled Figures TeXs")

detailed_comparison_figure_tex_parent_path = Path(join(detailed_comparison_parent_path, "Figures-Texs"))
detailed_comparison_figure_tex_parent_path.mkdir(exist_ok=True, parents=True)
detailed_comparison_figure_parent_path = Path(join(detailed_comparison_parent_path, "Figures"))
detailed_comparison_figure_parent_path.mkdir(exist_ok=True, parents=True)

detailed_comparison_data = {}
selected_methods = [
    'I8-I8',
    'No-Caching',
    'GEMMLOWP',
    'SelfDependent-W4A4',
    'ULPPACK-W4A4',
    'I4-I4',
    'BSM-W8A8',
]
selected_functions = {
    "Kernel": functions['Conv']['GEMM']['Kernel']['Overall'],
    "Packing": functions['Conv']['GEMM']['Packing']['Overall'],
    "Other in Conv": functions['Conv']['Others']['Overall'],
    "Other Layers": functions['Conv']['Overall'],
}

for model in models_names:
    with open (join(detailed_comparison_figure_tex_parent_path, f"{model}.tex"), 'w') as f:
        f.write(
            figure_stacked_ybar_tex_template.format(
                METHODS=", ".join(selected_methods),
                TITLE=f"{model} Layers Detailed Comparison",
                LEGENDS="\n                    ".join(list(map(lambda layer: f"{layer},", list(selected_functions.keys())))),
                PLOTS="\n                ".join(list(map(
                    lambda function_name: single_plot_tex_template.format(
                        COORDS=" ".join(
                            list(map(lambda method: f"({method},{functions_results[model][method][selected_functions[function_name][method]][0] * results[model][method]:.2f})" if method in functions_results[model] else f"({method},{0:.2f})", selected_methods))
                        ) if function_name != "Other Layers" else " ".join(
                            list(map(lambda method: f"({method},{(1 - functions_results[model][method][selected_functions[function_name][method]][0]) * results[model][method]:.2f})" if method in functions_results[model] else f"({method},{0:.2f})", selected_methods))
                        )
                    ),
                    list(selected_functions.keys())
                ))),
                WIDTH=float(f"{30 / len(list(selected_functions.keys())):.3f}")
            )
        )


print("Generating Figures PDFs")
for model in models_names:
    print("Running", " ".join(["pdflatex", "-shell-escape", "-interaction=nonstopmode", "-file-line-error", f"-output-directory={str(detailed_comparison_figure_parent_path)}", join(detailed_comparison_figure_tex_parent_path, f"{model}.tex")]))
    try:
        compeleted_process = subprocess.run(["pdflatex", "-shell-escape", "-interaction=nonstopmode", "-file-line-error", f"-output-directory={str(detailed_comparison_figure_parent_path)}", join(detailed_comparison_figure_tex_parent_path, f"{model}.tex")], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print("[!] stderr:", e.stderr)
        print("[!] stdout:", e.output)
        raise e

for texAuxFile in glob.iglob(os.path.join(detailed_comparison_figure_parent_path, '*.aux')):
    os.remove(texAuxFile)
for texLogFile in glob.iglob(os.path.join(detailed_comparison_figure_parent_path, '*.log')):
    os.remove(texLogFile)


