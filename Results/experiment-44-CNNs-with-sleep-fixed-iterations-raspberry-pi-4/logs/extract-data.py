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
    help="Set iteration count", default=20)
parser.add_option('-w', '--warmup-iterations',
    action="store", dest="warmup_iterations",
    help="Set iteration count", default=10)
parser.add_option('-o', '--output',
    action="store", dest="output",
    help="Set the output", default=sys.stdout)
parser.add_option('-s', '--speedups',
    action="store", dest="speedups",
    help="Enable Speedup Mode", default=None)
parser.add_option('-p', '--per-layer-breakdown',
    action="store", dest="per_layer_breakdown",
    help="Set Per Layer Breakdown saving path", default=None)

options, _ = parser.parse_args()

iterations = 1
warmup_iterations = 1
results = {}
methods = []


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

def print_as_csv_wrt_baseline(title: str, rows_name: str, column_names: list[str], row_names: list[str], data: dict[dict[str]], baseline: str, output_file = sys.stdout, close_output_at_finish = False, reverse_order = False, do_print = False):
    if title is not None:
        output_file.write("   " + title + "   " + "\n")
    output_file.write(rows_name + ",")
    for column in column_names:
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
                if reverse_order:
                    if do_print:
                        print(f"reverse_order: {reverse_order} -> data[{baseline}][{column}] / data[{row}][{column}]")
                    output_file.write("{:.2f}".format(data[baseline][column] / data[row][column]) + ",")
                else:
                    if do_print:
                        print(f"data[{row}][{baseline}] / data[{row}][{column}]")
                    output_file.write("{:.2f}".format(data[row][baseline] / data[row][column]) + ",")
            except KeyError:
                output_file.write("-,")
        output_file.write("\n")
    if close_output_at_finish:
        output_file.close()

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
        # models_names = lines[0::2]
        # models_time = list(map(lambda line: float(line), lines[1::2]))
        # models_name_time = zip(models_names, models_time)
        models_name_time = []
        name = None
        for line in lines:
            if (is_float(line)):
                models_name_time.append((
                    name, 
                    float(line),
                ))
            else:
                name = line
    for model_name_time in models_name_time:
        name, time = model_name_time
        if name in results:
            results[name][method] = time
        else:
            results[name] = { method: time }

methods = sorted(methods, key=lambda column: methods_order.index(column))
models_names = list(results.keys())
models_names_1K_imagenet = list(filter(lambda x: "-21K" not in x, models_names))
models_names_21K_imagenet = list(filter(lambda x: "-21K" in x, models_names))
if options.output != sys.stdout:
    imagenet_1K_csv_path = splitext(options.output)[0] + "-imagenet-1K" + splitext(options.output)[1]
    imagenet_21K_csv_path = splitext(options.output)[0] + "-imagenet-21K" + splitext(options.output)[1]
    print_as_csv("CNNs-ImageNet-1K", "Models", methods, models_names_1K_imagenet, results, open(imagenet_1K_csv_path, "w"), close_output_at_finish=True)
    print_as_csv("CNNs-ImageNet-21K", "Models", methods, models_names_21K_imagenet, results, open(imagenet_21K_csv_path, "w"), close_output_at_finish=True)
    if options.speedups:
        speedup_imagenet_1K_csv_path = splitext(options.speedups)[0] + "-imagenet-1K" + splitext(options.speedups)[1]
        speedup_imagenet_21K_csv_path = splitext(options.speedups)[0] + "-imagenet-21K" + splitext(options.speedups)[1]
        print_as_csv_wrt_baseline("CNNs-ImageNet-1K", "Models", methods, models_names_1K_imagenet, results, "I8-I8", open(speedup_imagenet_1K_csv_path, "w"), close_output_at_finish=True)
        print_as_csv_wrt_baseline("CNNs-ImageNet-21K", "Models", methods, models_names_21K_imagenet, results, "I8-I8", open(speedup_imagenet_21K_csv_path, "w"), close_output_at_finish=True)
else:
    print_as_csv("CNNs-ImageNet-1K", "Models", methods, models_names_1K_imagenet, results)
    print_as_csv("CNNs-ImageNet-21K", "Models", methods, models_names_21K_imagenet, results)
    print_as_csv_wrt_baseline("CNNs-ImageNet-1K", "Models", methods, models_names_1K_imagenet, results, "I8-I8", open(join("CSVs", "speedups-imagenet-1K.csv"), "w"), close_output_at_finish=True)
    print_as_csv_wrt_baseline("CNNs-ImageNet-21K", "Models", methods, models_names_21K_imagenet, results, "I8-I8", open(join("CSVs", "speedups-imagenet-21K.csv"), "w"), close_output_at_finish=True)

per_layer_breakdown_parent_path = Path(options.per_layer_breakdown if options.per_layer_breakdown else "PerLayerBreakdown")
per_layer_breakdown_parent_path.mkdir(exist_ok=True, parents=True)
per_layer_breakdown_figure_tex_parent_path = Path(join(per_layer_breakdown_parent_path, "Figures-Texs"))
per_layer_breakdown_figure_tex_parent_path.mkdir(exist_ok=True, parents=True)
per_layer_breakdown_figure_parent_path = Path(join(per_layer_breakdown_parent_path, "Figures"))
per_layer_breakdown_figure_parent_path.mkdir(exist_ok=True, parents=True)

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
single_plot_tex_template = """\\addplot coordinates {{ {COORDS} }};"""

for model in models_names:
    method_based_model_breakdown = {}
    for method in methods:
        file_path = join(method, model, f"output-{iterations}-{warmup_iterations}.log")
        with open(file_path) as f:
            lines = list(filter(lambda line: line, map(lambda line: line[:-1], f.readlines())))
        start_idx = lines.index("Operator-wise Profiling Info for Regular Benchmark Runs:") + 3
        end_idx = lines[start_idx:].index("============================== Top by Computation Time ==============================") + start_idx
        selected_lines = lines[start_idx:end_idx]
        selected_lines = list(map(lambda line: line.split(), selected_lines))
        conv_2d_operations = list(map(lambda operation: (float(operation[3]), float(operation[4][:-1])), filter(lambda line: line[0] == "CONV_2D", selected_lines)))
        fully_connected_operations = list(map(lambda operation: (float(operation[3]), float(operation[4][:-1])), filter(lambda line: line[0] == "FULLY_CONNECTED", selected_lines)))
        method_based_model_breakdown[method] = dict(
            [ 
                *list(map(lambda x: (f"CONV-{x[0] + 1}", x[1][0]), enumerate(conv_2d_operations))), 
                *list(map(lambda x: (f"FC-{x[0] + 1}", x[1][0]), enumerate(fully_connected_operations))), 
            ]
        )
    layers = [ 
        *list(map(lambda x: f"CONV-{x[0] + 1}", enumerate(conv_2d_operations))), 
        *list(map(lambda x: f"FC-{x[0] + 1}", enumerate(fully_connected_operations))), 
    ]
    figure_tex = figure_tex_template.format(
        METHODS=", ".join(methods[1:]),
        TITLE=model,
        LEGENDS="\n                    ".join(list(map(lambda layer: f"{layer},", layers))),
        PLOTS="\n                ".join(list(map(
            lambda layer: single_plot_tex_template.format(
                COORDS=" ".join(
                    list(map(lambda method: f"({method_based_model_breakdown['I8-I8'][layer] / method_based_model_breakdown[method][layer]:.2f},{method})", methods[1:]))
                )
            ),
            layers
        ))),
        WIDTH=float(f"{30 / len(layers):.3f}")
    )
    print_as_csv_wrt_baseline(
        # f"{model} Per Layer Breakdown for each method",
        None, "Methods",
        layers, methods, method_based_model_breakdown, "I8-I8",
        output_file = open(join(per_layer_breakdown_parent_path, f"{model}.csv"), "w"),
        reverse_order = True, close_output_at_finish = True
    )
    with open(join(per_layer_breakdown_figure_tex_parent_path, f"{model}.tex"), "w") as f:
        f.write(figure_tex)

print("Generating Figures PDFs")
for model in models_names:
    print("Running", " ".join(["pdflatex", "-shell-escape", "-interaction=nonstopmode", "-file-line-error", f"-output-directory={str(per_layer_breakdown_figure_parent_path)}", join(per_layer_breakdown_figure_tex_parent_path, f"{model}.tex")]))
    subprocess.run(["pdflatex", "-shell-escape", "-interaction=nonstopmode", "-file-line-error", f"-output-directory={str(per_layer_breakdown_figure_parent_path)}", join(per_layer_breakdown_figure_tex_parent_path, f"{model}.tex")], check=True, capture_output=True)

for zippath in glob.iglob(os.path.join(per_layer_breakdown_figure_parent_path, '*.aux')):
    os.remove(zippath)
for zippath in glob.iglob(os.path.join(per_layer_breakdown_figure_parent_path, '*.log')):
    os.remove(zippath)

