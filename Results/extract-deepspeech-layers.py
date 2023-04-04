#!/usr/bin/python3

from os import listdir
from os.path import join
from math import log2, log10

start_str = "Operator-wise Profiling Info for Regular Benchmark Runs:"
end_str = "============================== Top by Computation Time =============================="

groups = {
    'Layer 1': '[Relu6]',
    'Layer 2': '[Relu6_1]',
    'Layer 3': '[Relu6_2]',
    'LSTM': "cudnn_lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell",
    'Layer 5': '[Relu6_3]',
    'Layer 6': '[BiasAdd_4]',
}

methods = [
    "Binary-Binary",
    "Binary-Binary-XOR",
    "Binary-I8",
    "Eigen",
    "FP32",
    "GEMMLOWP",
    "I4-I4",
    "I4-I8",
    "I8-Binary",
    "I8-I4",
    "I8-I8",
    "XNNPack",
    "XNNPack-FP32",
    "I8-Ternary",
    "No-Caching",
    "No-Caching-FP32",
    "Ternary-I8",
    "Ternary-Ternary",
    "ULPPACK-W1A1",
    "ULPPACK-W2A2",
    "ULPPACK-W3A3",
]

selected_methods = [
    "ULPPACK-W1A1",
    "ULPPACK-W2A2",
    "ULPPACK-W3A3",
    "Binary-Binary",
    "Ternary-Ternary",
    "I4-I4",
    "I8-I8",
    "XNNPack",
    "No-Caching",
    "GEMMLOWP",
    "FP32",
    "XNNPack-FP32",
    "No-Caching-FP32",
    "Eigen",
]

methods_names = {
    "Binary-Binary": "W1A1",
    "Binary-Binary-XOR": "W1A1-XOR",
    "Binary-I8": "W8A1",
    "Eigen": "Eigen-FP32",
    "FP32": "Ruy-FP32",
    "XNNPack-FP32": "XNNPack-W8A8",
    "GEMMLOWP": "GEMMLOWP-W8A8",
    "I4-I4": "W4A4",
    "I4-I8": "W8A4",
    "I8-Binary": "W1A8",
    "I8-I4": "W4A8",
    "I8-I8": "W8A8",
    "XNNPack": "XNNPack-W8A8",
    "I8-Ternary": "W2A8",
    "No-Caching": "TFLite-W8A8",
    "No-Caching-FP32": "TFLite-FP32",
    "Ternary-I8": "W8A2",
    "Ternary-Ternary": "W2A2",
    "ULPPACK-W1A1": "ULPPACL-W1A1",
    "ULPPACK-W2A2": "ULPPACL-W2A2",
    "ULPPACK-W3A3": "ULPPACL-W3A3",
}
mode = 0
if len(list(filter(lambda x: "deepspeech-0.9.3" in x, listdir(selected_methods[0])))):
    mode = 1

if mode == 1:
    for method in selected_methods:
        model = list(filter(lambda x: "deepspeech-0.9.3" in x, listdir(method)))[0]
        file_name = "run.log"
        print(f"Running method {methods_names[method]}")
        with open(join(method, model, file_name)) as f:
            lines = f.readlines()
        lines = list(map(lambda line: line[:-1], lines))
        total_line = list(filter(lambda x: "Inference (avg):" in x, lines))
        if len(total_line) == 0:
            total_line = "Inference (avg): 0.0"
        else:
            total_line = total_line[-1]
        try:
            start_idx = lines.index(start_str) + 2
            end_idx = lines.index(end_str) + 1
        except ValueError as e:
            print("\tOperator-wise Profiling Info not found")
            print("\tTotal:   \t0.0")
            continue
        end_idx = lines[end_idx:].index(end_str) + end_idx - 1
        print("Scanning from line", start_idx, "to line", end_idx)
        data_lines = lines[start_idx:end_idx]
        total_time = float(total_line.split("Inference (avg): ")[1]) / 1000
        # data_names = ("[node type]","[start]","[first]","[avg ms]"," [%]","[cdf%]","[mem KB]","[times called]","[Name]")
        types = list(map(lambda data_line: data_line.split("\t")[1].replace(" ", ""), data_lines[1:]))
        avg_ms = list(map(lambda data_line: data_line.split("\t")[4].replace(" ", ""), data_lines[1:]))
        names = list(map(lambda data_line: data_line.split("\t")[9].replace(" ", ""), data_lines[1:]))
        desired_operations = {}
        for i, name in enumerate(names):
            if types[i] == "FULLY_CONNECTED":
                desired_operations[name] = float(avg_ms[i])
        group_vals = {}
        if len(list(desired_operations.keys())) < len(list(groups.keys())):
            print(f"Not enough FULLY_CONNECTED operations, got {len(list(desired_operations.keys()))} but expected {len(list(groups.keys()))}")
        else:
            for key in groups.keys():
                group_vals[key] = 0
            for operation_name in desired_operations.keys():
                for group in groups.keys():
                    if groups[group] in operation_name:
                        group_vals[group] += desired_operations[operation_name]
        group_vals["Total"] = total_time
        print("\t" + "\n\t".join(list(map(lambda x: f"{x}:   \t{log10(group_vals[x]):.3f}", group_vals.keys()))))
else:
    output_file_name = list(filter(lambda x: "output-" in x, listdir(selected_methods[0])))[0]
    model = open(join(selected_methods[0], output_file_name)).readline()[:-1]
    print(f"Found model {model}")
    for method in selected_methods:
        print(f"Running method {methods_names[method]}")
        with open(join(method, output_file_name)) as f:
            lines = f.readlines()
        lines = list(map(lambda line: line[:-1], lines))
        total_line = list(filter(lambda x: "Inference (avg):" in x, lines))
        if len(total_line) == 0:
            total_line = "Inference (avg): 0.0"
        else:
            total_line = total_line[-1]
        try:
            start_idx = lines.index(start_str) + 2
            end_idx = lines.index(end_str) + 1
        except ValueError as e:
            print("\tOperator-wise Profiling Info not found")
            print("\tTotal:   \t0.0")
            continue
        end_idx = lines[end_idx:].index(end_str) + end_idx - 1
        print("Scanning from line", start_idx, "to line", end_idx)
        data_lines = lines[start_idx:end_idx]
        total_time = float(total_line.split("Inference (avg): ")[1]) / 1000
        # data_names = ("[node type]","[start]","[first]","[avg ms]"," [%]","[cdf%]","[mem KB]","[times called]","[Name]")
        types = list(map(lambda data_line: data_line.split("\t")[1].replace(" ", ""), data_lines[1:]))
        avg_ms = list(map(lambda data_line: data_line.split("\t")[4].replace(" ", ""), data_lines[1:]))
        names = list(map(lambda data_line: data_line.split("\t")[9].replace(" ", ""), data_lines[1:]))
        desired_operations = {}
        for i, name in enumerate(names):
            if types[i] == "FULLY_CONNECTED":
                desired_operations[name] = float(avg_ms[i])
        group_vals = {}
        if len(list(desired_operations.keys())) < len(list(groups.keys())):
            print(f"Not enough FULLY_CONNECTED operations, got {len(list(desired_operations.keys()))} but expected {len(list(groups.keys()))}")
        else:
            for key in groups.keys():
                group_vals[key] = 0
            for operation_name in desired_operations.keys():
                for group in groups.keys():
                    if groups[group] in operation_name:
                        group_vals[group] += desired_operations[operation_name]
        group_vals["Total"] = total_time
        print("\t" + "\n\t".join(list(map(lambda x: f"{x}:   \t{log10(group_vals[x]):.3f}", group_vals.keys()))))





