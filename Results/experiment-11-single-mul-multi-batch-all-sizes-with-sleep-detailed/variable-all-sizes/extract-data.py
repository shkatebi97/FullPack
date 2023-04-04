#!/usr/bin/env python
from pathlib import Path
from os.path import isfile, join, splitext, isdir
from os import listdir, times
from pprint import pprint
from tokenize import group
import xlsxwriter
import re
import optparse

parser = optparse.OptionParser()

parser.add_option('-b', '--batch-size',
    action="store", dest="batch_size",
    help="Set batch size", default=16)
parser.add_option('-n', '--iterations',
    action="store", dest="iterations",
    help="Set iteration count", default=50)

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

def save_metric_each_method_as_square_csv(csv_file_path: str, title: str, rows_name: str, column_names: list[str], x_values: list[str], y_values: list[str], c_value: int, data: dict[dict[str]]):
    with open(csv_file_path, 'w') as csv_file:
        for column_name in sorted(column_names, key=lambda column: methods_order.index(column)):
            print(title.format(column_name), file=csv_file)
            print(rows_name, end=",", file=csv_file)
            for x in x_values:
                print(x, end=",", file=csv_file)
            print(file=csv_file)

            for y in y_values:
                print(y, end=",", file=csv_file)
                for x in x_values:
                    print("{:.2f}".format(data[f"{c_value}x{x}x{y}"][column_name]), end=",", file=csv_file)
                print(file=csv_file)

def add_worksheet_to_workbook(
    workbook: object, sheet_name: str, table_names: list[str],
    x_values: list[str], y_values: list[str], c_value: int, 
    data: dict[dict[str]], extra = None
    ) -> object:
    groups = []
    output_list = []
    num_columns = len(x_values) + 1
    num_rows = len(y_values) + 1
    num_tables = len(table_names) + 1
    simple = workbook.add_format()
    simple.set_align('center')
    simple.set_align('vcenter')
    simple.set_font_family('Liberation Sans')
    simple.set_font_size('12pt')
    simple.set_font_color("")

    bold_italic = workbook.add_format()
    bold_italic.set_align('center')
    bold_italic.set_align('vcenter')
    bold_italic.set_font_family('Liberation Sans')
    bold_italic.set_font_size('12pt')
    bold_italic.set_italic(True)
    bold_italic.set_bold(True)
    bold_italic.set_font_color("")

    baseline_extra = ""

    for k, method in enumerate(table_names):
        if extra is not None:
            output_list.append([(method, bold_italic), ("", bold_italic), (method, bold_italic)])
        else:
            output_list.append([(method, bold_italic)])
        output_row = [("", bold_italic)]
        for x in x_values:
            output_row.append((x, bold_italic))
        if extra is not None:
            output_row.append(("", bold_italic))
            output_row.append(("", bold_italic))
            for x in x_values:
                output_row.append((x, bold_italic))
        output_list.append(output_row)
        for i, y in enumerate(y_values):
            if extra is not None:
                output_row = [("", bold_italic)] * (num_columns * 2 + 1)
                output_row[0] = (y, bold_italic)
                output_row[num_columns + 1] = (y, bold_italic)
            else:
                output_row = [("", bold_italic)] * (num_columns)
                output_row[0] = (y, bold_italic)
            for j, x in enumerate(x_values):
                output_row[j + 1] = (data[f"{c_value}x{x}x{y}"][method], simple)
                if extra is not None:
                    output_row[j + num_columns + 2] = (extra(k, i, j, num_tables, num_rows, num_columns), simple, True)
            output_list.append(output_row)
        output_list.append([])
        if extra is not None:
            groups.append("{0}{1}:{2}{3}".format(chr(66 + num_columns + 1), k * (num_rows + 2) + 3, chr(66 + num_columns + len(x_values)), k * (num_rows + 2) + 2 + len(y_values)))
            if k == 0:
                baseline_extra = "{0}{1}:{2}{3}".format(chr(66 + num_columns), k * (num_rows + 2) + 1, chr(66 + num_columns + len(x_values)), k * (num_rows + 2) + 2 + len(y_values))
    worksheet = workbook.add_worksheet(sheet_name)
    for i, row in enumerate(output_list):
        if len(row) == 3:
            worksheet.merge_range(i, 0, i, 8, row[0][0], row[0][1])
            worksheet.merge_range(i, 10, i, 18, row[2][0], row[2][1])
        if len(row) == 1:
            worksheet.merge_range(i, 0, i, 8, row[0][0], row[0][1])
        else:
            for j, cell in enumerate(row):
                if len(cell) == 3:
                    cell_value, cell_format, is_formula = cell
                    if is_formula:
                        worksheet.write_formula(i, j, cell_value, cell_format, "")
                    else:
                        worksheet.write(i, j, cell_value, cell_format)
                else:
                    cell_value, cell_format = cell
                    worksheet.write(i, j, cell_value, cell_format)
    worksheet.set_column('A:XFD', 10)
    return worksheet, groups, (baseline_extra, simple)

def add_colorscale_to_groups(worksheet_and_group: tuple):
    worksheet, groups, baseline_extra = worksheet_and_group
    for group in groups:
        worksheet.conditional_format(group, {'type': '3_color_scale',
                                             'mid_type': 'num',
                                             'mid_value': 0,
                                             'min_color': "#ff0000",
                                             'mid_color': "#ffff00",
                                             'max_color': "#00ff00"})
    return worksheet, groups, baseline_extra

def remove_baseline_extra(worksheet_and_group: tuple):
    worksheet, groups, (baseline_extra, baseline_extra_format) = worksheet_and_group
    if baseline_extra:
        worksheet.merge_range(baseline_extra, "Not Valid", baseline_extra_format)
    return worksheet, groups, baseline_extra

iterations = options.iterations
result_dirs = listdir(".")
result_dirs = list(filter(isdir, result_dirs))
result_dirs = list(filter(lambda x: isfile(join('.', x, f"output-{iterations}.log")), result_dirs))
result_methods_dirs = list(map(lambda x: (x, join('.', x, f"output-{iterations}.log"), join('.', x)), result_dirs))
results = {}
methods = []
for result_method_dir in result_methods_dirs:
    method, file_name, _ = result_method_dir
    methods.append(method)
    with open(file_name) as file:
        lines = list(map(lambda x: x[:-1], file.readlines()))
        models_size = list(map(lambda line: "{}x{}x{}".format(line.split("-")[0], int(line.split("batch-")[1].split("x")[0]), line.split("x")[1].split(".tflite")[0]), lines[0::2]))
        # models_size = list(filter(lambda x: batch_size == int(x.split('x')[0]), models_size))
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
# batch_size = int(sizes[0][1].split("-batch")[0])
input_sizes = sorted(list(set(list(map(lambda x: int(x[1].split("batch-")[1].split("x")[0]), sizes)))))
output_sizes = sorted(list(set(list(map(lambda x: int(x[1].split("batch-")[1].split("x")[1]), sizes)))))
methods = sorted(methods, key=lambda column: methods_order.index(column))

print_as_each_method_as_square_csv(
    "Average Run of each model for {} with " + f"{batch_size} batch size.",
    "Output Size", methods, input_sizes, output_sizes,
    batch_size, results
)

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

Path("CSVs").mkdir(exist_ok=True, parents=True)

save_metric_each_method_as_square_csv(
    "CSVs/cpu_cycles.csv", "CPU Cycles for {}",
    "Output Sizes", methods, input_sizes,
    output_sizes, batch_size, cpu_cycles
)
save_metric_each_method_as_square_csv(
    "CSVs/cpu_cycles_kernel_share.csv", "Kernels CPU Cycles Share for {}",
    "Output Sizes", methods, input_sizes,
    output_sizes, batch_size, cpu_cycles_kernel_share
)

save_metric_each_method_as_square_csv(
    "CSVs/instructions.csv", "Instructions for {}",
    "Output Sizes", methods, input_sizes,
    output_sizes, batch_size, instructions
)
save_metric_each_method_as_square_csv(
    "CSVs/instructions_kernel_share.csv", "Kernels Instructions Share for {}",
    "Output Sizes", methods, input_sizes,
    output_sizes, batch_size, instructions_kernel_share
)

save_metric_each_method_as_square_csv(
    "CSVs/l1d_loads.csv", "L1 Data Cache Loads for {}",
    "Output Sizes", methods, input_sizes,
    output_sizes, batch_size, l1d_loads
)
save_metric_each_method_as_square_csv(
    "CSVs/l1d_loads_kernel_share.csv", "Kernels L1 Data Cache Loads Share for {}",
    "Output Sizes", methods, input_sizes,
    output_sizes, batch_size, l1d_loads_kernel_share
)

save_metric_each_method_as_square_csv(
    "CSVs/l1d_misses.csv", "L1 Data Cache Misses for {}",
    "Output Sizes", methods, input_sizes,
    output_sizes, batch_size, l1d_misses
)
save_metric_each_method_as_square_csv(
    "CSVs/l1d_misses_kernel_share.csv", "Kernels L1 Data Cache Misses Share for {}",
    "Output Sizes", methods, input_sizes,
    output_sizes, batch_size, l1d_misses_kernel_share
)

save_metric_each_method_as_square_csv(
    "CSVs/total_times.csv", "Total Time for {}",
    "Output Sizes", methods, input_sizes,
    output_sizes, batch_size, total_times
)

Path("Excels").mkdir(exist_ok=True, parents=True)

workbook = xlsxwriter.Workbook(join("Excels", 'detailed.xlsx'))
# workbook.add_worksheet().set_columns

decrease_based_on_basline_str = "=ROUND((({0}{1} - {0}{2}) / {0}{1}) * 100,2)"
increase_based_on_basline_str = "=ROUND((({0}{2} - {0}{1}) / {0}{1}) * 100,2)"

decrease_kernel_share_based_on_basline_str = "=ROUND(((({0}{1} * {3}!${0}${1}) - ({0}{2} * {3}!${0}${2})) / ({0}{1} * {3}!${0}${1})) * 100,2)"
increase_kernel_share_based_on_basline_str = "=ROUND(((({0}{2} * {3}!${0}${2}) - ({0}{1} * {3}!${0}${1})) / ({0}{1} * {3}!${0}${1})) * 100,2)"


remove_baseline_extra(
    add_colorscale_to_groups(
        add_worksheet_to_workbook(workbook, "cpu_cycles", methods, input_sizes, output_sizes, batch_size, cpu_cycles, 
                            lambda k, i, j, n_t, n_r, n_c: decrease_based_on_basline_str.format(chr(66 + j).upper(), i + 3, k * (n_r + 2) + (i + 3)))
    )
)
remove_baseline_extra(
    add_colorscale_to_groups(
        add_worksheet_to_workbook(workbook, "cpu_cycles_kernel_share", methods, input_sizes, output_sizes, batch_size, cpu_cycles_kernel_share, 
                            lambda k, i, j, n_t, n_r, n_c: decrease_kernel_share_based_on_basline_str.format(chr(66 + j).upper(), i + 3, k * (n_r + 2) + (i + 3), "cpu_cycles"))
    )
)
remove_baseline_extra(
    add_colorscale_to_groups(
        add_worksheet_to_workbook(workbook, "instructions", methods, input_sizes, output_sizes, batch_size, instructions, 
                            lambda k, i, j, n_t, n_r, n_c: decrease_based_on_basline_str.format(chr(66 + j).upper(), i + 3, k * (n_r + 2) + (i + 3)))
    )
)
remove_baseline_extra(
    add_colorscale_to_groups(
        add_worksheet_to_workbook(workbook, "instructions_kernel_share", methods, input_sizes, output_sizes, batch_size, instructions_kernel_share, 
                            lambda k, i, j, n_t, n_r, n_c: decrease_kernel_share_based_on_basline_str.format(chr(66 + j).upper(), i + 3, k * (n_r + 2) + (i + 3), "instructions"))
    )
)
remove_baseline_extra(
    add_colorscale_to_groups(
        add_worksheet_to_workbook(workbook, "l1d_loads", methods, input_sizes, output_sizes, batch_size, l1d_loads, 
                            lambda k, i, j, n_t, n_r, n_c: decrease_based_on_basline_str.format(chr(66 + j).upper(), i + 3, k * (n_r + 2) + (i + 3)))
    )
)
remove_baseline_extra(
    add_colorscale_to_groups(
        add_worksheet_to_workbook(workbook, "l1d_loads_kernel_share", methods, input_sizes, output_sizes, batch_size, l1d_loads_kernel_share, 
                            lambda k, i, j, n_t, n_r, n_c: decrease_kernel_share_based_on_basline_str.format(chr(66 + j).upper(), i + 3, k * (n_r + 2) + (i + 3), "l1d_loads"))
    )
)
remove_baseline_extra(
    add_colorscale_to_groups(
        add_worksheet_to_workbook(workbook, "l1d_misses", methods, input_sizes, output_sizes, batch_size, l1d_misses, 
                            lambda k, i, j, n_t, n_r, n_c: decrease_based_on_basline_str.format(chr(66 + j).upper(), i + 3, k * (n_r + 2) + (i + 3)))
    )
)
remove_baseline_extra(
    add_colorscale_to_groups(
        add_worksheet_to_workbook(workbook, "l1d_misses_kernel_share", methods, input_sizes, output_sizes, batch_size, l1d_misses_kernel_share, 
                            lambda k, i, j, n_t, n_r, n_c: decrease_kernel_share_based_on_basline_str.format(chr(66 + j).upper(), i + 3, k * (n_r + 2) + (i + 3), "l1d_misses"))
    )
)

workbook.close()




