#!/usr/bin/env python
from math import log10
from pathlib import Path
from os.path import isfile, join, isdir
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

parser = optparse.OptionParser()

parser.add_option('-b', '--batch-size',
    action="store", dest="batch_size",
    help="Set batch size", default=16)
parser.add_option('-n', '--iterations',
    action="store", dest="iterations",
    help="Set iteration count", default=5)
parser.add_option('-w', '--warmup-iterations',
    action="store", dest="warmup_iterations",
    help="Set iteration count", default=1)
parser.add_option('-g', '--generate-exec-time-plots',
    action="store_true", dest="generate_exec_time_plots",
    help="Generates the execution time plots", default=False)
parser.add_option('-G', '--generate-detailed-plots',
    action="store_true", dest="generate_detailed_plots",
    help="Generates detailed plots", default=False)

options, _ = parser.parse_args()

batch_size = int(options.batch_size)
iterations = 1
warmup_iterations = 1

results = {}
methods = []

extra_complex_metrics = [
    {
        "name": "IPC",
        "data_extractor": lambda x, y, c, method, k, i, j, n_t, n_r, n_c: f"=Instructions!${num_column_to_spreadsheet_letter(1 + j).upper()}${k * (n_r + 2) + (i + 3)} / CPUCycles!${num_column_to_spreadsheet_letter(1 + j).upper()}${k * (n_r + 2) + (i + 3)}",
        "extra_extractor": lambda k, i, j, n_t, n_r, n_c: increase_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), i + 3, k * (n_r + 2) + (i + 3)),
        "transform": lambda x: x,
        "transform_wrong_data": lambda x: x if x != 0 else 1
    },
    {
        "name": "Cache_Miss_Rate",
        "data_extractor": lambda x, y, c, method, k, i, j, n_t, n_r, n_c: f"=(L1DCacheMisses!${num_column_to_spreadsheet_letter(1 + j).upper()}${k * (n_r + 2) + (i + 3)} / L1DCacheAccess!${num_column_to_spreadsheet_letter(1 + j).upper()}${k * (n_r + 2) + (i + 3)}) * 100",
        "extra_extractor": lambda k, i, j, n_t, n_r, n_c: decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), i + 3, k * (n_r + 2) + (i + 3)),
        "transform": lambda x: x,
        "transform_wrong_data": lambda x: x if x != 0 else 1
    },
]


if batch_size <= 1:
    method_kernels = singlebatch_method_kernels
else:
    method_kernels = multibatch_method_kernels

def copy_format(book, fmt):
    properties = [f[4:] for f in dir(fmt) if f[0:4] == 'set_']
    dft_fmt = book.add_format()
    return book.add_format({k : v for k, v in fmt.__dict__.items() if k in properties and dft_fmt.__dict__[k] != v})

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

def add_method_as_worksheet_to_workbook(
    workbook: object, sheet_name: str, table_names: list[str],
    x_values: list[str], y_values: list[str], c_value: int, 
    data: dict[dict[dict[float | int]]] | object | FunctionType,
    transforms: list[object] | list[FunctionType] = None,
    extras: list[object] | list[FunctionType] = None, 
    transform_wrong_datas: list[object] | list[FunctionType] = None,
    highlight_area_extra: list[tuple] = None,
    border_comparison: bool = True,
    extra_is_extras: bool = False,
    ) -> tuple:
    
    transforms              = [lambda x: x]                  * len(table_names) if transforms is None               else transforms
    transform_wrong_datas   = [lambda x: x if x != 0 else 1] * len(table_names) if transform_wrong_datas is None    else transform_wrong_datas
    extras                  = [None]                         * len(table_names) if extras is None                   else extras
    
    if len(table_names) != len(extras): 
        raise ValueError("Table names must be the same length as extras while got %d != %d" % (len(table_names), len(extras)))
    if len(table_names) != len(transforms): 
        raise ValueError("Table names must be the same length as transform while got %d != %d" % (len(table_names), len(transform)))
    if len(table_names) != len(transform_wrong_datas): 
        raise ValueError("Table names must be the same length as transform_wrong_datas while got %d != %d" % (len(table_names), len(transform_wrong_datas)))

    groups = []
    extra_groups = []
    output_list = []

    num_columns = len(x_values) + 1
    num_rows = len(y_values) + 1
    num_tables = len(table_names) + 1
    num_extras = 1

    simple = workbook.add_format()
    simple.set_align('center')
    simple.set_align('vcenter')
    simple.set_font_name('FreeSans')
    simple.set_font_size('12pt')
    simple.set_font_color("")

    bold_italic = workbook.add_format()
    bold_italic.set_align('center')
    bold_italic.set_align('vcenter')
    bold_italic.set_font_name('FreeSans')
    bold_italic.set_font_size('12pt')
    bold_italic.set_italic(True)
    bold_italic.set_bold(True)
    bold_italic.set_font_color("")

    baseline_extra = ""
    method = sheet_name

    for k, table in enumerate(table_names):
        extra_title, extra, extra_value = extras[k]
        transform = transforms[k]
        transform_wrong_data = transform_wrong_datas[k]
        if extra_is_extras and extra is not None: 
            num_extras = len(extra)
        elif extra_is_extras and extra is None:
            num_extras = 0
        if extra_is_extras and extra is not None:
            output_row = []
            
            output_row.append((table, bold_italic))
            for z in range(num_extras):
                output_row.append(("", bold_italic))
                output_row.append((extra_title[z], bold_italic))

            output_list.append(output_row)
        elif extra is not None:
            output_list.append([(table, bold_italic), ("", bold_italic), (extra_title, bold_italic)])
        else:
            output_list.append([(table, bold_italic)])
        output_row = [("", bold_italic)]
        for x in x_values:
            output_row.append((x, bold_italic))
        if extra_is_extras and extra is not None:
            for z in range(num_extras):
                output_row.append(("", bold_italic))
                output_row.append(("", bold_italic))
                for x in x_values:
                    output_row.append((x, bold_italic))
        elif extra is not None:
            output_row.append(("", bold_italic))
            output_row.append(("", bold_italic))
            for x in x_values:
                output_row.append((x, bold_italic))
        output_list.append(output_row)
        for i, y in enumerate(y_values):
            if extra is not None:
                output_row = [("", bold_italic)] * ((num_columns + 1) * (num_extras + 1) + 1)
                output_row[0] = (y, bold_italic)
                if extra_is_extras:
                    for z in range(num_extras): 
                        output_row[(num_columns + 1) * (z + 1)] = (y, bold_italic)
                else:
                    output_row[num_columns + 1] = (y, bold_italic)
            else:
                output_row = [("", bold_italic)] * (num_columns)
                output_row[0] = (y, bold_italic)
            for j, x in enumerate(x_values):
                if data is not None and not callable(data):
                    this_value = data[table][f"{c_value}x{x_values[j]}x{y_values[i]}"][method]
                    next_in_row_value = data[table][f"{c_value}x{x_values[j]}x{y_values[i]}"][method] if i == len(y_values) - 1 else data[table][f"{c_value}x{x_values[j]}x{y_values[i + 1]}"][method]
                    next_in_column_value = data[table][f"{c_value}x{x_values[j]}x{y_values[i]}"][method] if j == len(x_values) - 1 else data[table][f"{c_value}x{x_values[j + 1]}x{y_values[i]}"][method]
                    this_value = transform_wrong_data(transform(this_value))
                    next_in_row_value = transform_wrong_data(transform(next_in_row_value))
                    next_in_column_value = transform_wrong_data(transform(next_in_column_value))
                    output_row[j + 1] = (
                        this_value,
                        simple,
                        False,
                        next_in_column_value - this_value if border_comparison else 0,
                        next_in_row_value - this_value if border_comparison else 0,
                    )
                elif data is not None and callable(data):
                    this_value = data[table](x, y, c_value, method, k, i, j, num_tables, num_rows, num_columns)
                    next_in_row_value = data[table](x, y, c_value, method, k, i, j, num_tables, num_rows, num_columns) if i == len(y_values) - 1 else data[table](x_values[j], y_values[i + 1], c_value, method, k, i + 1, j, num_tables, num_rows, num_columns)
                    next_in_column_value = data[table](x, y, c_value, method, k, i, j, num_tables, num_rows, num_columns) if j == len(x_values) - 1 else data[table](x_values[j + 1], y_values[i], c_value, method, k, i, j + 1, num_tables, num_rows, num_columns)
                    this_value = transform_wrong_data(transform(this_value))
                    next_in_row_value = transform_wrong_data(transform(next_in_row_value))
                    next_in_column_value = transform_wrong_data(transform(next_in_column_value))
                    output_row[j + 1] = (
                        this_value,
                        simple,
                        False,
                        next_in_column_value - this_value if border_comparison else 0,
                        next_in_row_value - this_value if border_comparison else 0,
                    )
                if extra is not None and not extra_is_extras:
                    try:
                        this_value           = float(extra_value(c_value, x_values[j], y_values[i]))
                    except ZeroDivisionError as e:
                        this_value = 0
                    try:
                        next_in_row_value    = float(extra_value(c_value, x_values[j], y_values[i]) if i == len(y_values) - 1 else extra_value(c_value, x_values[j], y_values[i + 1]))
                    except ZeroDivisionError as e:
                        next_in_row_value = 0
                    try:
                        next_in_column_value = float(extra_value(c_value, x_values[j], y_values[i]) if j == len(x_values) - 1 else extra_value(c_value, x_values[j + 1], y_values[i]))
                    except ZeroDivisionError as e:
                        next_in_column_value = 0
                    output_row[j + num_columns + 2] = (
                        extra(k, i, j, num_tables, num_rows, num_columns),
                        simple,
                        True,
                        next_in_column_value - this_value if border_comparison else 0,
                        next_in_row_value - this_value if border_comparison else 0,
                        (y_values[i], x_values[j]) in highlight_area_extra if highlight_area_extra is not None else False,
                    )
                if extra is not None and extra_is_extras:
                    for z in range(num_extras):
                        try:
                            this_value           = float(extra_value[z](c_value, x_values[j], y_values[i]))
                        except ZeroDivisionError as e:
                            this_value = 0
                        try:
                            next_in_row_value    = float(extra_value[z](c_value, x_values[j], y_values[i]) if i == len(y_values) - 1 else extra_value[z](c_value, x_values[j], y_values[i + 1]))
                        except ZeroDivisionError as e:
                            next_in_row_value = 0
                        try:
                            next_in_column_value = float(extra_value[z](c_value, x_values[j], y_values[i]) if j == len(x_values) - 1 else extra_value[z](c_value, x_values[j + 1], y_values[i]))
                        except ZeroDivisionError as e:
                            next_in_column_value = 0
                        output_row[j + (z + 1) * (num_columns + 1) + 1] = (
                            extra[z](k, i, j, num_tables, num_rows, num_columns),
                            simple,
                            True,
                            next_in_column_value - this_value if border_comparison else 0,
                            next_in_row_value - this_value if border_comparison else 0,
                            (y_values[i], x_values[j]) in highlight_area_extra if highlight_area_extra is not None else False,
                        )
            output_list.append(output_row)
        output_list.append([])
        groups.append("{0}{1}:{2}{3}".format(num_column_to_spreadsheet_letter(1), k * (num_rows + 2) + 3, num_column_to_spreadsheet_letter(1 + len(x_values) - 1), k * (num_rows + 2) + 2 + len(y_values)))
        if extra is not None:
            if not extra_is_extras:
                extra_groups.append("{0}{1}:{2}{3}".format(num_column_to_spreadsheet_letter(1 + num_columns + 1), k * (num_rows + 2) + 3, num_column_to_spreadsheet_letter(1 + num_columns + len(x_values)), k * (num_rows + 2) + 2 + len(y_values)))
            elif extra_is_extras:
                for z in range(1, num_extras + 1):
                   extra_groups.append("{0}{1}:{2}{3}".format(num_column_to_spreadsheet_letter(1 + z * (num_columns + 1)), k * (num_rows + 2) + 3, num_column_to_spreadsheet_letter(z * (num_columns + 1) + len(x_values)), k * (num_rows + 2) + 2 + len(y_values)))
    worksheet = workbook.add_worksheet(sheet_name.replace("-", "_"))
    for i, row in enumerate(output_list):
        if len(row) == 1 + 2 * num_extras:
            worksheet.merge_range(i, 0, i, len(x_values), row[0][0], row[0][1])
            for z in range(num_extras):
                worksheet.merge_range(i, (z + 1) * (len(x_values) + 2), i, (z + 1) * (len(x_values) + 2) + len(x_values), row[(z + 1) * 2][0], row[(z + 1) * 2][1])
        elif len(row) == 1:
            worksheet.merge_range(i, 0, i, len(x_values), row[0][0], row[0][1])
        else:
            for j, cell in enumerate(row):
                if len(cell) == 3:
                    cell_value, cell_format, is_formula = cell
                    if is_formula:
                        worksheet.write_formula(i, j, cell_value, cell_format, "")
                    else:
                        worksheet.write(i, j, cell_value, cell_format)
                elif len(cell) == 5:
                    cell_value, cell_format, is_formula, compare_to_next_row, compare_to_next_col = cell
                    new_cell_format = copy_format(workbook, cell_format)
                    if compare_to_next_row != 0:
                        new_cell_format.set_right(5)
                        if compare_to_next_row > 0:
                            new_cell_format.set_right_color('#1e6a39')
                        else:
                            new_cell_format.set_right_color('#bf0041')
                    if compare_to_next_col != 0:
                        new_cell_format.set_bottom(5)
                        if compare_to_next_col > 0:
                            new_cell_format.set_bottom_color('#1e6a39')
                        else:
                            new_cell_format.set_bottom_color('#bf0041')
                    if is_formula:
                        worksheet.write_formula(i, j, cell_value, new_cell_format, "")
                    else:
                        worksheet.write(i, j, cell_value, new_cell_format)
                elif len(cell) == 6:
                    cell_value, cell_format, is_formula, compare_to_next_row, compare_to_next_col, is_in_highlighted_area = cell
                    new_cell_format = copy_format(workbook, cell_format)
                    if not is_in_highlighted_area:
                        if compare_to_next_row != 0:
                            new_cell_format.set_right(5)
                            if compare_to_next_row > 0:
                                new_cell_format.set_right_color('#1e6a39')
                            else:
                                new_cell_format.set_right_color('#bf0041')
                        if compare_to_next_col != 0:
                            new_cell_format.set_bottom(5)
                            if compare_to_next_col > 0:
                                new_cell_format.set_bottom_color('#1e6a39')
                            else:
                                new_cell_format.set_bottom_color('#bf0041')
                    else:
                        new_cell_format.set_border(5)
                        new_cell_format.set_border_color('#00ffff')
                    if is_formula:
                        worksheet.write_formula(i, j, cell_value, new_cell_format, "")
                    else:
                        worksheet.write(i, j, cell_value, new_cell_format)
                else:
                    cell_value, cell_format = cell
                    worksheet.write(i, j, cell_value, cell_format)
    worksheet.set_column('A:XFD', 10)
    payload = {}
    payload["worksheet"] = worksheet
    payload["disable_max_min"] = extra_is_extras
    payload["groups"] = groups
    payload["extra_groups"] = extra_groups
    payload["baseline_extra"] = baseline_extra
    payload["baseline_extra_format"] = simple
    payload["simple_format"] = simple
    payload["bold_italic_format"] = bold_italic
    return payload

def add_worksheet_to_workbook(
    workbook: object, sheet_name: str, table_names: list[str],
    x_values: list[str], y_values: list[str], c_value: int, 
    data: dict[dict[str]] | object | FunctionType, transform = lambda x: x,
    extra = None, extra_value = None, extra_title = None, extra_is_extras = False,
    transform_wrong_data = lambda x: x if x != 0 else 1
    ) -> object:
    extra_groups = []
    output_list = []
    
    num_columns = len(x_values) + 1
    num_rows = len(y_values) + 1
    num_tables = len(table_names) + 1
    if extra_is_extras: 
        num_extras = len(extra)
    else:
        num_extras = 1

    simple = workbook.add_format()
    simple.set_align('center')
    simple.set_align('vcenter')
    simple.set_font_name('FreeSans')
    simple.set_font_size('12pt')
    simple.set_font_color("")

    bold_italic = workbook.add_format()
    bold_italic.set_align('center')
    bold_italic.set_align('vcenter')
    bold_italic.set_font_name('FreeSans')
    bold_italic.set_font_size('12pt')
    bold_italic.set_italic(True)
    bold_italic.set_bold(True)
    bold_italic.set_font_color("")

    baseline_extra = ""

    for k, method in enumerate(table_names):
        if extra_is_extras:
            output_row = []
            
            output_row.append((method, bold_italic))
            for z in range(num_extras):
                output_row.append(("", bold_italic))
                output_row.append((extra_title[z], bold_italic))

            output_list.append(output_row)
        elif extra is not None:
            if extra_title is not None:
                output_list.append([(method, bold_italic), ("", bold_italic), (extra_title, bold_italic)])
            else:
                output_list.append([(method, bold_italic), ("", bold_italic), (method, bold_italic)])
        else:
            output_list.append([(method, bold_italic)])
        output_row = [("", bold_italic)]
        for x in x_values:
            output_row.append((x, bold_italic))
        if extra_is_extras:
            for z in range(num_extras):
                output_row.append(("", bold_italic))
                output_row.append(("", bold_italic))
                for x in x_values:
                    output_row.append((x, bold_italic))
        elif extra is not None:
            output_row.append(("", bold_italic))
            output_row.append(("", bold_italic))
            for x in x_values:
                output_row.append((x, bold_italic))
        output_list.append(output_row)
        for i, y in enumerate(y_values):
            if extra is not None:
                output_row = [("", bold_italic)] * ((num_columns + 1) * (num_extras + 1) + 1)
                output_row[0] = (y, bold_italic)
                if extra_is_extras:
                    for z in range(num_extras): 
                        output_row[(num_columns + 1) * (z + 1)] = (y, bold_italic)
                else:
                    output_row[num_columns + 1] = (y, bold_italic)
            else:
                output_row = [("", bold_italic)] * (num_columns)
                output_row[0] = (y, bold_italic)
            for j, x in enumerate(x_values):
                if data is not None and not callable(data):
                    output_row[j + 1] = (transform_wrong_data(transform(data[f"{c_value}x{x}x{y}"][method])), simple)
                elif data is not None and callable(data):
                    output_row[j + 1] = (transform_wrong_data(transform(data(x, y, c_value, method, k, i, j, num_tables, num_rows, num_columns))), simple)
                if extra is not None and extra_value is None and not extra_is_extras:
                    output_row[j + num_columns + 2] = (extra(k, i, j, num_tables, num_rows, num_columns), simple, True)
                elif extra is not None and extra_value is not None and not extra_is_extras:
                    this_value           = float(extra_value(c_value, x_values[j], y_values[i], method))
                    next_in_row_value    = float(extra_value(c_value, x_values[j], y_values[i], method) if i == len(y_values) - 1 else extra_value(c_value, x_values[j], y_values[i + 1], method))
                    next_in_column_value = float(extra_value(c_value, x_values[j], y_values[i], method) if j == len(x_values) - 1 else extra_value(c_value, x_values[j + 1], y_values[i], method))
                    output_row[j + num_columns + 2] = (
                        extra(k, i, j, num_tables, num_rows, num_columns),
                        simple,
                        True,
                        next_in_column_value - this_value,
                        next_in_row_value - this_value,
                    )
                elif extra is not None and extra_value is not None and extra_is_extras:
                    for z in range(num_extras):
                        this_value           = float(extra_value[z](c_value, x_values[j], y_values[i], method))
                        next_in_row_value    = float(extra_value[z](c_value, x_values[j], y_values[i], method) if i == len(y_values) - 1 else extra_value[z](c_value, x_values[j], y_values[i + 1], method))
                        next_in_column_value = float(extra_value[z](c_value, x_values[j], y_values[i], method) if j == len(x_values) - 1 else extra_value[z](c_value, x_values[j + 1], y_values[i], method))
                        output_row[j + (z + 1) * (num_columns + 1) + 1] = (
                            extra[z](k, i, j, num_tables, num_rows, num_columns),
                            simple,
                            True,
                            next_in_column_value - this_value,
                            next_in_row_value - this_value,
                        )
                
            output_list.append(output_row)
        output_list.append([])
        if extra is not None:
            # if k == 0:
            #     baseline_extra = "{0}{1}:{2}{3}".format(num_column_to_spreadsheet_letter(1 + num_columns), k * (num_rows + 2) + 1, num_column_to_spreadsheet_letter(1 + num_columns + len(x_values)), k * (num_rows + 2) + 2 + len(y_values))
            if not extra_is_extras:
                extra_groups.append("{0}{1}:{2}{3}".format(num_column_to_spreadsheet_letter(1 + num_columns + 1), k * (num_rows + 2) + 3, num_column_to_spreadsheet_letter(1 + num_columns + len(x_values)), k * (num_rows + 2) + 2 + len(y_values)))
            elif extra_is_extras:
                for z in range(1, num_extras + 1):
                    extra_groups.append("{0}{1}:{2}{3}".format(num_column_to_spreadsheet_letter(1 + z * (num_columns + 1)), k * (num_rows + 2) + 3, num_column_to_spreadsheet_letter(z * (num_columns + 1) + len(x_values)), k * (num_rows + 2) + 2 + len(y_values)))
    worksheet = workbook.add_worksheet(sheet_name)
    for i, row in enumerate(output_list):
        if len(row) == 1 + 2 * num_extras:
            worksheet.merge_range(i, 0, i, len(x_values), row[0][0], row[0][1])
            for z in range(num_extras):
                worksheet.merge_range(i, (z + 1) * (len(x_values) + 2), i, (z + 1) * (len(x_values) + 2) + len(x_values), row[(z + 1) * 2][0], row[(z + 1) * 2][1])
        elif len(row) == 1:
            worksheet.merge_range(i, 0, i, len(x_values), row[0][0], row[0][1])
        else:
            for j, cell in enumerate(row):
                if len(cell) == 3:
                    cell_value, cell_format, is_formula = cell
                    if is_formula:
                        worksheet.write_formula(i, j, cell_value, cell_format, "")
                    else:
                        worksheet.write(i, j, cell_value, cell_format)
                elif len(cell) == 5:
                    cell_value, cell_format, is_formula, compare_to_next_row, compare_to_next_col = cell
                    new_cell_format = copy_format(workbook, cell_format)
                    if compare_to_next_row != 0:
                        new_cell_format.set_right(5)
                        if compare_to_next_row > 0:
                            new_cell_format.set_right_color('#1e6a39')
                        else:
                            new_cell_format.set_right_color('#bf0041')
                    if compare_to_next_col != 0:
                        new_cell_format.set_bottom(5)
                        if compare_to_next_col > 0:
                            new_cell_format.set_bottom_color('#1e6a39')
                        else:
                            new_cell_format.set_bottom_color('#bf0041')
                    if is_formula:
                        worksheet.write_formula(i, j, cell_value, new_cell_format, "")
                    else:
                        worksheet.write(i, j, cell_value, new_cell_format)
                else:
                    cell_value, cell_format = cell
                    try:
                        worksheet.write(i, j, cell_value, cell_format)
                    except TypeError as e:
                        print(i, j, cell_value, cell_format)
                        raise e
    worksheet.set_column('A:XFD', 10)
    payload = {}
    payload["worksheet"] = worksheet
    payload["disable_max_min"] = extra_is_extras
    payload["extra_groups"] = extra_groups
    payload["baseline_extra"] = baseline_extra
    payload["baseline_extra_format"] = simple
    payload["simple_format"] = simple
    payload["bold_italic_format"] = bold_italic
    return payload

def add_extra_min_max(payload: dict):
    worksheet, extra_groups, simple_format, bold_italic_format = payload["worksheet"], payload["extra_groups"], payload["simple_format"], payload["bold_italic_format"]
    if "disable_max_min" in payload:
        disable_max_min = payload["disable_max_min"]
    else:
        disable_max_min = False
    if not disable_max_min:
        new_groups = []
        for group in extra_groups:
            group_start_rows = int(re.findall(r'\d+', group.split(':')[0])[0])
            group_end_rows = int(re.findall(r'\d+', group.split(':')[1])[0])
            
            group_start_columns = letter_column_spreadsheet_to_number(group.split(':')[0][: -len(re.findall(r'\d+', group.split(':')[0])[0])])
            group_end_columns = letter_column_spreadsheet_to_number(group.split(':')[1][: -len(re.findall(r'\d+', group.split(':')[1])[0])])
            worksheet.write(group_start_rows - 2, group_end_columns + 2, "Maximum", bold_italic_format)
            worksheet.write(group_start_rows - 2, group_end_columns + 3, "Minimum", bold_italic_format)
            for i in range(group_start_rows - 1, group_end_rows):
                worksheet.write_formula(
                    i, group_end_columns + 2, 
                    "=MAX({0}{2}:{1}{2})".format(num_column_to_spreadsheet_letter(group_start_columns), num_column_to_spreadsheet_letter(group_end_columns), i + 1), 
                    simple_format
                )
                worksheet.write_formula(
                    i, group_end_columns + 3, 
                    "=MIN({0}{2}:{1}{2})".format(num_column_to_spreadsheet_letter(group_start_columns), num_column_to_spreadsheet_letter(group_end_columns), i + 1), 
                    simple_format
                )
            new_groups.append("{0}{1}:{0}{2}".format(num_column_to_spreadsheet_letter(group_end_columns + 2), group_start_rows, group_end_rows))
            new_groups.append("{0}{1}:{0}{2}".format(num_column_to_spreadsheet_letter(group_end_columns + 3), group_start_rows, group_end_rows))
        extra_groups.extend(new_groups)
        payload["extra_groups"] = extra_groups
    return payload

def add_colorscale_to_groups(payload: dict, fixed_range: tuple = None):
    worksheet, groups, extra_groups = payload["worksheet"], payload["groups"] if 'groups' in payload else [], payload["extra_groups"]
    for group in groups:
        if fixed_range is None:
            worksheet.conditional_format(group, 
                                                {'type': '3_color_scale',
                                                'mid_type': 'num',
                                                'mid_value': 0,
                                                'min_color': "#ff0000",
                                                'mid_color': "#ffff00",
                                                'max_color': "#00a933"})
        elif len(fixed_range) == 2:
            worksheet.conditional_format(group, 
                                                {'type': '3_color_scale',
                                                'mid_type': 'num',
                                                'max_type': 'num',
                                                'min_type': 'num',
                                                'mid_value': 0,
                                                'max_value': fixed_range[1],
                                                'min_value': fixed_range[0],
                                                'min_color': "#ff0000",
                                                'mid_color': "#ffff00",
                                                'max_color': "#00a933"})
        elif len(fixed_range) == 3:
            worksheet.conditional_format(group, 
                                                {'type': '3_color_scale',
                                                'mid_type': 'num',
                                                'max_type': 'num',
                                                'min_type': 'num',
                                                'mid_value': fixed_range[1],
                                                'max_value': fixed_range[2],
                                                'min_value': fixed_range[0],
                                                'min_color': "#ff0000",
                                                'mid_color': "#ffff00",
                                                'max_color': "#00a933"})
    for extra_group in extra_groups:
        if fixed_range is None:
            worksheet.conditional_format(extra_group, 
                                                {'type': '3_color_scale',
                                                'mid_type': 'num',
                                                'mid_value': 0,
                                                'min_color': "#ff0000",
                                                'mid_color': "#ffff00",
                                                'max_color': "#00a933"})
        elif len(fixed_range) == 2:
            worksheet.conditional_format(extra_group, 
                                                {'type': '3_color_scale',
                                                'mid_type': 'num',
                                                'max_type': 'num',
                                                'min_type': 'num',
                                                'mid_value': 0,
                                                'max_value': fixed_range[1],
                                                'min_value': fixed_range[0],
                                                'min_color': "#ff0000",
                                                'mid_color': "#ffff00",
                                                'max_color': "#00a933"})
        elif len(fixed_range) == 3:
            worksheet.conditional_format(extra_group, 
                                                {'type': '3_color_scale',
                                                'mid_type': 'num',
                                                'max_type': 'num',
                                                'min_type': 'num',
                                                'mid_value': fixed_range[1],
                                                'max_value': fixed_range[2],
                                                'min_value': fixed_range[0],
                                                'min_color': "#ff0000",
                                                'mid_color': "#ffff00",
                                                'max_color': "#00a933"})
    return payload

def remove_baseline_extra(payload: dict):
    worksheet, baseline_extra, baseline_extra_format = payload["worksheet"], payload["baseline_extra"], payload["baseline_extra_format"]
    if baseline_extra:
        worksheet.merge_range(baseline_extra, "Not Valid", baseline_extra_format)
    return payload

config_path = Path('latest_run.config')
if config_path.exists():
    with open(config_path) as config_json:
        config_str = config_json.read()
        config = json.loads(config_str)
    print("Loading from 'latest_run.config':")
    print(f"\titerations: {config['iterations']}")
    iterations = int(config["iterations"])
    print(f"\twarmup-iterations: {config['warmup_iterations']}")
    warmup_iterations = int(config["warmup_iterations"])

if options.iterations != 5:
    iterations = options.iterations
if options.warmup_iterations != 1:
    warmup_iterations = options.warmup_iterations

result_dirs = listdir(".")
result_dirs = list(filter(isdir, result_dirs))
result_dirs = list(filter(lambda x: isfile(join('.', x, f"output-{iterations}-{warmup_iterations}.log")), result_dirs))
result_methods_dirs = list(map(lambda x: (x, join('.', x, f"output-{iterations}-{warmup_iterations}.log"), join('.', x)), result_dirs))

for result_method_dir in result_methods_dirs:
    method, file_name, _ = result_method_dir
    methods.append(method)
    with open(file_name) as file:
        lines = list(map(lambda x: x[:-1], file.readlines()))
        models_size = list(map(lambda line: "{}x{}x{}".format(line.split("-")[0], int(line.split("batch-")[1].split("x")[0]), line.split("x")[1].split(".tflite")[0]), lines[0::2]))
        models_time = list(map(lambda line: float(line), lines[1::2]))
        models_size_time = zip(models_size, models_time)
    for model_size_time in models_size_time:
        size, time = model_size_time
        if size in results:
            results[size][method] = time
        else:
            results[size] = { method: time }

sizes = results.keys()
sizes = list(map(lambda x: (x, "{}-batch-{}x{}".format(x.split('x')[0], int(x.split('x')[1]), x.split('x')[2])), sizes))
input_sizes = sorted(list(set(list(map(lambda x: int(x[0].split("x")[1]), sizes)))))
output_sizes = sorted(list(set(list(map(lambda x: int(x[0].split("x")[2]), sizes)))))
methods = sorted(methods, key=lambda column: methods_order.index(column))

Path("Excels").mkdir(exist_ok=True, parents=True)

workbook = xlsxwriter.Workbook(join("Excels", f'{Path.cwd().__str__().split("/")[-2]}-results.xlsx'))

extras = [
    lambda k, i, j, n_t, n_r, n_c: decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), methods_order.index("I8-I8") * (n_r + 2) + (i + 3), k * (n_r + 2) + (i + 3)),
    lambda k, i, j, n_t, n_r, n_c: decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), methods_order.index("XNNPack") * (n_r + 2) + (i + 3), k * (n_r + 2) + (i + 3)),
    lambda k, i, j, n_t, n_r, n_c: decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), methods_order.index("No-Caching") * (n_r + 2) + (i + 3), k * (n_r + 2) + (i + 3)),
    lambda k, i, j, n_t, n_r, n_c: decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), methods_order.index("GEMMLOWP") * (n_r + 2) + (i + 3), k * (n_r + 2) + (i + 3)),
    lambda k, i, j, n_t, n_r, n_c: decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), methods_order.index("FP32") * (n_r + 2) + (i + 3), k * (n_r + 2) + (i + 3)),
    lambda k, i, j, n_t, n_r, n_c: decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), methods_order.index("No-Caching-FP32") * (n_r + 2) + (i + 3), k * (n_r + 2) + (i + 3)),
    lambda k, i, j, n_t, n_r, n_c: decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), methods_order.index("XNNPack-FP32") * (n_r + 2) + (i + 3), k * (n_r + 2) + (i + 3)),
    lambda k, i, j, n_t, n_r, n_c: decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), methods_order.index("Eigen") * (n_r + 2) + (i + 3), k * (n_r + 2) + (i + 3)),
]
extras_values = [
    lambda c, x, y, method: method_based_decrease_based_on_basline_val(results, c, x, y, method, 'I8-I8'),
    lambda c, x, y, method: method_based_decrease_based_on_basline_val(results, c, x, y, method, 'XNNPack'),
    lambda c, x, y, method: method_based_decrease_based_on_basline_val(results, c, x, y, method, 'No-Caching'),
    lambda c, x, y, method: method_based_decrease_based_on_basline_val(results, c, x, y, method, 'GEMMLOWP'),
    lambda c, x, y, method: method_based_decrease_based_on_basline_val(results, c, x, y, method, 'FP32'),
    lambda c, x, y, method: method_based_decrease_based_on_basline_val(results, c, x, y, method, 'No-Caching-FP32'),
    lambda c, x, y, method: method_based_decrease_based_on_basline_val(results, c, x, y, method, 'XNNPack-FP32'),
    lambda c, x, y, method: method_based_decrease_based_on_basline_val(results, c, x, y, method, 'Eigen'),
]
extras_titles = [
    "Speed-up against I8-I8",
    "Speed-up against XNNPack",
    "Speed-up against No-Caching",
    "Speed-up against GEMMLOWP",
    "Speed-up against FP32",
    "Speed-up against No-Caching-FP32",
    "Speed-up against XNNPack-FP32",
    "Speed-up against Eigen",
]

print("Generating {}".format(join("Excels", f'{Path.cwd().__str__().split("/")[-2]}-results.xlsx')))

remove_baseline_extra(
    add_colorscale_to_groups(
        add_extra_min_max(
            add_worksheet_to_workbook(
                workbook, "Variable-Sizes", methods, input_sizes, output_sizes, batch_size, results, 
                extra=extras, extra_value=extras_values, extra_title=extras_titles, extra_is_extras=True,
                transform=lambda x: x
            )
        )
    )
)

workbook.close()

# exit(0)

metrics_values = {}

metrics_values['CPUCycles'] = {}
metrics_values['Instructions'] = {}
metrics_values['IPC'] = {}

metrics_values['L1DCacheAccess'] = {}
metrics_values['L1DCacheMisses'] = {}
metrics_values['L1DCacheMissRate'] = {}

metrics_values['L1ICacheAccess'] = {}
metrics_values['L1ICacheMisses'] = {}
metrics_values['L1ICacheMissRate'] = {}

metrics_values['L2DCacheAccess'] = {}
metrics_values['L2DCacheMisses'] = {}
metrics_values['L2DCacheMissRate'] = {}
metrics_values['L2DCacheMissLatency'] = {}

metrics_values['L2ICacheAccess'] = {}
metrics_values['L2ICacheMisses'] = {}
metrics_values['L2ICacheMissRate'] = {}

metrics_values['LLDCacheAccess'] = {}
metrics_values['LLDCacheMisses'] = {}
metrics_values['LLDCacheMissRate'] = {}
metrics_values['LLDCacheMissLatency'] = {}

metrics_values['LLICacheAccess'] = {}
metrics_values['LLICacheMisses'] = {}
metrics_values['LLICacheMissRate'] = {}

metrics_values['TotalTime'] = {}

for result_method_dir in result_methods_dirs:
    method, file_name, dir_name = result_method_dir
    for size_tuple in sizes:
        size, size_string = size_tuple

        with open(join(dir_name, f"model-{size_string}", "stats.txt")) as stat_f:
            # Remove First line containing nothing important
            stat_lines = stat_f.readlines()[1:]
            # Select Only Main Inference
            indices = [i for i, x in enumerate(stat_lines) if x == "---------- Begin Simulation Statistics ----------\n"]
            # Seperating and removing other lines
            stat_lines = stat_lines[indices[1] + 1:indices[2] - 3]
            # Remove the end of line form each line
            stat_lines = list(map(lambda x: x[:-1], stat_lines))
            # Remove extra non important data and converting each value to float
            stat_lines = list(map(lambda x: (x.split()[0], float(x.split()[1])), stat_lines))
            # Convert to dictionary
            stats = dict(stat_lines)
            # Extracting values from the dictionary
            cpu_total           = stats['system.switch_cpus.numCycles'] if 'system.switch_cpus.numCycles' in stats else 0 # 1
            instruction_total   = stats['system.switch_cpus.committedInsts'] if 'system.switch_cpus.committedInsts' in stats else 0 # 1
            ipc                 = stats['system.switch_cpus.ipc'] if 'system.switch_cpus.ipc' in stats else 0 # 0.001

            l1d_miss_total      = stats['system.cpu.dcache.ReadReq.misses::switch_cpus.data'] if 'system.cpu.dcache.ReadReq.misses::switch_cpus.data' in stats else 0 # 1
            l1d_load_total      = stats['system.cpu.dcache.ReadReq.accesses::switch_cpus.data'] if 'system.cpu.dcache.ReadReq.accesses::switch_cpus.data' in stats else 0 # 1
            L1DCacheMissRate    = stats['system.cpu.dcache.ReadReq.missRate::switch_cpus.data'] if 'system.cpu.dcache.ReadReq.missRate::switch_cpus.data' in stats else 0 # 0.001

            l1i_miss_total      = stats['system.cpu.icache.ReadReq.misses::switch_cpus.inst'] if 'system.cpu.icache.ReadReq.misses::switch_cpus.inst' in stats else 0 # 1
            l1i_load_total      = stats['system.cpu.icache.ReadReq.accesses::switch_cpus.inst'] if 'system.cpu.icache.ReadReq.accesses::switch_cpus.inst' in stats else 0 # 1
            L1ICacheMissRate    = stats['system.cpu.icache.ReadReq.missRate::switch_cpus.inst'] if 'system.cpu.icache.ReadReq.missRate::switch_cpus.inst' in stats else 0 # 0.001

            L2DCacheAccess      = stats['system.l2.overallAccesses::switch_cpus.data'] if 'system.l2.overallAccesses::switch_cpus.data' in stats else 0 # 1
            L2DCacheMisses      = stats['system.l2.overallMisses::switch_cpus.data'] if 'system.l2.overallMisses::switch_cpus.data' in stats else 0 # 1
            L2DCacheMissRate    = stats['system.l2.overallMissRate::switch_cpus.data'] if 'system.l2.overallMissRate::switch_cpus.data' in stats else 0 # 0.001
            L2DCacheMissLatency = stats['system.l2.overallMissLatency::switch_cpus.data'] if 'system.l2.overallMissLatency::switch_cpus.data' in stats else 0 # 0.001

            L2ICacheAccess      = stats['system.l2.overallAccesses::switch_cpus.inst'] if 'system.l2.overallAccesses::switch_cpus.inst' in stats else 0 # 1
            L2ICacheMisses      = stats['system.l2.overallMisses::switch_cpus.inst'] if 'system.l2.overallMisses::switch_cpus.inst' in stats else 0 # 1
            L2ICacheMissRate    = stats['system.l2.overallMissRate::switch_cpus.inst'] if 'system.l2.overallMissRate::switch_cpus.inst' in stats else 0 # 0.001

            LLDCacheAccess      = stats['system.l3.overallAccesses::switch_cpus.data'] if 'system.l3.overallAccesses::switch_cpus.data' in stats else 0 # 1
            LLDCacheMisses      = stats['system.l3.overallMisses::switch_cpus.data'] if 'system.l3.overallMisses::switch_cpus.data' in stats else 0 # 1
            LLDCacheMissRate    = stats['system.l3.overallMissRate::switch_cpus.data'] if 'system.l3.overallMissRate::switch_cpus.data' in stats else 0 # 0.001
            LLDCacheMissLatency = stats['system.l3.overallMissLatency::switch_cpus.data'] if 'system.l3.overallMissLatency::switch_cpus.data' in stats else 0 # 0.001

            LLICacheAccess      = stats['system.l3.overallAccesses::switch_cpus.inst'] if 'system.l3.overallAccesses::switch_cpus.inst' in stats else 0 # 1
            LLICacheMisses      = stats['system.l3.overallMisses::switch_cpus.inst'] if 'system.l3.overallMisses::switch_cpus.inst' in stats else 0 # 1
            LLICacheMissRate    = stats['system.l3.overallMissRate::switch_cpus.inst'] if 'system.l3.overallMissRate::switch_cpus.inst' in stats else 0 # 0.001

            Total_Time          = stats['simSeconds'] if 'simSeconds' in stats else 0 # 0.001
        
        if size in metrics_values['CPUCycles']:
            metrics_values['CPUCycles'][size][method]           = cpu_total
            metrics_values['Instructions'][size][method]        = instruction_total
            metrics_values['IPC'][size][method]                 = ipc

            metrics_values['L1DCacheMisses'][size][method]      = l1d_miss_total
            metrics_values['L1DCacheAccess'][size][method]      = l1d_load_total
            metrics_values['L1DCacheMissRate'][size][method]    = L1DCacheMissRate

            metrics_values['L1ICacheMisses'][size][method]      = l1i_miss_total
            metrics_values['L1ICacheAccess'][size][method]      = l1i_load_total
            metrics_values['L1ICacheMissRate'][size][method]    = L1ICacheMissRate

            metrics_values['L2DCacheAccess'][size][method]      = L2DCacheAccess
            metrics_values['L2DCacheMisses'][size][method]      = L2DCacheMisses
            metrics_values['L2DCacheMissRate'][size][method]    = L2DCacheMissRate
            metrics_values['L2DCacheMissLatency'][size][method] = L2DCacheMissLatency

            metrics_values['L2ICacheAccess'][size][method]      = L2ICacheAccess
            metrics_values['L2ICacheMisses'][size][method]      = L2ICacheMisses
            metrics_values['L2ICacheMissRate'][size][method]    = L2ICacheMissRate

            metrics_values['LLDCacheAccess'][size][method]      = LLDCacheAccess
            metrics_values['LLDCacheMisses'][size][method]      = LLDCacheMisses
            metrics_values['LLDCacheMissRate'][size][method]    = LLDCacheMissRate
            metrics_values['LLDCacheMissLatency'][size][method] = LLDCacheMissLatency

            metrics_values['LLICacheAccess'][size][method]      = LLICacheAccess
            metrics_values['LLICacheMisses'][size][method]      = LLICacheMisses
            metrics_values['LLICacheMissRate'][size][method]    = LLICacheMissRate

            metrics_values['TotalTime'][size][method]           = Total_Time
        else:
            metrics_values['CPUCycles'][size]                   = { method: cpu_total }
            metrics_values['Instructions'][size]                = { method: instruction_total }
            metrics_values['IPC'][size]                         = { method: ipc }

            metrics_values['L1DCacheMisses'][size]              = { method: l1d_miss_total }
            metrics_values['L1DCacheAccess'][size]              = { method: l1d_load_total }
            metrics_values['L1DCacheMissRate'][size]            = { method: L1DCacheMissRate }

            metrics_values['L1ICacheMisses'][size]              = { method: l1i_miss_total }
            metrics_values['L1ICacheAccess'][size]              = { method: l1i_load_total }
            metrics_values['L1ICacheMissRate'][size]            = { method: L1ICacheMissRate }

            metrics_values['L2DCacheAccess'][size]              = { method: L2DCacheAccess }
            metrics_values['L2DCacheMisses'][size]              = { method: L2DCacheMisses }
            metrics_values['L2DCacheMissRate'][size]            = { method: L2DCacheMissRate }
            metrics_values['L2DCacheMissLatency'][size]         = { method: L2DCacheMissLatency }

            metrics_values['L2ICacheAccess'][size]              = { method: L2ICacheAccess }
            metrics_values['L2ICacheMisses'][size]              = { method: L2ICacheMisses }
            metrics_values['L2ICacheMissRate'][size]            = { method: L2ICacheMissRate }

            metrics_values['LLDCacheAccess'][size]              = { method: LLDCacheAccess }
            metrics_values['LLDCacheMisses'][size]              = { method: LLDCacheMisses }
            metrics_values['LLDCacheMissRate'][size]            = { method: LLDCacheMissRate }
            metrics_values['LLDCacheMissLatency'][size]         = { method: LLDCacheMissLatency }

            metrics_values['LLICacheAccess'][size]              = { method: LLICacheAccess }
            metrics_values['LLICacheMisses'][size]              = { method: LLICacheMisses }
            metrics_values['LLICacheMissRate'][size]            = { method: LLICacheMissRate }

            metrics_values['TotalTime'][size]                   = { method: Total_Time }

Path("Excels").mkdir(exist_ok=True, parents=True)

workbook = xlsxwriter.Workbook(join("Excels", f'{Path.cwd().__str__().split("/")[-2]}-cpu-results.xlsx'))

extras = [
    lambda k, i, j, n_t, n_r, n_c: decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), methods_order.index("I8-I8") * (n_r + 2) + (i + 3), k * (n_r + 2) + (i + 3)),
    lambda k, i, j, n_t, n_r, n_c: decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), methods_order.index("XNNPack") * (n_r + 2) + (i + 3), k * (n_r + 2) + (i + 3)),
    lambda k, i, j, n_t, n_r, n_c: decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), methods_order.index("No-Caching") * (n_r + 2) + (i + 3), k * (n_r + 2) + (i + 3)),
    lambda k, i, j, n_t, n_r, n_c: decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), methods_order.index("GEMMLOWP") * (n_r + 2) + (i + 3), k * (n_r + 2) + (i + 3)),
    lambda k, i, j, n_t, n_r, n_c: decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), methods_order.index("FP32") * (n_r + 2) + (i + 3), k * (n_r + 2) + (i + 3)),
    lambda k, i, j, n_t, n_r, n_c: decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), methods_order.index("No-Caching-FP32") * (n_r + 2) + (i + 3), k * (n_r + 2) + (i + 3)),
    lambda k, i, j, n_t, n_r, n_c: decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), methods_order.index("XNNPack-FP32") * (n_r + 2) + (i + 3), k * (n_r + 2) + (i + 3)),
    lambda k, i, j, n_t, n_r, n_c: decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), methods_order.index("Eigen") * (n_r + 2) + (i + 3), k * (n_r + 2) + (i + 3)),
]
extras_values = [
    lambda c, x, y, method: method_based_decrease_based_on_basline_val(metrics_values['CPUCycles'], c, x, y, method, 'I8-I8'),
    lambda c, x, y, method: method_based_decrease_based_on_basline_val(metrics_values['CPUCycles'], c, x, y, method, 'XNNPack'),
    lambda c, x, y, method: method_based_decrease_based_on_basline_val(metrics_values['CPUCycles'], c, x, y, method, 'No-Caching'),
    lambda c, x, y, method: method_based_decrease_based_on_basline_val(metrics_values['CPUCycles'], c, x, y, method, 'GEMMLOWP'),
    lambda c, x, y, method: method_based_decrease_based_on_basline_val(metrics_values['CPUCycles'], c, x, y, method, 'FP32'),
    lambda c, x, y, method: method_based_decrease_based_on_basline_val(metrics_values['CPUCycles'], c, x, y, method, 'No-Caching-FP32'),
    lambda c, x, y, method: method_based_decrease_based_on_basline_val(metrics_values['CPUCycles'], c, x, y, method, 'XNNPack-FP32'),
    lambda c, x, y, method: method_based_decrease_based_on_basline_val(metrics_values['CPUCycles'], c, x, y, method, 'Eigen'),
]
extras_titles = [
    "Speed-up against I8-I8",
    "Speed-up against XNNPack",
    "Speed-up against No-Caching",
    "Speed-up against GEMMLOWP",
    "Speed-up against FP32",
    "Speed-up against No-Caching-FP32",
    "Speed-up against XNNPack-FP32",
    "Speed-up against Eigen",
]

print("Generating {}".format(join("Excels", f'{Path.cwd().__str__().split("/")[-2]}-cpu-results.xlsx')))

remove_baseline_extra(
    add_colorscale_to_groups(
        add_extra_min_max(
            add_worksheet_to_workbook(
                workbook, "Variable-Sizes", methods, input_sizes, output_sizes, batch_size, metrics_values['CPUCycles'], 
                extra=extras, extra_value=extras_values, extra_title=extras_titles, extra_is_extras=True,
                transform=lambda x: x
            )
        )
    )
)

workbook.close()

workbook = xlsxwriter.Workbook(join("Excels", f'{Path.cwd().__str__().split("/")[-2]}-detailed-method-based.xlsx'))

print("Generating {}".format(join("Excels", f'{Path.cwd().__str__().split("/")[-2]}-detailed-method-based.xlsx')))

for method in methods:
    metrics = list(metrics_values.keys())
    metrics = sorted(metrics, key=lambda metric: metrics_sorted.index(metric))
    extras = []
    transforms = []
    transform_wrong_datas = []
    # highlight_area = [ (i, j) for i in [512, 2048, 4096, 8192] for j in [512, 2048, 4096, 8192] ]
    idx_to_value = [32, 128, 512, 2048, 4096, 8192]
    highlight_area = [
        (idx_to_value[2], idx_to_value[4]),
        (idx_to_value[2], idx_to_value[5]),
        (idx_to_value[3], idx_to_value[3]),
        (idx_to_value[4], idx_to_value[2]),
        (idx_to_value[5], idx_to_value[2]),
    ]
    for metric in metrics:
        if 'CPUCycles' == metric:
            extras.append((
                [
                    'CPUCycles reduction against I8-I8',
                    'CPUCycles reduction against XNNPack',
                    'CPUCycles reduction against No-Caching',
                    'CPUCycles reduction against GEMMLOWP',
                    'CPUCycles reduction against FP32',
                    'CPUCycles reduction against No-Caching-FP32',
                    'CPUCycles reduction against XNNPack-FP32',
                    'CPUCycles reduction against Eigen',
                ], 
                [
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("CPUCycles") * (n_r + 2) + (i + 3), metrics_sorted.index("CPUCycles") * (n_r + 2) + (i + 3), 'I8_I8'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("CPUCycles") * (n_r + 2) + (i + 3), metrics_sorted.index("CPUCycles") * (n_r + 2) + (i + 3), 'XNNPack'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("CPUCycles") * (n_r + 2) + (i + 3), metrics_sorted.index("CPUCycles") * (n_r + 2) + (i + 3), 'No_Caching'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("CPUCycles") * (n_r + 2) + (i + 3), metrics_sorted.index("CPUCycles") * (n_r + 2) + (i + 3), 'GEMMLOWP'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("CPUCycles") * (n_r + 2) + (i + 3), metrics_sorted.index("CPUCycles") * (n_r + 2) + (i + 3), 'FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("CPUCycles") * (n_r + 2) + (i + 3), metrics_sorted.index("CPUCycles") * (n_r + 2) + (i + 3), 'No_Caching_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("CPUCycles") * (n_r + 2) + (i + 3), metrics_sorted.index("CPUCycles") * (n_r + 2) + (i + 3), 'XNNPack_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("CPUCycles") * (n_r + 2) + (i + 3), metrics_sorted.index("CPUCycles") * (n_r + 2) + (i + 3), 'Eigen'),
                ],
                [
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['CPUCycles'], c, x, y, method, 'I8-I8'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['CPUCycles'], c, x, y, method, 'XNNPack'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['CPUCycles'], c, x, y, method, 'No-Caching'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['CPUCycles'], c, x, y, method, 'GEMMLOWP'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['CPUCycles'], c, x, y, method, 'FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['CPUCycles'], c, x, y, method, 'No-Caching-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['CPUCycles'], c, x, y, method, 'XNNPack-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['CPUCycles'], c, x, y, method, 'Eigen'),
                ],
            ))
            transforms.append(lambda x: x / (iterations + warmup_iterations))
            transform_wrong_datas.append(lambda x: x)# lambda x: x if x != 0 else 1)
        elif 'Instructions' == metric:
            extras.append((
                [
                    'Instructions reduction against I8-I8',
                    'Instructions reduction against XNNPack',
                    'Instructions reduction against No-Caching',
                    'Instructions reduction against GEMMLOWP',
                    'Instructions reduction against FP32',
                    'Instructions reduction against No-Caching-FP32',
                    'Instructions reduction against XNNPack-FP32',
                    'Instructions reduction against Eigen',
                ], 
                [
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("Instructions") * (n_r + 2) + (i + 3), metrics_sorted.index("Instructions") * (n_r + 2) + (i + 3), 'I8_I8'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("Instructions") * (n_r + 2) + (i + 3), metrics_sorted.index("Instructions") * (n_r + 2) + (i + 3), 'XNNPack'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("Instructions") * (n_r + 2) + (i + 3), metrics_sorted.index("Instructions") * (n_r + 2) + (i + 3), 'No_Caching'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("Instructions") * (n_r + 2) + (i + 3), metrics_sorted.index("Instructions") * (n_r + 2) + (i + 3), 'GEMMLOWP'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("Instructions") * (n_r + 2) + (i + 3), metrics_sorted.index("Instructions") * (n_r + 2) + (i + 3), 'FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("Instructions") * (n_r + 2) + (i + 3), metrics_sorted.index("Instructions") * (n_r + 2) + (i + 3), 'No_Caching_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("Instructions") * (n_r + 2) + (i + 3), metrics_sorted.index("Instructions") * (n_r + 2) + (i + 3), 'XNNPack_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("Instructions") * (n_r + 2) + (i + 3), metrics_sorted.index("Instructions") * (n_r + 2) + (i + 3), 'Eigen'),
                ],
                [
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['Instructions'], c, x, y, method, 'I8-I8'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['Instructions'], c, x, y, method, 'XNNPack'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['Instructions'], c, x, y, method, 'No-Caching'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['Instructions'], c, x, y, method, 'GEMMLOWP'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['Instructions'], c, x, y, method, 'FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['Instructions'], c, x, y, method, 'No-Caching-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['Instructions'], c, x, y, method, 'XNNPack-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['Instructions'], c, x, y, method, 'Eigen'),
                ],
            ))
            transforms.append(lambda x: x)
            transform_wrong_datas.append(lambda x: x)# lambda x: x if x != 0 else 1)
        elif 'IPC' == metric:
            extras.append((
                [
                    'IPC increase against I8-I8',
                    'IPC increase against XNNPack',
                    'IPC increase against No-Caching',
                    'IPC increase against GEMMLOWP',
                    'IPC increase against FP32',
                    'IPC increase against No-Caching-FP32',
                    'IPC increase against XNNPack-FP32',
                    'IPC increase against Eigen',
                ], 
                [
                    lambda k, i, j, n_t, n_r, n_c: method_based_increase_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("IPC") * (n_r + 2) + (i + 3), metrics_sorted.index("IPC") * (n_r + 2) + (i + 3), 'I8_I8'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_increase_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("IPC") * (n_r + 2) + (i + 3), metrics_sorted.index("IPC") * (n_r + 2) + (i + 3), 'XNNPack'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_increase_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("IPC") * (n_r + 2) + (i + 3), metrics_sorted.index("IPC") * (n_r + 2) + (i + 3), 'No_Caching'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_increase_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("IPC") * (n_r + 2) + (i + 3), metrics_sorted.index("IPC") * (n_r + 2) + (i + 3), 'GEMMLOWP'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_increase_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("IPC") * (n_r + 2) + (i + 3), metrics_sorted.index("IPC") * (n_r + 2) + (i + 3), 'FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_increase_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("IPC") * (n_r + 2) + (i + 3), metrics_sorted.index("IPC") * (n_r + 2) + (i + 3), 'No_Caching_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_increase_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("IPC") * (n_r + 2) + (i + 3), metrics_sorted.index("IPC") * (n_r + 2) + (i + 3), 'XNNPack_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_increase_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("IPC") * (n_r + 2) + (i + 3), metrics_sorted.index("IPC") * (n_r + 2) + (i + 3), 'Eigen'),
                ],
                [
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['IPC'], c, x, y, method, 'I8-I8'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['IPC'], c, x, y, method, 'XNNPack'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['IPC'], c, x, y, method, 'No-Caching'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['IPC'], c, x, y, method, 'GEMMLOWP'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['IPC'], c, x, y, method, 'FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['IPC'], c, x, y, method, 'No-Caching-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['IPC'], c, x, y, method, 'XNNPack-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['IPC'], c, x, y, method, 'Eigen'),
                ],
            ))
            transforms.append(lambda x: x)
            transform_wrong_datas.append(lambda x: x)# lambda x: x if x != 0 else 1)
        elif 'L1DCacheAccess' == metric:
            extras.append((
                [
                    'L1 Data Cache Loads reduction against I8-I8',
                    'L1 Data Cache Loads reduction against XNNPack',
                    'L1 Data Cache Loads reduction against No-Caching',
                    'L1 Data Cache Loads reduction against GEMMLOWP',
                    'L1 Data Cache Loads reduction against FP32',
                    'L1 Data Cache Loads reduction against No-Caching-FP32',
                    'L1 Data Cache Loads reduction against XNNPack-FP32',
                    'L1 Data Cache Loads reduction against Eigen',
                ], 
                [
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1DCacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("L1DCacheAccess") * (n_r + 2) + (i + 3), 'I8_I8'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1DCacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("L1DCacheAccess") * (n_r + 2) + (i + 3), 'XNNPack'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1DCacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("L1DCacheAccess") * (n_r + 2) + (i + 3), 'No_Caching'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1DCacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("L1DCacheAccess") * (n_r + 2) + (i + 3), 'GEMMLOWP'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1DCacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("L1DCacheAccess") * (n_r + 2) + (i + 3), 'FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1DCacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("L1DCacheAccess") * (n_r + 2) + (i + 3), 'No_Caching_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1DCacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("L1DCacheAccess") * (n_r + 2) + (i + 3), 'XNNPack_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1DCacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("L1DCacheAccess") * (n_r + 2) + (i + 3), 'Eigen'),
                ],
                [
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1DCacheAccess'], c, x, y, method, 'I8-I8'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1DCacheAccess'], c, x, y, method, 'XNNPack'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1DCacheAccess'], c, x, y, method, 'No-Caching'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1DCacheAccess'], c, x, y, method, 'GEMMLOWP'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1DCacheAccess'], c, x, y, method, 'FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1DCacheAccess'], c, x, y, method, 'No-Caching-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1DCacheAccess'], c, x, y, method, 'XNNPack-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1DCacheAccess'], c, x, y, method, 'Eigen'),
                ],
            ))
            transforms.append(lambda x: x)
            transform_wrong_datas.append(lambda x: x)# lambda x: x if x != 0 else 1)
        elif 'L1DCacheMisses' == metric:
            extras.append((
                [
                    'L1 Data Cache Load Misses reduction against I8-I8',
                    'L1 Data Cache Load Misses reduction against XNNPack',
                    'L1 Data Cache Load Misses reduction against No-Caching',
                    'L1 Data Cache Load Misses reduction against GEMMLOWP',
                    'L1 Data Cache Load Misses reduction against FP32',
                    'L1 Data Cache Load Misses reduction against No-Caching-FP32',
                    'L1 Data Cache Load Misses reduction against XNNPack-FP32',
                    'L1 Data Cache Load Misses reduction against Eigen',
                ], 
                [
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1DCacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("L1DCacheMisses") * (n_r + 2) + (i + 3), 'I8_I8'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1DCacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("L1DCacheMisses") * (n_r + 2) + (i + 3), 'XNNPack'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1DCacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("L1DCacheMisses") * (n_r + 2) + (i + 3), 'No_Caching'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1DCacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("L1DCacheMisses") * (n_r + 2) + (i + 3), 'GEMMLOWP'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1DCacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("L1DCacheMisses") * (n_r + 2) + (i + 3), 'FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1DCacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("L1DCacheMisses") * (n_r + 2) + (i + 3), 'No_Caching_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1DCacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("L1DCacheMisses") * (n_r + 2) + (i + 3), 'XNNPack_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1DCacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("L1DCacheMisses") * (n_r + 2) + (i + 3), 'Eigen'),
                ],
                [
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1DCacheMisses'], c, x, y, method, 'I8-I8'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1DCacheMisses'], c, x, y, method, 'XNNPack'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1DCacheMisses'], c, x, y, method, 'No-Caching'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1DCacheMisses'], c, x, y, method, 'GEMMLOWP'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1DCacheMisses'], c, x, y, method, 'FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1DCacheMisses'], c, x, y, method, 'No-Caching-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1DCacheMisses'], c, x, y, method, 'XNNPack-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1DCacheMisses'], c, x, y, method, 'Eigen'),
                ],
            ))
            transforms.append(lambda x: x)
            transform_wrong_datas.append(lambda x: x)# lambda x: x if x != 0 else 1)
        elif 'L1DCacheMissRate' == metric:
            extras.append((
                [
                    'L1 Data Cache Miss Rate reduction against I8-I8',
                    'L1 Data Cache Miss Rate reduction against XNNPack',
                    'L1 Data Cache Miss Rate reduction against No-Caching',
                    'L1 Data Cache Miss Rate reduction against GEMMLOWP',
                    'L1 Data Cache Miss Rate reduction against FP32',
                    'L1 Data Cache Miss Rate reduction against No-Caching-FP32',
                    'L1 Data Cache Miss Rate reduction against XNNPack-FP32',
                    'L1 Data Cache Miss Rate reduction against Eigen',
                ], 
                [
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1DCacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("L1DCacheMissRate") * (n_r + 2) + (i + 3), 'I8_I8'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1DCacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("L1DCacheMissRate") * (n_r + 2) + (i + 3), 'XNNPack'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1DCacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("L1DCacheMissRate") * (n_r + 2) + (i + 3), 'No_Caching'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1DCacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("L1DCacheMissRate") * (n_r + 2) + (i + 3), 'GEMMLOWP'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1DCacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("L1DCacheMissRate") * (n_r + 2) + (i + 3), 'FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1DCacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("L1DCacheMissRate") * (n_r + 2) + (i + 3), 'No_Caching_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1DCacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("L1DCacheMissRate") * (n_r + 2) + (i + 3), 'XNNPack_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1DCacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("L1DCacheMissRate") * (n_r + 2) + (i + 3), 'Eigen'),
                ],
                [
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1DCacheMissRate'], c, x, y, method, 'I8-I8'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1DCacheMissRate'], c, x, y, method, 'XNNPack'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1DCacheMissRate'], c, x, y, method, 'No-Caching'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1DCacheMissRate'], c, x, y, method, 'GEMMLOWP'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1DCacheMissRate'], c, x, y, method, 'FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1DCacheMissRate'], c, x, y, method, 'No-Caching-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1DCacheMissRate'], c, x, y, method, 'XNNPack-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1DCacheMissRate'], c, x, y, method, 'Eigen'),
                ],
            ))
            transforms.append(lambda x: x)
            transform_wrong_datas.append(lambda x: x if x != 0 else 1)
        elif 'L1ICacheAccess' == metric:
            extras.append((
                [
                    'L1 Instruction Cache Loads reduction against I8-I8',
                    'L1 Instruction Cache Loads reduction against XNNPack',
                    'L1 Instruction Cache Loads reduction against No-Caching',
                    'L1 Instruction Cache Loads reduction against GEMMLOWP',
                    'L1 Instruction Cache Loads reduction against FP32',
                    'L1 Instruction Cache Loads reduction against No-Caching-FP32',
                    'L1 Instruction Cache Loads reduction against XNNPack-FP32',
                    'L1 Instruction Cache Loads reduction against Eigen',
                ], 
                [
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1ICacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("L1ICacheAccess") * (n_r + 2) + (i + 3), 'I8_I8'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1ICacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("L1ICacheAccess") * (n_r + 2) + (i + 3), 'XNNPack'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1ICacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("L1ICacheAccess") * (n_r + 2) + (i + 3), 'No_Caching'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1ICacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("L1ICacheAccess") * (n_r + 2) + (i + 3), 'GEMMLOWP'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1ICacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("L1ICacheAccess") * (n_r + 2) + (i + 3), 'FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1ICacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("L1ICacheAccess") * (n_r + 2) + (i + 3), 'No_Caching_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1ICacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("L1ICacheAccess") * (n_r + 2) + (i + 3), 'XNNPack_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1ICacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("L1ICacheAccess") * (n_r + 2) + (i + 3), 'Eigen'),
                ],
                [
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1ICacheAccess'], c, x, y, method, 'I8-I8'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1ICacheAccess'], c, x, y, method, 'XNNPack'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1ICacheAccess'], c, x, y, method, 'No-Caching'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1ICacheAccess'], c, x, y, method, 'GEMMLOWP'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1ICacheAccess'], c, x, y, method, 'FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1ICacheAccess'], c, x, y, method, 'No-Caching-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1ICacheAccess'], c, x, y, method, 'XNNPack-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1ICacheAccess'], c, x, y, method, 'Eigen'),
                ],
            ))
            transforms.append(lambda x: x)
            transform_wrong_datas.append(lambda x: x)# lambda x: x if x != 0 else 1)
        elif 'L1ICacheMisses' == metric:
            extras.append((
                [
                    'L1 Instruction Cache Load Misses reduction against I8-I8',
                    'L1 Instruction Cache Load Misses reduction against XNNPack',
                    'L1 Instruction Cache Load Misses reduction against No-Caching',
                    'L1 Instruction Cache Load Misses reduction against GEMMLOWP',
                    'L1 Instruction Cache Load Misses reduction against FP32',
                    'L1 Instruction Cache Load Misses reduction against No-Caching-FP32',
                    'L1 Instruction Cache Load Misses reduction against XNNPack-FP32',
                    'L1 Instruction Cache Load Misses reduction against Eigen',
                ], 
                [
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1ICacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("L1ICacheMisses") * (n_r + 2) + (i + 3), 'I8_I8'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1ICacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("L1ICacheMisses") * (n_r + 2) + (i + 3), 'XNNPack'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1ICacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("L1ICacheMisses") * (n_r + 2) + (i + 3), 'No_Caching'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1ICacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("L1ICacheMisses") * (n_r + 2) + (i + 3), 'GEMMLOWP'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1ICacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("L1ICacheMisses") * (n_r + 2) + (i + 3), 'FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1ICacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("L1ICacheMisses") * (n_r + 2) + (i + 3), 'No_Caching_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1ICacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("L1ICacheMisses") * (n_r + 2) + (i + 3), 'XNNPack_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1ICacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("L1ICacheMisses") * (n_r + 2) + (i + 3), 'Eigen'),
                ],
                [
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1ICacheMisses'], c, x, y, method, 'I8-I8'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1ICacheMisses'], c, x, y, method, 'XNNPack'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1ICacheMisses'], c, x, y, method, 'No-Caching'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1ICacheMisses'], c, x, y, method, 'GEMMLOWP'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1ICacheMisses'], c, x, y, method, 'FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1ICacheMisses'], c, x, y, method, 'No-Caching-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1ICacheMisses'], c, x, y, method, 'XNNPack-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1ICacheMisses'], c, x, y, method, 'Eigen'),
                ],
            ))
            transforms.append(lambda x: x)
            transform_wrong_datas.append(lambda x: x)# lambda x: x if x != 0 else 1)
        elif 'L1ICacheMissRate' == metric:
            extras.append((
                [
                    'L1 Instruction Cache Miss Rate reduction against I8-I8',
                    'L1 Instruction Cache Miss Rate reduction against XNNPack',
                    'L1 Instruction Cache Miss Rate reduction against No-Caching',
                    'L1 Instruction Cache Miss Rate reduction against GEMMLOWP',
                    'L1 Instruction Cache Miss Rate reduction against FP32',
                    'L1 Instruction Cache Miss Rate reduction against No-Caching-FP32',
                    'L1 Instruction Cache Miss Rate reduction against XNNPack-FP32',
                    'L1 Instruction Cache Miss Rate reduction against Eigen',
                ], 
                [
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1ICacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("L1ICacheMissRate") * (n_r + 2) + (i + 3), 'I8_I8'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1ICacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("L1ICacheMissRate") * (n_r + 2) + (i + 3), 'XNNPack'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1ICacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("L1ICacheMissRate") * (n_r + 2) + (i + 3), 'No_Caching'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1ICacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("L1ICacheMissRate") * (n_r + 2) + (i + 3), 'GEMMLOWP'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1ICacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("L1ICacheMissRate") * (n_r + 2) + (i + 3), 'FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1ICacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("L1ICacheMissRate") * (n_r + 2) + (i + 3), 'No_Caching_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1ICacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("L1ICacheMissRate") * (n_r + 2) + (i + 3), 'XNNPack_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L1ICacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("L1ICacheMissRate") * (n_r + 2) + (i + 3), 'Eigen'),
                ],
                [
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1ICacheMissRate'], c, x, y, method, 'I8-I8'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1ICacheMissRate'], c, x, y, method, 'XNNPack'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1ICacheMissRate'], c, x, y, method, 'No-Caching'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1ICacheMissRate'], c, x, y, method, 'GEMMLOWP'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1ICacheMissRate'], c, x, y, method, 'FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1ICacheMissRate'], c, x, y, method, 'No-Caching-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1ICacheMissRate'], c, x, y, method, 'XNNPack-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L1ICacheMissRate'], c, x, y, method, 'Eigen'),
                ],
            ))
            transforms.append(lambda x: x)
            transform_wrong_datas.append(lambda x: x if x != 0 else 1)
        elif 'L2DCacheAccess' == metric:
            extras.append((
                [
                    'L2 Data Cache Accesses reduction against I8-I8',
                    'L2 Data Cache Accesses reduction against XNNPack',
                    'L2 Data Cache Accesses reduction against No-Caching',
                    'L2 Data Cache Accesses reduction against GEMMLOWP',
                    'L2 Data Cache Accesses reduction against FP32',
                    'L2 Data Cache Accesses reduction against No-Caching-FP32',
                    'L2 Data Cache Accesses reduction against XNNPack-FP32',
                    'L2 Data Cache Accesses reduction against Eigen',
                ], 
                [
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2DCacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("L2DCacheAccess") * (n_r + 2) + (i + 3), 'I8_I8'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2DCacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("L2DCacheAccess") * (n_r + 2) + (i + 3), 'XNNPack'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2DCacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("L2DCacheAccess") * (n_r + 2) + (i + 3), 'No_Caching'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2DCacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("L2DCacheAccess") * (n_r + 2) + (i + 3), 'GEMMLOWP'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2DCacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("L2DCacheAccess") * (n_r + 2) + (i + 3), 'FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2DCacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("L2DCacheAccess") * (n_r + 2) + (i + 3), 'No_Caching_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2DCacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("L2DCacheAccess") * (n_r + 2) + (i + 3), 'XNNPack_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2DCacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("L2DCacheAccess") * (n_r + 2) + (i + 3), 'Eigen'),
                ],
                [
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2DCacheAccess'], c, x, y, method, 'I8-I8'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2DCacheAccess'], c, x, y, method, 'XNNPack'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2DCacheAccess'], c, x, y, method, 'No-Caching'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2DCacheAccess'], c, x, y, method, 'GEMMLOWP'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2DCacheAccess'], c, x, y, method, 'FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2DCacheAccess'], c, x, y, method, 'No-Caching-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2DCacheAccess'], c, x, y, method, 'XNNPack-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2DCacheAccess'], c, x, y, method, 'Eigen'),
                ],
            ))
            transforms.append(lambda x: x)
            transform_wrong_datas.append(lambda x: x)# lambda x: x if x != 0 else 1)
        elif 'L2DCacheMisses' == metric:
            extras.append((
                [
                    'L2 Data Cache Misses reduction against I8-I8',
                    'L2 Data Cache Misses reduction against XNNPack',
                    'L2 Data Cache Misses reduction against No-Caching',
                    'L2 Data Cache Misses reduction against GEMMLOWP',
                    'L2 Data Cache Misses reduction against FP32',
                    'L2 Data Cache Misses reduction against No-Caching-FP32',
                    'L2 Data Cache Misses reduction against XNNPack-FP32',
                    'L2 Data Cache Misses reduction against Eigen',
                ], 
                [
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2DCacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("L2DCacheMisses") * (n_r + 2) + (i + 3), 'I8_I8'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2DCacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("L2DCacheMisses") * (n_r + 2) + (i + 3), 'XNNPack'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2DCacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("L2DCacheMisses") * (n_r + 2) + (i + 3), 'No_Caching'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2DCacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("L2DCacheMisses") * (n_r + 2) + (i + 3), 'GEMMLOWP'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2DCacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("L2DCacheMisses") * (n_r + 2) + (i + 3), 'FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2DCacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("L2DCacheMisses") * (n_r + 2) + (i + 3), 'No_Caching_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2DCacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("L2DCacheMisses") * (n_r + 2) + (i + 3), 'XNNPack_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2DCacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("L2DCacheMisses") * (n_r + 2) + (i + 3), 'Eigen'),
                ],
                [
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2DCacheMisses'], c, x, y, method, 'I8-I8'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2DCacheMisses'], c, x, y, method, 'XNNPack'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2DCacheMisses'], c, x, y, method, 'No-Caching'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2DCacheMisses'], c, x, y, method, 'GEMMLOWP'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2DCacheMisses'], c, x, y, method, 'FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2DCacheMisses'], c, x, y, method, 'No-Caching-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2DCacheMisses'], c, x, y, method, 'XNNPack-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2DCacheMisses'], c, x, y, method, 'Eigen'),
                ],
            ))
            transforms.append(lambda x: x)
            transform_wrong_datas.append(lambda x: x)# lambda x: x if x != 0 else 1)
        elif 'L2DCacheMissRate' == metric:
            extras.append((
                [
                    'L2 Data Cache Miss Rate reduction against I8-I8',
                    'L2 Data Cache Miss Rate reduction against XNNPack',
                    'L2 Data Cache Miss Rate reduction against No-Caching',
                    'L2 Data Cache Miss Rate reduction against GEMMLOWP',
                    'L2 Data Cache Miss Rate reduction against FP32',
                    'L2 Data Cache Miss Rate reduction against No-Caching-FP32',
                    'L2 Data Cache Miss Rate reduction against XNNPack-FP32',
                    'L2 Data Cache Miss Rate reduction against Eigen',
                ], 
                [
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2DCacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("L2DCacheMissRate") * (n_r + 2) + (i + 3), 'I8_I8'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2DCacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("L2DCacheMissRate") * (n_r + 2) + (i + 3), 'XNNPack'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2DCacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("L2DCacheMissRate") * (n_r + 2) + (i + 3), 'No_Caching'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2DCacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("L2DCacheMissRate") * (n_r + 2) + (i + 3), 'GEMMLOWP'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2DCacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("L2DCacheMissRate") * (n_r + 2) + (i + 3), 'FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2DCacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("L2DCacheMissRate") * (n_r + 2) + (i + 3), 'No_Caching_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2DCacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("L2DCacheMissRate") * (n_r + 2) + (i + 3), 'XNNPack_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2DCacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("L2DCacheMissRate") * (n_r + 2) + (i + 3), 'Eigen'),
                ],
                [
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2DCacheMissRate'], c, x, y, method, 'I8-I8'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2DCacheMissRate'], c, x, y, method, 'XNNPack'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2DCacheMissRate'], c, x, y, method, 'No-Caching'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2DCacheMissRate'], c, x, y, method, 'GEMMLOWP'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2DCacheMissRate'], c, x, y, method, 'FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2DCacheMissRate'], c, x, y, method, 'No-Caching-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2DCacheMissRate'], c, x, y, method, 'XNNPack-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2DCacheMissRate'], c, x, y, method, 'Eigen'),
                ],
            ))
            transforms.append(lambda x: x)
            transform_wrong_datas.append(lambda x: x)# lambda x: x if x != 0 else 1)
        elif 'L2DCacheMissLatency' == metric:
            extras.append((
                [
                    'L2 Data Cache Miss Latency reduction against I8-I8',
                    'L2 Data Cache Miss Latency reduction against XNNPack',
                    'L2 Data Cache Miss Latency reduction against No-Caching',
                    'L2 Data Cache Miss Latency reduction against GEMMLOWP',
                    'L2 Data Cache Miss Latency reduction against FP32',
                    'L2 Data Cache Miss Latency reduction against No-Caching-FP32',
                    'L2 Data Cache Miss Latency reduction against XNNPack-FP32',
                    'L2 Data Cache Miss Latency reduction against Eigen',
                ], 
                [
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2DCacheMissLatency") * (n_r + 2) + (i + 3), metrics_sorted.index("L2DCacheMissLatency") * (n_r + 2) + (i + 3), 'I8_I8'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2DCacheMissLatency") * (n_r + 2) + (i + 3), metrics_sorted.index("L2DCacheMissLatency") * (n_r + 2) + (i + 3), 'XNNPack'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2DCacheMissLatency") * (n_r + 2) + (i + 3), metrics_sorted.index("L2DCacheMissLatency") * (n_r + 2) + (i + 3), 'No_Caching'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2DCacheMissLatency") * (n_r + 2) + (i + 3), metrics_sorted.index("L2DCacheMissLatency") * (n_r + 2) + (i + 3), 'GEMMLOWP'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2DCacheMissLatency") * (n_r + 2) + (i + 3), metrics_sorted.index("L2DCacheMissLatency") * (n_r + 2) + (i + 3), 'FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2DCacheMissLatency") * (n_r + 2) + (i + 3), metrics_sorted.index("L2DCacheMissLatency") * (n_r + 2) + (i + 3), 'No_Caching_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2DCacheMissLatency") * (n_r + 2) + (i + 3), metrics_sorted.index("L2DCacheMissLatency") * (n_r + 2) + (i + 3), 'XNNPack_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2DCacheMissLatency") * (n_r + 2) + (i + 3), metrics_sorted.index("L2DCacheMissLatency") * (n_r + 2) + (i + 3), 'Eigen'),
                ],
                [
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2DCacheMissLatency'], c, x, y, method, 'I8-I8'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2DCacheMissLatency'], c, x, y, method, 'XNNPack'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2DCacheMissLatency'], c, x, y, method, 'No-Caching'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2DCacheMissLatency'], c, x, y, method, 'GEMMLOWP'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2DCacheMissLatency'], c, x, y, method, 'FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2DCacheMissLatency'], c, x, y, method, 'No-Caching-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2DCacheMissLatency'], c, x, y, method, 'XNNPack-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2DCacheMissLatency'], c, x, y, method, 'Eigen'),
                ],
            ))
            transforms.append(lambda x: x)
            transform_wrong_datas.append(lambda x: x)# lambda x: x if x != 0 else 1)
        elif 'L2ICacheAccess' == metric:
            extras.append((
                [
                    'L2 Instructions Cache Accesses reduction against I8-I8',
                    'L2 Instructions Cache Accesses reduction against XNNPack',
                    'L2 Instructions Cache Accesses reduction against No-Caching',
                    'L2 Instructions Cache Accesses reduction against GEMMLOWP',
                    'L2 Instructions Cache Accesses reduction against FP32',
                    'L2 Instructions Cache Accesses reduction against No-Caching-FP32',
                    'L2 Instructions Cache Accesses reduction against XNNPack-FP32',
                    'L2 Instructions Cache Accesses reduction against Eigen',
                ], 
                [
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2ICacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("L2ICacheAccess") * (n_r + 2) + (i + 3), 'I8_I8'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2ICacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("L2ICacheAccess") * (n_r + 2) + (i + 3), 'XNNPack'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2ICacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("L2ICacheAccess") * (n_r + 2) + (i + 3), 'No_Caching'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2ICacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("L2ICacheAccess") * (n_r + 2) + (i + 3), 'GEMMLOWP'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2ICacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("L2ICacheAccess") * (n_r + 2) + (i + 3), 'FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2ICacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("L2ICacheAccess") * (n_r + 2) + (i + 3), 'No_Caching_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2ICacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("L2ICacheAccess") * (n_r + 2) + (i + 3), 'XNNPack_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2ICacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("L2ICacheAccess") * (n_r + 2) + (i + 3), 'Eigen'),
                ],
                [
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2ICacheAccess'], c, x, y, method, 'I8-I8'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2ICacheAccess'], c, x, y, method, 'XNNPack'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2ICacheAccess'], c, x, y, method, 'No-Caching'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2ICacheAccess'], c, x, y, method, 'GEMMLOWP'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2ICacheAccess'], c, x, y, method, 'FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2ICacheAccess'], c, x, y, method, 'No-Caching-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2ICacheAccess'], c, x, y, method, 'XNNPack-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2ICacheAccess'], c, x, y, method, 'Eigen'),
                ],
            ))
            transforms.append(lambda x: x)
            transform_wrong_datas.append(lambda x: x)# lambda x: x if x != 0 else 1)
        elif 'L2ICacheMisses' == metric:
            extras.append((
                [
                    'L2 Instructions Cache Misses reduction against I8-I8',
                    'L2 Instructions Cache Misses reduction against XNNPack',
                    'L2 Instructions Cache Misses reduction against No-Caching',
                    'L2 Instructions Cache Misses reduction against GEMMLOWP',
                    'L2 Instructions Cache Misses reduction against FP32',
                    'L2 Instructions Cache Misses reduction against No-Caching-FP32',
                    'L2 Instructions Cache Misses reduction against XNNPack-FP32',
                    'L2 Instructions Cache Misses reduction against Eigen',
                ], 
                [
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2ICacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("L2ICacheMisses") * (n_r + 2) + (i + 3), 'I8_I8'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2ICacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("L2ICacheMisses") * (n_r + 2) + (i + 3), 'XNNPack'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2ICacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("L2ICacheMisses") * (n_r + 2) + (i + 3), 'No_Caching'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2ICacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("L2ICacheMisses") * (n_r + 2) + (i + 3), 'GEMMLOWP'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2ICacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("L2ICacheMisses") * (n_r + 2) + (i + 3), 'FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2ICacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("L2ICacheMisses") * (n_r + 2) + (i + 3), 'No_Caching_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2ICacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("L2ICacheMisses") * (n_r + 2) + (i + 3), 'XNNPack_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2ICacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("L2ICacheMisses") * (n_r + 2) + (i + 3), 'Eigen'),
                ],
                [
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2ICacheMisses'], c, x, y, method, 'I8-I8'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2ICacheMisses'], c, x, y, method, 'XNNPack'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2ICacheMisses'], c, x, y, method, 'No-Caching'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2ICacheMisses'], c, x, y, method, 'GEMMLOWP'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2ICacheMisses'], c, x, y, method, 'FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2ICacheMisses'], c, x, y, method, 'No-Caching-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2ICacheMisses'], c, x, y, method, 'XNNPack-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2ICacheMisses'], c, x, y, method, 'Eigen'),
                ],
            ))
            transforms.append(lambda x: x)
            transform_wrong_datas.append(lambda x: x)# lambda x: x if x != 0 else 1)
        elif 'L2ICacheMissRate' == metric:
            extras.append((
                [
                    'L2 Instructions Cache Miss Rate reduction against I8-I8',
                    'L2 Instructions Cache Miss Rate reduction against XNNPack',
                    'L2 Instructions Cache Miss Rate reduction against No-Caching',
                    'L2 Instructions Cache Miss Rate reduction against GEMMLOWP',
                    'L2 Instructions Cache Miss Rate reduction against FP32',
                    'L2 Instructions Cache Miss Rate reduction against No-Caching-FP32',
                    'L2 Instructions Cache Miss Rate reduction against XNNPack-FP32',
                    'L2 Instructions Cache Miss Rate reduction against Eigen',
                ], 
                [
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2ICacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("L2ICacheMissRate") * (n_r + 2) + (i + 3), 'I8_I8'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2ICacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("L2ICacheMissRate") * (n_r + 2) + (i + 3), 'XNNPack'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2ICacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("L2ICacheMissRate") * (n_r + 2) + (i + 3), 'No_Caching'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2ICacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("L2ICacheMissRate") * (n_r + 2) + (i + 3), 'GEMMLOWP'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2ICacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("L2ICacheMissRate") * (n_r + 2) + (i + 3), 'FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2ICacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("L2ICacheMissRate") * (n_r + 2) + (i + 3), 'No_Caching_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2ICacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("L2ICacheMissRate") * (n_r + 2) + (i + 3), 'XNNPack_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("L2ICacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("L2ICacheMissRate") * (n_r + 2) + (i + 3), 'Eigen'),
                ],
                [
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2ICacheMissRate'], c, x, y, method, 'I8-I8'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2ICacheMissRate'], c, x, y, method, 'XNNPack'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2ICacheMissRate'], c, x, y, method, 'No-Caching'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2ICacheMissRate'], c, x, y, method, 'GEMMLOWP'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2ICacheMissRate'], c, x, y, method, 'FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2ICacheMissRate'], c, x, y, method, 'No-Caching-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2ICacheMissRate'], c, x, y, method, 'XNNPack-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['L2ICacheMissRate'], c, x, y, method, 'Eigen'),
                ],
            ))
            transforms.append(lambda x: x)
            transform_wrong_datas.append(lambda x: x)# lambda x: x if x != 0 else 1)
        elif 'LLDCacheAccess' == metric:
            extras.append((
                [
                    'Last-Level Data Cache Accesses reduction against I8-I8',
                    'Last-Level Data Cache Accesses reduction against XNNPack',
                    'Last-Level Data Cache Accesses reduction against No-Caching',
                    'Last-Level Data Cache Accesses reduction against GEMMLOWP',
                    'Last-Level Data Cache Accesses reduction against FP32',
                    'Last-Level Data Cache Accesses reduction against No-Caching-FP32',
                    'Last-Level Data Cache Accesses reduction against XNNPack-FP32',
                    'Last-Level Data Cache Accesses reduction against Eigen',
                ], 
                [
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLDCacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("LLDCacheAccess") * (n_r + 2) + (i + 3), 'I8_I8'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLDCacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("LLDCacheAccess") * (n_r + 2) + (i + 3), 'XNNPack'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLDCacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("LLDCacheAccess") * (n_r + 2) + (i + 3), 'No_Caching'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLDCacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("LLDCacheAccess") * (n_r + 2) + (i + 3), 'GEMMLOWP'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLDCacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("LLDCacheAccess") * (n_r + 2) + (i + 3), 'FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLDCacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("LLDCacheAccess") * (n_r + 2) + (i + 3), 'No_Caching_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLDCacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("LLDCacheAccess") * (n_r + 2) + (i + 3), 'XNNPack_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLDCacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("LLDCacheAccess") * (n_r + 2) + (i + 3), 'Eigen'),
                ],
                [
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLDCacheAccess'], c, x, y, method, 'I8-I8'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLDCacheAccess'], c, x, y, method, 'XNNPack'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLDCacheAccess'], c, x, y, method, 'No-Caching'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLDCacheAccess'], c, x, y, method, 'GEMMLOWP'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLDCacheAccess'], c, x, y, method, 'FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLDCacheAccess'], c, x, y, method, 'No-Caching-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLDCacheAccess'], c, x, y, method, 'XNNPack-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLDCacheAccess'], c, x, y, method, 'Eigen'),
                ],
            ))
            transforms.append(lambda x: x)
            transform_wrong_datas.append(lambda x: x)# lambda x: x if x != 0 else 1)
        elif 'LLDCacheMisses' == metric:
            extras.append((
                [
                    'Last-Level Data Cache Misses reduction against I8-I8',
                    'Last-Level Data Cache Misses reduction against XNNPack',
                    'Last-Level Data Cache Misses reduction against No-Caching',
                    'Last-Level Data Cache Misses reduction against GEMMLOWP',
                    'Last-Level Data Cache Misses reduction against FP32',
                    'Last-Level Data Cache Misses reduction against No-Caching-FP32',
                    'Last-Level Data Cache Misses reduction against XNNPack-FP32',
                    'Last-Level Data Cache Misses reduction against Eigen',
                ], 
                [
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLDCacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("LLDCacheMisses") * (n_r + 2) + (i + 3), 'I8_I8'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLDCacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("LLDCacheMisses") * (n_r + 2) + (i + 3), 'XNNPack'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLDCacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("LLDCacheMisses") * (n_r + 2) + (i + 3), 'No_Caching'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLDCacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("LLDCacheMisses") * (n_r + 2) + (i + 3), 'GEMMLOWP'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLDCacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("LLDCacheMisses") * (n_r + 2) + (i + 3), 'FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLDCacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("LLDCacheMisses") * (n_r + 2) + (i + 3), 'No_Caching_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLDCacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("LLDCacheMisses") * (n_r + 2) + (i + 3), 'XNNPack_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLDCacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("LLDCacheMisses") * (n_r + 2) + (i + 3), 'Eigen'),
                ],
                [
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLDCacheMisses'], c, x, y, method, 'I8-I8'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLDCacheMisses'], c, x, y, method, 'XNNPack'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLDCacheMisses'], c, x, y, method, 'No-Caching'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLDCacheMisses'], c, x, y, method, 'GEMMLOWP'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLDCacheMisses'], c, x, y, method, 'FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLDCacheMisses'], c, x, y, method, 'No-Caching-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLDCacheMisses'], c, x, y, method, 'XNNPack-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLDCacheMisses'], c, x, y, method, 'Eigen'),
                ],
            ))
            transforms.append(lambda x: x)
            transform_wrong_datas.append(lambda x: x)# lambda x: x if x != 0 else 1)
        elif 'LLDCacheMissRate' == metric:
            extras.append((
                [
                    'Last-Level Data Cache Miss Rate reduction against I8-I8',
                    'Last-Level Data Cache Miss Rate reduction against XNNPack',
                    'Last-Level Data Cache Miss Rate reduction against No-Caching',
                    'Last-Level Data Cache Miss Rate reduction against GEMMLOWP',
                    'Last-Level Data Cache Miss Rate reduction against FP32',
                    'Last-Level Data Cache Miss Rate reduction against No-Caching-FP32',
                    'Last-Level Data Cache Miss Rate reduction against XNNPack-FP32',
                    'Last-Level Data Cache Miss Rate reduction against Eigen',
                ], 
                [
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLDCacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("LLDCacheMissRate") * (n_r + 2) + (i + 3), 'I8_I8'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLDCacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("LLDCacheMissRate") * (n_r + 2) + (i + 3), 'XNNPack'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLDCacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("LLDCacheMissRate") * (n_r + 2) + (i + 3), 'No_Caching'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLDCacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("LLDCacheMissRate") * (n_r + 2) + (i + 3), 'GEMMLOWP'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLDCacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("LLDCacheMissRate") * (n_r + 2) + (i + 3), 'FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLDCacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("LLDCacheMissRate") * (n_r + 2) + (i + 3), 'No_Caching_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLDCacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("LLDCacheMissRate") * (n_r + 2) + (i + 3), 'XNNPack_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLDCacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("LLDCacheMissRate") * (n_r + 2) + (i + 3), 'Eigen'),
                ],
                [
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLDCacheMissRate'], c, x, y, method, 'I8-I8'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLDCacheMissRate'], c, x, y, method, 'XNNPack'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLDCacheMissRate'], c, x, y, method, 'No-Caching'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLDCacheMissRate'], c, x, y, method, 'GEMMLOWP'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLDCacheMissRate'], c, x, y, method, 'FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLDCacheMissRate'], c, x, y, method, 'No-Caching-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLDCacheMissRate'], c, x, y, method, 'XNNPack-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLDCacheMissRate'], c, x, y, method, 'Eigen'),
                ],
            ))
            transforms.append(lambda x: x)
            transform_wrong_datas.append(lambda x: x)# lambda x: x if x != 0 else 1)
        elif 'LLDCacheMissLatency' == metric:
            extras.append((
                [
                    'Last-Level Data Cache Miss Latency reduction against I8-I8',
                    'Last-Level Data Cache Miss Latency reduction against XNNPack',
                    'Last-Level Data Cache Miss Latency reduction against No-Caching',
                    'Last-Level Data Cache Miss Latency reduction against GEMMLOWP',
                    'Last-Level Data Cache Miss Latency reduction against FP32',
                    'Last-Level Data Cache Miss Latency reduction against No-Caching-FP32',
                    'Last-Level Data Cache Miss Latency reduction against XNNPack-FP32',
                    'Last-Level Data Cache Miss Latency reduction against Eigen',
                ], 
                [
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLDCacheMissLatency") * (n_r + 2) + (i + 3), metrics_sorted.index("LLDCacheMissLatency") * (n_r + 2) + (i + 3), 'I8_I8'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLDCacheMissLatency") * (n_r + 2) + (i + 3), metrics_sorted.index("LLDCacheMissLatency") * (n_r + 2) + (i + 3), 'XNNPack'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLDCacheMissLatency") * (n_r + 2) + (i + 3), metrics_sorted.index("LLDCacheMissLatency") * (n_r + 2) + (i + 3), 'No_Caching'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLDCacheMissLatency") * (n_r + 2) + (i + 3), metrics_sorted.index("LLDCacheMissLatency") * (n_r + 2) + (i + 3), 'GEMMLOWP'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLDCacheMissLatency") * (n_r + 2) + (i + 3), metrics_sorted.index("LLDCacheMissLatency") * (n_r + 2) + (i + 3), 'FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLDCacheMissLatency") * (n_r + 2) + (i + 3), metrics_sorted.index("LLDCacheMissLatency") * (n_r + 2) + (i + 3), 'No_Caching_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLDCacheMissLatency") * (n_r + 2) + (i + 3), metrics_sorted.index("LLDCacheMissLatency") * (n_r + 2) + (i + 3), 'XNNPack_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLDCacheMissLatency") * (n_r + 2) + (i + 3), metrics_sorted.index("LLDCacheMissLatency") * (n_r + 2) + (i + 3), 'Eigen'),
                ],
                [
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLDCacheMissLatency'], c, x, y, method, 'I8-I8'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLDCacheMissLatency'], c, x, y, method, 'XNNPack'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLDCacheMissLatency'], c, x, y, method, 'No-Caching'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLDCacheMissLatency'], c, x, y, method, 'GEMMLOWP'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLDCacheMissLatency'], c, x, y, method, 'FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLDCacheMissLatency'], c, x, y, method, 'No-Caching-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLDCacheMissLatency'], c, x, y, method, 'XNNPack-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLDCacheMissLatency'], c, x, y, method, 'Eigen'),
                ],
            ))
            transforms.append(lambda x: x)
            transform_wrong_datas.append(lambda x: x)# lambda x: x if x != 0 else 1)
        elif 'LLICacheAccess' == metric:
            extras.append((
                [
                    'Last-Level Instructions Cache Accesses reduction against I8-I8',
                    'Last-Level Instructions Cache Accesses reduction against XNNPack',
                    'Last-Level Instructions Cache Accesses reduction against No-Caching',
                    'Last-Level Instructions Cache Accesses reduction against GEMMLOWP',
                    'Last-Level Instructions Cache Accesses reduction against FP32',
                    'Last-Level Instructions Cache Accesses reduction against No-Caching-FP32',
                    'Last-Level Instructions Cache Accesses reduction against XNNPack-FP32',
                    'Last-Level Instructions Cache Accesses reduction against Eigen',
                ], 
                [
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLICacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("LLICacheAccess") * (n_r + 2) + (i + 3), 'I8_I8'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLICacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("LLICacheAccess") * (n_r + 2) + (i + 3), 'XNNPack'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLICacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("LLICacheAccess") * (n_r + 2) + (i + 3), 'No_Caching'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLICacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("LLICacheAccess") * (n_r + 2) + (i + 3), 'GEMMLOWP'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLICacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("LLICacheAccess") * (n_r + 2) + (i + 3), 'FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLICacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("LLICacheAccess") * (n_r + 2) + (i + 3), 'No_Caching_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLICacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("LLICacheAccess") * (n_r + 2) + (i + 3), 'XNNPack_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLICacheAccess") * (n_r + 2) + (i + 3), metrics_sorted.index("LLICacheAccess") * (n_r + 2) + (i + 3), 'Eigen'),
                ],
                [
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLICacheAccess'], c, x, y, method, 'I8-I8'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLICacheAccess'], c, x, y, method, 'XNNPack'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLICacheAccess'], c, x, y, method, 'No-Caching'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLICacheAccess'], c, x, y, method, 'GEMMLOWP'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLICacheAccess'], c, x, y, method, 'FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLICacheAccess'], c, x, y, method, 'No-Caching-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLICacheAccess'], c, x, y, method, 'XNNPack-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLICacheAccess'], c, x, y, method, 'Eigen'),
                ],
            ))
            transforms.append(lambda x: x)
            transform_wrong_datas.append(lambda x: x)# lambda x: x if x != 0 else 1)
        elif 'LLICacheMisses' == metric:
            extras.append((
                [
                    'Last-Level Instructions Cache Misses reduction against I8-I8',
                    'Last-Level Instructions Cache Misses reduction against XNNPack',
                    'Last-Level Instructions Cache Misses reduction against No-Caching',
                    'Last-Level Instructions Cache Misses reduction against GEMMLOWP',
                    'Last-Level Instructions Cache Misses reduction against FP32',
                    'Last-Level Instructions Cache Misses reduction against No-Caching-FP32',
                    'Last-Level Instructions Cache Misses reduction against XNNPack-FP32',
                    'Last-Level Instructions Cache Misses reduction against Eigen',
                ], 
                [
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLICacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("LLICacheMisses") * (n_r + 2) + (i + 3), 'I8_I8'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLICacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("LLICacheMisses") * (n_r + 2) + (i + 3), 'XNNPack'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLICacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("LLICacheMisses") * (n_r + 2) + (i + 3), 'No_Caching'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLICacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("LLICacheMisses") * (n_r + 2) + (i + 3), 'GEMMLOWP'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLICacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("LLICacheMisses") * (n_r + 2) + (i + 3), 'FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLICacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("LLICacheMisses") * (n_r + 2) + (i + 3), 'No_Caching_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLICacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("LLICacheMisses") * (n_r + 2) + (i + 3), 'XNNPack_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLICacheMisses") * (n_r + 2) + (i + 3), metrics_sorted.index("LLICacheMisses") * (n_r + 2) + (i + 3), 'Eigen'),
                ],
                [
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLICacheMisses'], c, x, y, method, 'I8-I8'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLICacheMisses'], c, x, y, method, 'XNNPack'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLICacheMisses'], c, x, y, method, 'No-Caching'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLICacheMisses'], c, x, y, method, 'GEMMLOWP'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLICacheMisses'], c, x, y, method, 'FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLICacheMisses'], c, x, y, method, 'No-Caching-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLICacheMisses'], c, x, y, method, 'XNNPack-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLICacheMisses'], c, x, y, method, 'Eigen'),
                ],
            ))
            transforms.append(lambda x: x)
            transform_wrong_datas.append(lambda x: x)# lambda x: x if x != 0 else 1)
        elif 'LLICacheMissRate' == metric:
            extras.append((
                [
                    'Last-Level Instructions Cache Miss Rate reduction against I8-I8',
                    'Last-Level Instructions Cache Miss Rate reduction against XNNPack',
                    'Last-Level Instructions Cache Miss Rate reduction against No-Caching',
                    'Last-Level Instructions Cache Miss Rate reduction against GEMMLOWP',
                    'Last-Level Instructions Cache Miss Rate reduction against FP32',
                    'Last-Level Instructions Cache Miss Rate reduction against No-Caching-FP32',
                    'Last-Level Instructions Cache Miss Rate reduction against XNNPack-FP32',
                    'Last-Level Instructions Cache Miss Rate reduction against Eigen',
                ], 
                [
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLICacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("LLICacheMissRate") * (n_r + 2) + (i + 3), 'I8_I8'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLICacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("LLICacheMissRate") * (n_r + 2) + (i + 3), 'XNNPack'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLICacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("LLICacheMissRate") * (n_r + 2) + (i + 3), 'No_Caching'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLICacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("LLICacheMissRate") * (n_r + 2) + (i + 3), 'GEMMLOWP'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLICacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("LLICacheMissRate") * (n_r + 2) + (i + 3), 'FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLICacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("LLICacheMissRate") * (n_r + 2) + (i + 3), 'No_Caching_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLICacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("LLICacheMissRate") * (n_r + 2) + (i + 3), 'XNNPack_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("LLICacheMissRate") * (n_r + 2) + (i + 3), metrics_sorted.index("LLICacheMissRate") * (n_r + 2) + (i + 3), 'Eigen'),
                ],
                [
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLICacheMissRate'], c, x, y, method, 'I8-I8'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLICacheMissRate'], c, x, y, method, 'XNNPack'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLICacheMissRate'], c, x, y, method, 'No-Caching'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLICacheMissRate'], c, x, y, method, 'GEMMLOWP'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLICacheMissRate'], c, x, y, method, 'FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLICacheMissRate'], c, x, y, method, 'No-Caching-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLICacheMissRate'], c, x, y, method, 'XNNPack-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['LLICacheMissRate'], c, x, y, method, 'Eigen'),
                ],
            ))
            transforms.append(lambda x: x)
            transform_wrong_datas.append(lambda x: x)# lambda x: x if x != 0 else 1)
        elif 'TotalTime' == metric:
            extras.append((
                [
                    'Total Execution Time reduction against I8-I8',
                    'Total Execution Time reduction against XNNPack',
                    'Total Execution Time reduction against No-Caching',
                    'Total Execution Time reduction against GEMMLOWP',
                    'Total Execution Time reduction against FP32',
                    'Total Execution Time reduction against No-Caching-FP32',
                    'Total Execution Time reduction against XNNPack-FP32',
                    'Total Execution Time reduction against Eigen',
                ], 
                [
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("TotalTime") * (n_r + 2) + (i + 3), metrics_sorted.index("TotalTime") * (n_r + 2) + (i + 3), 'I8_I8'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("TotalTime") * (n_r + 2) + (i + 3), metrics_sorted.index("TotalTime") * (n_r + 2) + (i + 3), 'XNNPack'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("TotalTime") * (n_r + 2) + (i + 3), metrics_sorted.index("TotalTime") * (n_r + 2) + (i + 3), 'No_Caching'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("TotalTime") * (n_r + 2) + (i + 3), metrics_sorted.index("TotalTime") * (n_r + 2) + (i + 3), 'GEMMLOWP'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("TotalTime") * (n_r + 2) + (i + 3), metrics_sorted.index("TotalTime") * (n_r + 2) + (i + 3), 'FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("TotalTime") * (n_r + 2) + (i + 3), metrics_sorted.index("TotalTime") * (n_r + 2) + (i + 3), 'No_Caching_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("TotalTime") * (n_r + 2) + (i + 3), metrics_sorted.index("TotalTime") * (n_r + 2) + (i + 3), 'XNNPack_FP32'),
                    lambda k, i, j, n_t, n_r, n_c: method_based_decrease_based_on_basline_str.format(num_column_to_spreadsheet_letter(1 + j).upper(), metrics_sorted.index("TotalTime") * (n_r + 2) + (i + 3), metrics_sorted.index("TotalTime") * (n_r + 2) + (i + 3), 'Eigen'),
                ],
                [
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['TotalTime'], c, x, y, method, 'I8-I8'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['TotalTime'], c, x, y, method, 'XNNPack'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['TotalTime'], c, x, y, method, 'No-Caching'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['TotalTime'], c, x, y, method, 'GEMMLOWP'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['TotalTime'], c, x, y, method, 'FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['TotalTime'], c, x, y, method, 'No-Caching-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['TotalTime'], c, x, y, method, 'XNNPack-FP32'),
                    lambda c, x, y: method_based_decrease_based_on_basline_val(metrics_values['TotalTime'], c, x, y, method, 'Eigen'),
                ],
            ))
            transforms.append(lambda x: x)
            transform_wrong_datas.append(lambda x: x)# lambda x: x if x != 0 else 1)
    if method == "I8-I8":
        extras = [(None, None, None)] * len(metrics)
    add_colorscale_to_groups(
        add_extra_min_max(
            add_method_as_worksheet_to_workbook(
                workbook, method, metrics, input_sizes, output_sizes, batch_size, metrics_values, 
                extras=extras, transforms=transforms, transform_wrong_datas=transform_wrong_datas,
                highlight_area_extra=highlight_area, extra_is_extras=True,
            )
        )
    )

workbook.close()
if options.generate_exec_time_plots or options.generate_detailed_plots:
    Path("Plots").mkdir(exist_ok=True, parents=True)

if options.generate_exec_time_plots:
    Path(join("Plots", "ExecTime")).mkdir(exist_ok=True, parents=True)

    print("Generating ExecTime Plots.")

    for size, size_string in sizes:
        avg = [ log10(results[size][method]) for method in methods ]
        plt.figure(figsize=(16,9))
        plt.bar(methods, avg)
        plt.title(f"Execution Time of each method in running a GEMV with size of {size_string}")
        plt.xticks(rotation='40')
        plt.xlabel("Method")
        plt.ylabel("log(Execution Time (us))")
        plt.savefig(join("Plots", "ExecTime", f"{size_string}.png"), dpi=400)
        plt.close()
if options.generate_detailed_plots:
    for metric in [ 'CPUCycles', 'Instructions', 'IPC', 'TotalTime' ]:

        print(f"Generating {metric} Plots.")
        Path(join("Plots", metric)).mkdir(exist_ok=True, parents=True)

        for size, size_string in sizes:
            if metric in [ 'IPC', 'TotalTime' ]:
                avg = [ metrics_values[metric][size][method] for method in methods ]
            else:
                avg = [ log10(metrics_values[metric][size][method]) for method in methods ]
            plt.figure(figsize=(16,9))
            plt.bar(methods, avg)
            plt.title(f"Execution Time of each method in running a GEMV with size of {size_string}")
            plt.xticks(rotation='40')
            plt.xlabel("Method")
            if metric in [ 'IPC', 'TotalTime' ]:
                plt.ylabel(metric)
            else:
                plt.ylabel(f"log10({metric})")
            plt.savefig(join("Plots", metric, f"{size_string}.png"), dpi=400)
            plt.close()

exit()


for result_method_dir in result_methods_dirs:
    method, file_name, dir_name = result_method_dir
    for size_tuple in sizes:
        size, size_string = size_tuple
        if "128x32" not in size:
            continue
        report_file_name = f"simpleperf-{iterations}-{warmup_iterations}-{size_string}.report"
        with open(join(dir_name, report_file_name)) as report_f:
            # Remove First 2 lines containing nothing important
            report_lines = report_f.readlines()[2:]
            # Remove the end of line form each line
            report_lines = list(map(lambda x: x[:-1], report_lines))
            # Remove lines that start with " " or are empty
            report_lines = list(filter(lambda x: x and x[0] != " ", report_lines))
            # Seperating each metric's report
            c_idx = i_idx = 0
            for i, line in enumerate(report_lines):
                if 'Event: cpu-cycles:u (type 0, config 0)'                 in line: c_idx = i
                if 'Event: Instructions:u (type 0, config 1)'               in line: i_idx = i
            report_lines_c = report_lines[c_idx + 4:i_idx]
            tops = {}
            for line in report_lines_c:
                if line[0] == ' ':
                    continue
                line_parts = list(filter(lambda x: x, line.split(' ')))
                self_precent = float(line_parts[1][:-1])
                function_name = ' '.join(line_parts[6:]).split('(')[0].split('\n')[0].split('<')[0]
                if self_precent > 0.01:
                    if function_name not in tops:
                        tops[function_name] = self_precent
                    else:
                        tops[function_name] += self_precent
            tops = [ (k, tops[k]) for k, _ in list(tops.items()) ]
            tops_sorted = sorted(tops, key=lambda x: x[1], reverse=True)
            tops_sorted = tops_sorted[:10]
            print("[+] Top {} of {} of {} method".format(10, report_file_name, method))
            print('\t[-]', '\n\t[-] '.join(list(map(lambda x: "{}: {:.2f}".format(x[0], x[1]), tops_sorted))))
            print("\033[0m")
        print('\n', '******************************************************************************************************************', '\n')





