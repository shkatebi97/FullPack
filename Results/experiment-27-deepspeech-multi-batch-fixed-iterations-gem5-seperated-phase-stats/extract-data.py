#!/usr/bin/env python
from pathlib import Path
from os.path import isfile, join, isdir
from os import listdir
from types import FunctionType
import xlsxwriter
import optparse
import json
import re
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common_config import *

parser = optparse.OptionParser()

parser.add_option('-n', '--iterations',
    action="store", dest="iterations",
    help="Set iteration count", default=5)
parser.add_option('-w', '--warmup-iterations',
    action="store", dest="warmup_iterations",
    help="Set iteration count", default=1)

options, _ = parser.parse_args()

# batch_size = int(options.batch_size)
iterations = 5
warmup_iterations = 1

results = {}
methods = []

extra_complex_metrics = [
    {
        "name": "IPC",
        "data_extractor": lambda x, y, c, method, k, i, j, n_t, n_r, n_c: f"=Instructions!${chr(66 + j).upper()}${k * (n_r + 2) + (i + 3)} / CPUCycles!${chr(66 + j).upper()}${k * (n_r + 2) + (i + 3)}",
        "extra_extractor": lambda k, i, j, n_t, n_r, n_c: increase_based_on_basline_str.format(chr(66 + j).upper(), i + 3, k * (n_r + 2) + (i + 3)),
        "transform": lambda x: x,
        "transform_wrong_data": lambda x: x if x != 0 else 1
    },
    {
        "name": "Cache_Miss_Rate",
        "data_extractor": lambda x, y, c, method, k, i, j, n_t, n_r, n_c: f"=(L1DCacheMisses!${chr(66 + j).upper()}${k * (n_r + 2) + (i + 3)} / L1DCacheAccess!${chr(66 + j).upper()}${k * (n_r + 2) + (i + 3)}) * 100",
        "extra_extractor": lambda k, i, j, n_t, n_r, n_c: decrease_based_on_basline_str.format(chr(66 + j).upper(), i + 3, k * (n_r + 2) + (i + 3)),
        "transform": lambda x: x,
        "transform_wrong_data": lambda x: x if x != 0 else 1
    },
]

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
        if extra is not None:
            output_list.append([(table, bold_italic), ("", bold_italic), (extra_title, bold_italic)])
        else:
            output_list.append([(table, bold_italic)])
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
                if extra is not None:
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
            output_list.append(output_row)
        output_list.append([])
        groups.append("{0}{1}:{2}{3}".format(chr(66), k * (num_rows + 2) + 3, chr(66 + len(x_values) - 1), k * (num_rows + 2) + 2 + len(y_values)))
        if extra is not None:
            # if k == 0:
            #     baseline_extra = "{0}{1}:{2}{3}".format(chr(66 + num_columns), k * (num_rows + 2) + 1, chr(66 + num_columns + len(x_values)), k * (num_rows + 2) + 2 + len(y_values))
            extra_groups.append("{0}{1}:{2}{3}".format(chr(66 + num_columns + 1), k * (num_rows + 2) + 3, chr(66 + num_columns + len(x_values)), k * (num_rows + 2) + 2 + len(y_values)))
    worksheet = workbook.add_worksheet(sheet_name.replace("-", "_"))
    for i, row in enumerate(output_list):
        if len(row) == 3:
            worksheet.merge_range(i, 0, i, len(x_values), row[0][0], row[0][1])
            worksheet.merge_range(i, len(x_values) + 2, i, len(x_values) * 2 + 2, row[2][0], row[2][1])
        if len(row) == 1:
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
    payload["groups"] = groups
    payload["extra_groups"] = extra_groups
    payload["baseline_extra"] = baseline_extra
    payload["baseline_extra_format"] = simple
    payload["simple_format"] = simple
    payload["bold_italic_format"] = bold_italic
    return payload

def add_worksheet_to_workbook(
    workbook: object, sheet_name: str,
    x_values: list[str], y_values: list[str], 
    data: dict[dict[str]] | object | FunctionType,
    transform = lambda x: x,
    extra = None, extra_value = None, extra_title = None,
    transform_wrong_data = lambda x: x if x != 0 else 1,
    existing_worksheet: object = None, 
    existing_worksheet_y_offset: int = 0
    ) -> object:
    extra_groups = []
    output_list = []
    
    num_columns = len(x_values) + 1
    num_rows = len(y_values) + 1

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

    if extra is not None:
        output_list.append([(sheet_name, bold_italic), ("", bold_italic), ("Speed up against baseline" if extra_title is None else extra_title, bold_italic)])
    else:
        output_list.append([(sheet_name, bold_italic)])
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
            if data is not None and not callable(data):
                output_row[j + 1] = (transform_wrong_data(transform(data[y][x])), simple)
            elif data is not None and callable(data):
                output_row[j + 1] = (transform_wrong_data(transform(data(x, y, method, i, j, num_rows, num_columns))), simple)
            if extra is not None and extra_value is None:
                output_row[j + num_columns + 2] = (extra(i, j, existing_worksheet_y_offset, num_rows, num_columns), simple, True)
            if extra is not None and extra_value is not None:
                this_value           = float(extra_value(x_values[j], y_values[i]))
                next_in_row_value    = float(extra_value(x_values[j], y_values[i]) if i == len(y_values) - 1 else extra_value(x_values[j], y_values[i + 1]))
                next_in_column_value = float(extra_value(x_values[j], y_values[i]) if j == len(x_values) - 1 else extra_value(x_values[j + 1], y_values[i]))
                output_row[j + num_columns + 2] = (
                    extra(i, j, existing_worksheet_y_offset, num_rows, num_columns),
                    simple,
                    True,
                    next_in_column_value - this_value,
                    next_in_row_value - this_value,
                )
        output_list.append(output_row)
    output_list.append([])
    if extra is not None:
        extra_groups.append("{0}{1}:{2}{1}".format(chr(66 + num_columns + 1), existing_worksheet_y_offset + 3, chr(66 + num_columns + len(x_values))))
    if existing_worksheet is not None:
        worksheet = existing_worksheet
    else:    
        worksheet = workbook.add_worksheet(sheet_name)
    for i, row in enumerate(output_list):
        i += existing_worksheet_y_offset
        if len(row) == 3:
            worksheet.merge_range(i, 0, i, len(x_values), row[0][0], row[0][1])
            worksheet.merge_range(i, len(x_values) + 2, i, len(x_values) * 2 + 2, row[2][0], row[2][1])
        if len(row) == 1:
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
                    worksheet.write(i, j, cell_value, cell_format)
    worksheet.set_column(0,0, max(list(map(len, y_values))) + 4)
    worksheet.set_column(len(x_values) + 1,len(x_values) + 1, 2)
    worksheet.set_column(len(x_values) * 2 + 3,len(x_values) * 2 + 3, 2)
    if extra is not None:
        worksheet.set_column(len(x_values) + 2, len(x_values) + 2, max(list(map(len, y_values))) + 4)
    for i in range(len(x_values)):
        selected_len = int(len(x_values[i]) * 1.5) if len(x_values[i]) > 6 else len(x_values[i]) * 2
        worksheet.set_column(i + 1, i + 1, selected_len)
        if extra is not None:
            worksheet.set_column(len(x_values) + 2 + i + 1, len(x_values) + 2 + i + 1, selected_len)
    baseline_extra = ""
    if extra is not None:
        baseline_extra = "{0}{1}:{0}{2}".format(chr(66 + num_columns + 1), existing_worksheet_y_offset + 3 - 1, existing_worksheet_y_offset + 3)
    payload = {}
    payload["worksheet"] = worksheet
    payload["extra_groups"] = extra_groups
    payload["baseline_extra"] = baseline_extra
    payload["baseline_extra_format"] = simple
    payload["simple_format"] = simple
    payload["bold_italic_format"] = bold_italic
    if existing_worksheet_y_offset is not None:
        payload["new_y_offset"] = existing_worksheet_y_offset + 1 + len(y_values) + 2
    else:
        payload["new_y_offset"] = 1 + len(y_values) + 2
    return payload

def add_extra_min_max(payload: dict):
    worksheet, extra_groups, simple_format, bold_italic_format = payload["worksheet"], payload["extra_groups"], payload["simple_format"], payload["bold_italic_format"]
    new_groups = []
    for group in extra_groups:
        group_start_rows = int(re.findall(r'\d+', group.split(':')[0])[0])
        group_end_rows = int(re.findall(r'\d+', group.split(':')[1])[0])
        
        group_start_columns = ord(group.split(':')[0][: -len(re.findall(r'\d+', group.split(':')[0])[0])]) - 66
        group_end_columns = ord(group.split(':')[1][: -len(re.findall(r'\d+', group.split(':')[1])[0])]) - 66
        worksheet.write(group_start_rows - 2, group_end_columns + 3, "Maximum", bold_italic_format)
        worksheet.write(group_start_rows - 2, group_end_columns + 4, "Minimum", bold_italic_format)
        for i in range(group_start_rows - 1, group_end_rows):
            worksheet.write_formula(
                i, group_end_columns + 3, 
                "=MAX({0}{2}:{1}{2})".format(chr(group_start_columns + 66), chr(group_end_columns + 66), i + 1), 
                simple_format
            )
            worksheet.write_formula(
                i, group_end_columns + 4, 
                "=MIN({0}{2}:{1}{2})".format(chr(group_start_columns + 66), chr(group_end_columns + 66), i + 1), 
                simple_format
            )
        new_groups.append("{0}{1}:{0}{2}".format(chr(group_end_columns + 68), group_start_rows, group_end_rows))
        new_groups.append("{0}{1}:{0}{2}".format(chr(group_end_columns + 69), group_start_rows, group_end_rows))
        worksheet.set_column(group_end_columns + 4, group_end_columns + 4, 10)
        worksheet.set_column(group_end_columns + 3, group_end_columns + 3, 10)
    extra_groups.extend(new_groups)
    payload["extra_groups"] = extra_groups
    return payload

def add_colorscale_to_groups(payload: dict):
    worksheet, groups, extra_groups = payload["worksheet"], payload["groups"] if 'groups' in payload else [], payload["extra_groups"]
    for group in groups:
        worksheet.conditional_format(group, 
                                            {'type': '3_color_scale',
                                             'mid_type': 'num',
                                             'mid_value': 0,
                                             'min_color': "#ff0000",
                                             'mid_color': "#ffff00",
                                             'max_color': "#00a933"})
    for extra_group in extra_groups:
        worksheet.conditional_format(extra_group, 
                                            {'type': '3_color_scale',
                                             'mid_type': 'num',
                                             'mid_value': 0,
                                             'min_color': "#ff0000",
                                             'mid_color': "#ffff00",
                                             'max_color': "#00a933"})
    return payload

def remove_baseline_extra(payload: dict):
    worksheet, baseline_extra, baseline_extra_format = payload["worksheet"], payload["baseline_extra"], payload["baseline_extra_format"]
    if baseline_extra:
        if type(baseline_extra) == str:
            worksheet.merge_range(baseline_extra, "", baseline_extra_format)
        else:
            for single_basline_extra in baseline_extra:
                worksheet.merge_range(single_basline_extra, "", baseline_extra_format)
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
        models_name = lines[0::2]
        models_time = list(map(lambda line: float(line), lines[1::2]))
        models_name_time = zip(models_name, models_time)
    for model_name_time in models_name_time:
        name, time = model_name_time
        if name in results:
            results[name][method] = time
        else:
            results[name] = { method: time }


names = list(results.keys())
methods = sorted(methods, key=lambda column: methods_order.index(column))

Path("Excels").mkdir(exist_ok=True, parents=True)

workbook = xlsxwriter.Workbook(join("Excels", f'{Path.cwd().__str__().split("/")[-1]}-results.xlsx'))

remove_baseline_extra(
    add_colorscale_to_groups(
        add_extra_min_max(
            add_worksheet_to_workbook(
                workbook, "Average", methods, names, results, 
                extra=lambda i, j, w_o, n_r, n_c: single_dimension_models_decrease_based_on_basline_str.format(chr(66 + j).upper(), chr(66).upper(), w_o + i + 3),
                extra_value=lambda x, y: single_dimension_models_method_based_decrease_based_on_basline_val(results, y, x, 'I8-I8'),
                transform=lambda x: x
            )
        )
    )
)

workbook.close()

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
    for name in names:
        # size, size_string = size_tuple

        with open(join(dir_name, f"{name}", "stats.txt")) as stat_f:
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
            cpu_total           = stats['system.cpu.numCycles'] if 'system.cpu.numCycles' in stats else 0 # 1
            instruction_total   = stats['system.cpu.committedInsts'] if 'system.cpu.committedInsts' in stats else 0 # 1
            ipc                 = stats['system.cpu.ipc'] if 'system.cpu.ipc' in stats else 0 # 0.001

            l1d_miss_total      = stats['system.cpu.dcache.ReadReq_misses::.cpu.data'] if 'system.cpu.dcache.ReadReq_misses::.cpu.data' in stats else 0 # 1
            l1d_load_total      = stats['system.cpu.dcache.ReadReq_accesses::.cpu.data'] if 'system.cpu.dcache.ReadReq_accesses::.cpu.data' in stats else 0 # 1
            L1DCacheMissRate    = stats['system.cpu.dcache.ReadReq_miss_rate::.cpu.data'] if 'system.cpu.dcache.ReadReq_miss_rate::.cpu.data' in stats else 0 # 0.001

            l1i_miss_total      = stats['system.cpu.icache.ReadReq_misses::.cpu.inst'] if 'system.cpu.icache.ReadReq_misses::.cpu.inst' in stats else 0 # 1
            l1i_load_total      = stats['system.cpu.icache.ReadReq_accesses::.cpu.inst'] if 'system.cpu.icache.ReadReq_accesses::.cpu.inst' in stats else 0 # 1
            L1ICacheMissRate    = stats['system.cpu.icache.ReadReq_miss_rate::.cpu.inst'] if 'system.cpu.icache.ReadReq_miss_rate::.cpu.inst' in stats else 0 # 0.001

            LLDCacheAccess      = stats['system.l2.overall_accesses::.cpu.data'] if 'system.l2.overall_accesses::.cpu.data' in stats else 0 # 1
            LLDCacheMisses      = stats['system.l2.overall_misses::.cpu.data'] if 'system.l2.overall_misses::.cpu.data' in stats else 0 # 1
            LLDCacheMissRate    = stats['system.l2.overall_miss_rate::.cpu.data'] if 'system.l2.overall_miss_rate::.cpu.data' in stats else 0 # 0.001
            LLDCacheMissLatency = stats['system.l2.overall_miss_latency::.cpu.data'] if 'system.l2.overall_miss_latency::.cpu.data' in stats else 0 # 0.001

            LLICacheAccess      = stats['system.l2.overall_accesses::.cpu.inst'] if 'system.l2.overall_accesses::.cpu.inst' in stats else 0 # 1
            LLICacheMisses      = stats['system.l2.overall_misses::.cpu.inst'] if 'system.l2.overall_misses::.cpu.inst' in stats else 0 # 1
            LLICacheMissRate    = stats['system.l2.overall_miss_rate::.cpu.inst'] if 'system.l2.overall_miss_rate::.cpu.inst' in stats else 0 # 0.001

            Total_Time          = stats['sim_seconds'] if 'sim_seconds' in stats else 0 # 0.001
        
        if name in metrics_values['CPUCycles']:
            metrics_values['CPUCycles'][name][method]           = cpu_total
            metrics_values['Instructions'][name][method]        = instruction_total
            metrics_values['IPC'][name][method]                 = ipc

            metrics_values['L1DCacheMisses'][name][method]      = l1d_miss_total
            metrics_values['L1DCacheAccess'][name][method]      = l1d_load_total
            metrics_values['L1DCacheMissRate'][name][method]    = L1DCacheMissRate

            metrics_values['L1ICacheMisses'][name][method]      = l1i_miss_total
            metrics_values['L1ICacheAccess'][name][method]      = l1i_load_total
            metrics_values['L1ICacheMissRate'][name][method]    = L1ICacheMissRate

            metrics_values['LLDCacheAccess'][name][method]      = LLDCacheAccess
            metrics_values['LLDCacheMisses'][name][method]      = LLDCacheMisses
            metrics_values['LLDCacheMissRate'][name][method]    = LLDCacheMissRate
            metrics_values['LLDCacheMissLatency'][name][method] = LLDCacheMissLatency

            metrics_values['LLICacheAccess'][name][method]      = LLICacheAccess
            metrics_values['LLICacheMisses'][name][method]      = LLICacheMisses
            metrics_values['LLICacheMissRate'][name][method]    = LLICacheMissRate

            metrics_values['TotalTime'][name][method]           = Total_Time
        else:
            metrics_values['CPUCycles'][name]                   = { method: cpu_total }
            metrics_values['Instructions'][name]                = { method: instruction_total }
            metrics_values['IPC'][name]                         = { method: ipc }

            metrics_values['L1DCacheMisses'][name]              = { method: l1d_miss_total }
            metrics_values['L1DCacheAccess'][name]              = { method: l1d_load_total }
            metrics_values['L1DCacheMissRate'][name]            = { method: L1DCacheMissRate }

            metrics_values['L1ICacheMisses'][name]              = { method: l1i_miss_total }
            metrics_values['L1ICacheAccess'][name]              = { method: l1i_load_total }
            metrics_values['L1ICacheMissRate'][name]            = { method: L1ICacheMissRate }

            metrics_values['LLDCacheAccess'][name]              = { method: LLDCacheAccess }
            metrics_values['LLDCacheMisses'][name]              = { method: LLDCacheMisses }
            metrics_values['LLDCacheMissRate'][name]            = { method: LLDCacheMissRate }
            metrics_values['LLDCacheMissLatency'][name]         = { method: LLDCacheMissLatency }

            metrics_values['LLICacheAccess'][name]              = { method: LLICacheAccess }
            metrics_values['LLICacheMisses'][name]              = { method: LLICacheMisses }
            metrics_values['LLICacheMissRate'][name]            = { method: LLICacheMissRate }

            metrics_values['TotalTime'][name]                   = { method: Total_Time }

Path("Excels").mkdir(exist_ok=True, parents=True)

workbook = xlsxwriter.Workbook(join("Excels", f'{Path.cwd().__str__().split("/")[-1]}-detailed-method-based.xlsx'))
worksheet = workbook.add_worksheet()
workseet_current_y_offset = 0

metrics = list(metrics_values.keys())
metrics = sorted(metrics, key=lambda metric: metrics_sorted.index(metric))

extras = {}
transforms = {}
transform_wrong_datas = {}
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
        extras[metric] = (
            'CPU Cycles reduction against I8-I8', 
            lambda i, j, w_o, n_r, n_c: single_dimension_models_decrease_based_on_basline_str.format(chr(66 + j).upper(), chr(66).upper(), i + w_o + 3),
            lambda x, y: single_dimension_models_method_based_decrease_based_on_basline_val(metrics_values['CPUCycles'], y, x, 'I8-I8')
        )
        transforms[metric] = lambda x: x
        transform_wrong_datas[metric] = lambda x: x # lambda x: x if x != 0 else 1)
    elif 'Instructions' == metric:
        extras[metric] = (
            'Instructions reduction against I8-I8', 
            lambda i, j, w_o, n_r, n_c: single_dimension_models_decrease_based_on_basline_str.format(chr(66 + j).upper(), chr(66).upper(), i + w_o + 3),
            lambda x, y: single_dimension_models_method_based_decrease_based_on_basline_val(metrics_values['Instructions'], y, x, 'I8-I8')
        )
        transforms[metric] = lambda x: x
        transform_wrong_datas[metric] = lambda x: x # lambda x: x if x != 0 else 1)
    elif 'IPC' == metric:
        extras[metric] = (
            'IPC increase against I8-I8', 
            lambda i, j, w_o, n_r, n_c: single_dimension_models_increase_based_on_basline_str.format(chr(66 + j).upper(), chr(66).upper(), i + w_o + 3),
            lambda x, y: single_dimension_models_method_based_decrease_based_on_basline_val(metrics_values['IPC'], y, x, 'I8-I8')
        )
        transforms[metric] = lambda x: x
        transform_wrong_datas[metric] = lambda x: x # lambda x: x if x != 0 else 1)
    elif 'L1DCacheAccess' == metric:
        extras[metric] = (
            'L1 Data Cache Loads reduction against I8-I8', 
            lambda i, j, w_o, n_r, n_c: single_dimension_models_decrease_based_on_basline_str.format(chr(66 + j).upper(), chr(66).upper(), i + w_o + 3),
            lambda x, y: single_dimension_models_method_based_decrease_based_on_basline_val(metrics_values['L1DCacheAccess'], y, x, 'I8-I8')
        )
        transforms[metric] = lambda x: x
        transform_wrong_datas[metric] = lambda x: x # lambda x: x if x != 0 else 1)
    elif 'L1DCacheMisses' == metric:
        extras[metric] = (
            'L1 Data Cache Load Misses reduction against I8-I8', 
            lambda i, j, w_o, n_r, n_c: single_dimension_models_decrease_based_on_basline_str.format(chr(66 + j).upper(), chr(66).upper(), i + w_o + 3),
            lambda x, y: single_dimension_models_method_based_decrease_based_on_basline_val(metrics_values['L1DCacheMisses'], y, x, 'I8-I8')
        )
        transforms[metric] = lambda x: x
        transform_wrong_datas[metric] = lambda x: x # lambda x: x if x != 0 else 1)
    elif 'L1DCacheMissRate' == metric:
        extras[metric] = (
            'L1 Data Cache Miss Rate reduction against I8-I8', 
            lambda i, j, w_o, n_r, n_c: single_dimension_models_decrease_based_on_basline_str.format(chr(66 + j).upper(), chr(66).upper(), i + w_o + 3),
            lambda x, y: single_dimension_models_method_based_decrease_based_on_basline_val(metrics_values['L1DCacheMissRate'], y, x, 'I8-I8')
        )
        transforms[metric] = lambda x: x
        transform_wrong_datas[metric] = lambda x: x # if x != 0 else 1)
    elif 'L1ICacheAccess' == metric:
        extras[metric] = (
            'L1 Instruction Cache Loads reduction against I8-I8', 
            lambda i, j, w_o, n_r, n_c: single_dimension_models_decrease_based_on_basline_str.format(chr(66 + j).upper(), chr(66).upper(), i + w_o + 3),
            lambda x, y: single_dimension_models_method_based_decrease_based_on_basline_val(metrics_values['L1ICacheAccess'], y, x, 'I8-I8')
        )
        transforms[metric] = lambda x: x
        transform_wrong_datas[metric] = lambda x: x # lambda x: x if x != 0 else 1)
    elif 'L1ICacheMisses' == metric:
        extras[metric] = (
            'L1 Instruction Cache Load Misses reduction against I8-I8', 
            lambda i, j, w_o, n_r, n_c: single_dimension_models_decrease_based_on_basline_str.format(chr(66 + j).upper(), chr(66).upper(), i + w_o + 3),
            lambda x, y: single_dimension_models_method_based_decrease_based_on_basline_val(metrics_values['L1ICacheMisses'], y, x, 'I8-I8')
        )
        transforms[metric] = lambda x: x
        transform_wrong_datas[metric] = lambda x: x # lambda x: x if x != 0 else 1)
    elif 'L1ICacheMissRate' == metric:
        extras[metric] = (
            'L1 Instruction Cache Miss Rate reduction against I8-I8', 
            lambda i, j, w_o, n_r, n_c: single_dimension_models_decrease_based_on_basline_str.format(chr(66 + j).upper(), chr(66).upper(), i + w_o + 3),
            lambda x, y: single_dimension_models_method_based_decrease_based_on_basline_val(metrics_values['L1ICacheMissRate'], y, x, 'I8-I8')
        )
        transforms[metric] = lambda x: x
        transform_wrong_datas[metric] = lambda x: x # xif x != 0 else 1)
    elif 'LLDCacheAccess' == metric:
        extras[metric] = (
            'Last-Level Data Cache Accesses reduction against I8-I8', 
            lambda i, j, w_o, n_r, n_c: single_dimension_models_decrease_based_on_basline_str.format(chr(66 + j).upper(), chr(66).upper(), i + w_o + 3),
            lambda x, y: single_dimension_models_method_based_decrease_based_on_basline_val(metrics_values['LLDCacheAccess'], y, x, 'I8-I8')
        )
        transforms[metric] = lambda x: x
        transform_wrong_datas[metric] = lambda x: x # lambda x: x if x != 0 else 1)
    elif 'LLDCacheMisses' == metric:
        extras[metric] = (
            'Last-Level Data Cache Misses reduction against I8-I8', 
            lambda i, j, w_o, n_r, n_c: single_dimension_models_decrease_based_on_basline_str.format(chr(66 + j).upper(), chr(66).upper(), i + w_o + 3),
            lambda x, y: single_dimension_models_method_based_decrease_based_on_basline_val(metrics_values['LLDCacheMisses'], y, x, 'I8-I8')
        )
        transforms[metric] = lambda x: x
        transform_wrong_datas[metric] = lambda x: x # lambda x: x if x != 0 else 1)
    elif 'LLDCacheMissRate' == metric:
        extras[metric] = (
            'Last-Level Data Cache Miss Rate reduction against I8-I8', 
            lambda i, j, w_o, n_r, n_c: single_dimension_models_decrease_based_on_basline_str.format(chr(66 + j).upper(), chr(66).upper(), i + w_o + 3),
            lambda x, y: single_dimension_models_method_based_decrease_based_on_basline_val(metrics_values['LLDCacheMissRate'], y, x, 'I8-I8')
        )
        transforms[metric] = lambda x: x
        transform_wrong_datas[metric] = lambda x: x # lambda x: x if x != 0 else 1)
    elif 'LLDCacheMissLatency' == metric:
        extras[metric] = (
            'Last-Level Data Cache Miss Latency reduction against I8-I8', 
            lambda i, j, w_o, n_r, n_c: single_dimension_models_decrease_based_on_basline_str.format(chr(66 + j).upper(), chr(66).upper(), i + w_o + 3),
            lambda x, y: single_dimension_models_method_based_decrease_based_on_basline_val(metrics_values['LLDCacheMissLatency'], y, x, 'I8-I8')
        )
        transforms[metric] = lambda x: x
        transform_wrong_datas[metric] = lambda x: x # lambda x: x if x != 0 else 1)
    elif 'LLICacheAccess' == metric:
        extras[metric] = (
            'Last-Level Instructions Cache Accesses reduction against I8-I8', 
            lambda i, j, w_o, n_r, n_c: single_dimension_models_decrease_based_on_basline_str.format(chr(66 + j).upper(), chr(66).upper(), i + w_o + 3),
            lambda x, y: single_dimension_models_method_based_decrease_based_on_basline_val(metrics_values['LLICacheAccess'], y, x, 'I8-I8')
        )
        transforms[metric] = lambda x: x
        transform_wrong_datas[metric] = lambda x: x # lambda x: x if x != 0 else 1)
    elif 'LLICacheMisses' == metric:
        extras[metric] = (
            'Last-Level Instructions Cache Misses reduction against I8-I8', 
            lambda i, j, w_o, n_r, n_c: single_dimension_models_decrease_based_on_basline_str.format(chr(66 + j).upper(), chr(66).upper(), i + w_o + 3),
            lambda x, y: single_dimension_models_method_based_decrease_based_on_basline_val(metrics_values['LLICacheMisses'], y, x, 'I8-I8')
        )
        transforms[metric] = lambda x: x
        transform_wrong_datas[metric] = lambda x: x # lambda x: x if x != 0 else 1)
    elif 'LLICacheMissRate' == metric:
        extras[metric] = (
            'Last-Level Instructions Cache Miss Rate reduction against I8-I8', 
            lambda i, j, w_o, n_r, n_c: single_dimension_models_decrease_based_on_basline_str.format(chr(66 + j).upper(), chr(66).upper(), i + w_o + 3),
            lambda x, y: single_dimension_models_method_based_decrease_based_on_basline_val(metrics_values['LLICacheMissRate'], y, x, 'I8-I8')
        )
        transforms[metric] = lambda x: x
        transform_wrong_datas[metric] = lambda x: x # lambda x: x if x != 0 else 1)
    elif 'TotalTime' == metric:
        extras[metric] = (
            'Total Execution Time reduction against I8-I8', 
            lambda i, j, w_o, n_r, n_c: single_dimension_models_decrease_based_on_basline_str.format(chr(66 + j).upper(), chr(66).upper(), i + w_o + 3),
            lambda x, y: single_dimension_models_method_based_decrease_based_on_basline_val(metrics_values['TotalTime'], y, x, 'I8-I8')
        )
        transforms[metric] = lambda x: x
        transform_wrong_datas[metric] = lambda x: x # lambda x: x if x != 0 else 1)

for metric in metrics:
    workseet_current_y_offset = \
    remove_baseline_extra(
        add_colorscale_to_groups(
            add_extra_min_max(
                add_worksheet_to_workbook(
                    workbook, metric, methods, names, metrics_values[metric], 
                    extra=extras[metric][1], extra_value=extras[metric][2], extra_title=extras[metric][0],
                    transform=transforms[metric], transform_wrong_data=transform_wrong_datas[metric],
                    existing_worksheet=worksheet, existing_worksheet_y_offset=workseet_current_y_offset
                    # highlight_area_extra=highlight_area, 
                )
            )
        )
    )["new_y_offset"]

workbook.close()






