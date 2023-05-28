from math import floor


decrease_based_on_basline_str   = "=IFERROR(ROUND((({0}{1} - {0}{2}) / {0}{1}) * 100,2), 0)"
increase_based_on_basline_str   = "=IFERROR(ROUND((({0}{2} - {0}{1}) / {0}{1}) * 100,2), 0)"
speedup_based_on_basline_str    = "=IFERROR(ROUND(({0}{1} / {0}{2}),4), 0)"
speeddown_based_on_basline_str  = "=IFERROR(ROUND(({0}{2} / {0}{1}),4), 0)"
single_dimension_models_decrease_based_on_basline_str = "=IFERROR(ROUND((({1}{2} - {0}{2}) / {1}{2}) * 100,2), 0)"
single_dimension_models_increase_based_on_basline_str = "=IFERROR(ROUND((({0}{2} - {1}{2}) / {1}{2}) * 100,2), 0)"
single_dimension_models_on_x_axis_decrease_based_on_basline_str = "=IFERROR(ROUND((({0}{1} - {0}{2}) / {0}{1}) * 100,2), 0)"
single_dimension_models_on_x_axis_increase_based_on_basline_str = "=IFERROR(ROUND((({0}{2} - {0}{1}) / {0}{1}) * 100,2), 0)"
single_dimension_models_on_x_axis_speedup_based_on_basline_str  = "=IFERROR(ROUND(({0}{1} / {0}{2}) * 100,2), 0)"
single_dimension_models_on_x_axis_speeddow_based_on_basline_str = "=IFERROR(ROUND(({0}{2} / {0}{1}) * 100,2), 0)"
decrease_kernel_share_based_on_basline_str = "=IFERROR(ROUND(((({0}{1} * {3}!${0}${1}) - ({0}{2} * {3}!${0}${2})) / ({0}{1} * {3}!${0}${1})) * 100,2), 0)"
increase_kernel_share_based_on_basline_str = "=IFERROR(ROUND(((({0}{2} * {3}!${0}${2}) - ({0}{1} * {3}!${0}${1})) / ({0}{1} * {3}!${0}${1})) * 100,2), 0)"

method_based_decrease_based_on_basline_str  = "=IFERROR(ROUND((({3}!{0}{1} - {0}{2}) / {3}!{0}{1}) * 100,2), 0)"
method_based_increase_based_on_basline_str  = "=IFERROR(ROUND((({0}{2} - {3}!{0}{1}) / {3}!{0}{1}) * 100,2), 0)"
method_based_speedup_based_on_basline_str   = "=IFERROR(ROUND(({3}!{0}{1} / {0}{2}),4), 0)"
method_based_speeddown_based_on_basline_str = "=IFERROR(ROUND(({0}{2} / {3}!{0}{1}),4), 0)"
method_based_decrease_based_on_basline_val = lambda values, batch_size, input_size, output_size, method, baseline : \
    ((values[f"{batch_size}x{input_size}x{output_size}"][baseline] - values[f"{batch_size}x{input_size}x{output_size}"][method]) / \
      values[f"{batch_size}x{input_size}x{output_size}"][baseline]) * 100
method_based_increase_based_on_basline_val = lambda values, batch_size, input_size, output_size, method, baseline : \
    ((values[f"{batch_size}x{input_size}x{output_size}"][method] - values[f"{batch_size}x{input_size}x{output_size}"][baseline]) / \
      values[f"{batch_size}x{input_size}x{output_size}"][baseline]) * 100
method_based_speedup_based_on_basline_val = lambda values, batch_size, input_size, output_size, method, baseline : \
    (values[f"{batch_size}x{input_size}x{output_size}"][baseline] / \
      values[f"{batch_size}x{input_size}x{output_size}"][method]) * 100
method_based_speeddown_based_on_basline_val = lambda values, batch_size, input_size, output_size, method, baseline : \
    (values[f"{batch_size}x{input_size}x{output_size}"][method] / \
      values[f"{batch_size}x{input_size}x{output_size}"][baseline]) * 100
single_dimension_models_method_based_decrease_based_on_basline_val = lambda values, model, method, baseline : \
    ((values[model][baseline] - values[model][method]) / values[model][baseline]) * 100
single_dimension_models_method_based_increase_based_on_basline_val = lambda values, model, method, baseline : \
    ((values[model][method] - values[model][baseline]) / values[model][baseline]) * 100
method_based_decrease_kernel_share_based_on_basline_str = "=IFERROR(ROUND(((({3}!{0}{1} * {3}!${0}${2}) - ({0}{1} * {0}{2})) / ({3}!{0}{1} * {3}!${0}${2})) * 100,2), 0)"
method_based_increase_kernel_share_based_on_basline_str = "=IFERROR(ROUND(((({0}{1} * {0}{2}) - ({3}!{0}{1} * {3}!${0}${2})) / ({3}!{0}{1} * {3}!${0}${2})) * 100,2), 0)"

methods_order = [
    'I8-I8',
    # Special benchmarks
    'XNNPack',
    'No-Caching',
    'GEMMLOWP',
    'FP32',
    'No-Caching-FP32',
    'XNNPack-FP32',
    'Eigen',
    'Binary-Binary-XOR',
    # ULPPACK benchmarks
    'ULPPACK-W1A1',
    'ULPPACK-W2A2',
    'ULPPACK-W3A3',
    'ULPPACK-W4A4',
    # 'ULPPACK-W5A5',
    # 'ULPPACK-W6A6',
    # 'ULPPACK-W7A7',
    # Our benchmarks
    'I8-I4',
    'I4-I8',
    'I4-I4',
    'I8-Ternary',
    'Ternary-I8',
    'Ternary-Ternary',
    'I8-Binary',
    'Binary-I8',
    'Binary-Binary',
    'I8-I4-without-selection',
    # Our New benchmarks
    'SelfDependent-W4A4',
    'BarrelShift-Mul-W8A8',
]
metrics_order = [
    'cpu_cycles',
    'instructions',
    'IPC',

    'l1d_loads',
    'l1d_misses',
    'L1DCacheMissRate',
    'L2DCacheAccess',
    'L2DCacheMisses',
    'L2DCacheMissRate',
    'LLDCacheAccess',
    'LLDCacheMisses',
    'LLDCacheMissRate',

    'l1i_loads',
    'l1i_misses',
    'L1ICacheMissRate',
    'L2ICacheAccess',
    'L2ICacheMisses',
    'L2ICacheMissRate',
    'L2DCacheMissLatency',
    'LLICacheAccess',
    'LLICacheMisses',
    'LLICacheMissRate',
    'LLDCacheMissLatency',

    'MemoryBandwidth',

    'time_total',

    'cpu_cycles_kernel_share',
    'instructions_kernel_share',
    'l1d_loads_kernel_share',
    'l1d_misses_kernel_share',

    'cpu_cycles_method_share',
    'instructions_method_share',
    'l1d_loads_method_share',
    'l1d_misses_method_share',

    'cpu_cycles_pack_share',
    'instructions_pack_share',
    'l1d_loads_pack_share',
    'l1d_misses_pack_share',
]
metrics_sorted = [
    'CPUCycles',
    'Instructions',
    'IPC',

    'L1DCacheAccess',
    'L1DCacheMisses',
    'L1DCacheMissRate',

    'L2DCacheAccess',
    'L2DCacheMisses',
    'L2DCacheMissRate',
    'L2DCacheMissLatency',

    'LLDCacheAccess',
    'LLDCacheMisses',
    'LLDCacheMissRate',
    'LLDCacheMissLatency',

    'L1ICacheAccess',
    'L1ICacheMisses',
    'L1ICacheMissRate',

    'L2ICacheAccess',
    'L2ICacheMisses',
    'L2ICacheMissRate',

    'LLICacheAccess',
    'LLICacheMisses',
    'LLICacheMissRate',

    'MemoryBandwidth',

    'TotalTime',
]
metrics_base_reletive_positions = {
    'cpu_cycles': 0,
    'instructions': 1,
    'l1d_loads': 2,
    'l1d_misses': 3,
    'IPC': 4,
    'L1DCacheMissRate': 5,
    'LLDCacheMissRate': 6,
    'time_total': 7,

    'cpu_cycles_kernel_share': 8,
    'instructions_kernel_share': 9,
    'l1d_loads_kernel_share': 10,
    'l1d_misses_kernel_share': 11,

    'cpu_cycles_method_share': 12,
    'instructions_method_share': 13,
    'l1d_loads_method_share': 14,
    'l1d_misses_method_share': 15,

    'cpu_cycles_pack_share': 16,
    'instructions_pack_share': 17,
    'l1d_loads_pack_share': 18,
    'l1d_misses_pack_share': 19,
}
multibatch_method_kernels = {
    "I8-I8": "ruy::Kernel8bitNeon(",
    "I8-I4": "LowPrecision::FullyConnected::Int4::MultiplyInt8MultiBatched(",
    "I8-I4-without-selection": "LowPrecision::FullyConnected::Int4::MultiplyInt8MultiBatched(",
    "I4-I8": "LowPrecision::FullyConnected::Int4InputsInt8Weights::MultiplyInt8MultiBatched(",
    "I4-I4": "LowPrecision::FullyConnected::Int4InputsInt4Weights::MultiplyInt8MultiBatched(",
    "I8-Ternary": "LowPrecision::FullyConnected::Ternary::MultiplyInt8MultiBatched(",
    "Ternary-I8": "LowPrecision::FullyConnected::TernaryInputsInt8Weights::MultiplyInt8MultiBatched(",
    "Ternary-Ternary": "LowPrecision::FullyConnected::TernaryInputsTernaryWeights::MultiplyInt8MultiBatched(",
    "I8-Binary": "LowPrecision::FullyConnected::Binary::MultiplyInt8MultiBatched(",
    "Binary-I8": "LowPrecision::FullyConnected::BinaryInputsInt8Weights::MultiplyInt8MultiBatched(",
    "Binary-Binary": "LowPrecision::FullyConnected::BinaryInputsBinaryWeights::MultiplyInt8MultiBatched(",
    "Binary-Binary-XOR": "LowPrecision::FullyConnected::BinaryInputsBinaryWeightsXOR::MultiplyInt8MultiBatched(",
}
singlebatch_method_kernels = {
    "I8-I8": "ruy::Kernel8bitNeon1Col(",
    "I8-I4": "LowPrecision::FullyConnected::Int4::MultiplyInt8SingleBatch(",
    "I8-I4-without-selection": "LowPrecision::FullyConnected::Int4::MultiplyInt8SingleBatch(",
    "I4-I8": "LowPrecision::FullyConnected::Int4InputsInt8Weights::MultiplyInt8SingleBatch(",
    "I4-I4": "LowPrecision::FullyConnected::Int4InputsInt4Weights::MultiplyInt8SingleBatch(",
    "I8-Ternary": "LowPrecision::FullyConnected::Ternary::MultiplyInt8SingleBatch(",
    "Ternary-I8": "LowPrecision::FullyConnected::TernaryInputsInt8Weights::MultiplyInt8SingleBatch(",
    "Ternary-Ternary": "LowPrecision::FullyConnected::TernaryInputsTernaryWeights::MultiplyInt8SingleBatch(",
    "I8-Binary": "LowPrecision::FullyConnected::Binary::MultiplyInt8SingleBatch(",
    "Binary-I8": "LowPrecision::FullyConnected::BinaryInputsInt8Weights::MultiplyInt8SingleBatch(",
    "Binary-Binary": "LowPrecision::FullyConnected::BinaryInputsBinaryWeights::MultiplyInt8SingleBatch(",
    "Binary-Binary-XOR": "LowPrecision::FullyConnected::BinaryInputsBinaryWeightsXOR::MultiplyInt8SingleBatch(",
}
multibatch_method_namespace = {
    "I8-I8": "ruy::",
    "I8-I4": "LowPrecision::FullyConnected::",
    "I8-I4-without-selection": "LowPrecision::FullyConnected::",
    "I4-I8": "LowPrecision::FullyConnected::",
    "I4-I4": "LowPrecision::FullyConnected::",
    "I8-Ternary": "LowPrecision::FullyConnected::",
    "Ternary-I8": "LowPrecision::FullyConnected::",
    "Ternary-Ternary": "LowPrecision::FullyConnected::",
    "I8-Binary": "LowPrecision::FullyConnected::",
    "Binary-I8": "LowPrecision::FullyConnected::",
    "Binary-Binary": "LowPrecision::FullyConnected::",
    "Binary-Binary-XOR": "LowPrecision::FullyConnected::",
}
singlebatch_method_namespace = {
    "I8-I8": "ruy::",
    "I8-I4": "LowPrecision::FullyConnected::",
    "I8-I4-without-selection": "LowPrecision::FullyConnected::",
    "I4-I8": "LowPrecision::FullyConnected::",
    "I4-I4": "LowPrecision::FullyConnected::",
    "I8-Ternary": "LowPrecision::FullyConnected::",
    "Ternary-I8": "LowPrecision::FullyConnected::",
    "Ternary-Ternary": "LowPrecision::FullyConnected::",
    "I8-Binary": "LowPrecision::FullyConnected::",
    "Binary-I8": "LowPrecision::FullyConnected::",
    "Binary-Binary": "LowPrecision::FullyConnected::",
    "Binary-Binary-XOR": "LowPrecision::FullyConnected::",
}
multibatch_method_packs = {
    "I8-I8": "ruy::Pack8bitColMajorForNeon",
    "I8-I4": "LowPrecision::FullyConnected::Int4::QuantizeInput",
    "I8-I4-without-selection": "LowPrecision::FullyConnected::Int4::QuantizeInput",
    "I4-I8": "LowPrecision::FullyConnected::Int4InputsInt8Weights::QuantizeInput",
    "I4-I4": "LowPrecision::FullyConnected::Int4InputsInt4Weights::QuantizeInput",
    "I8-Ternary": "LowPrecision::FullyConnected::Ternary::QuantizeInput",
    "Ternary-I8": "LowPrecision::FullyConnected::TernaryInputsInt8Weights::QuantizeInput",
    "Ternary-Ternary": "LowPrecision::FullyConnected::TernaryInputsTernaryWeights::QuantizeInput",
    "I8-Binary": "LowPrecision::FullyConnected::Binary::QuantizeInput",
    "Binary-I8": "LowPrecision::FullyConnected::BinaryInputsInt8Weights::QuantizeInput",
    "Binary-Binary": "LowPrecision::FullyConnected::BinaryInputsBinaryWeights::QuantizeInput",
    "Binary-Binary-XOR": "LowPrecision::FullyConnected::BinaryInputsBinaryWeightsXOR::QuantizeInput",
}
singlebatch_method_packs = {
    "I8-I8": "ruy::Pack8bitColMajorForNeon",
    "I8-I4": "LowPrecision::FullyConnected::Int4::QuantizeInput",
    "I8-I4-without-selection": "LowPrecision::FullyConnected::Int4::QuantizeInput",
    "I4-I8": "LowPrecision::FullyConnected::Int4InputsInt8Weights::QuantizeInput",
    "I4-I4": "LowPrecision::FullyConnected::Int4InputsInt4Weights::QuantizeInput",
    "I8-Ternary": "LowPrecision::FullyConnected::Ternary::QuantizeInput",
    "Ternary-I8": "LowPrecision::FullyConnected::TernaryInputsInt8Weights::QuantizeInput",
    "Ternary-Ternary": "LowPrecision::FullyConnected::TernaryInputsTernaryWeights::QuantizeInput",
    "I8-Binary": "LowPrecision::FullyConnected::Binary::QuantizeInput",
    "Binary-I8": "LowPrecision::FullyConnected::BinaryInputsInt8Weights::QuantizeInput",
    "Binary-Binary": "LowPrecision::FullyConnected::BinaryInputsBinaryWeights::QuantizeInput",
    "Binary-Binary-XOR": "LowPrecision::FullyConnected::BinaryInputsBinaryWeightsXOR::QuantizeInput",
}

def num_column_to_spreadsheet_letter(num_column: int) -> str:
    num_column_ = num_column
    output_string = ""
    if num_column < 26:
        output_string = chr(65 + num_column)
    else:
        while num_column_ >= 0:
            current_letter = num_column_% 26
            num_column_ = floor(num_column_ / 26) - 1
            output_string = chr(65 + current_letter) + output_string
    return output_string

def letter_column_spreadsheet_to_number(column_letters: str) -> int:
    column_letters_ = column_letters.upper()
    output_number = 0
    if len(column_letters_) == 1:
        output_number = ord(column_letters_) - 65
    else:
        current_coeff = 1
        current_offset = 0
        while len(column_letters_) > 0:
            current_letter = column_letters_[-1]
            output_number += (ord(current_letter) - 65 + current_offset) * current_coeff
            current_coeff *= 26
            current_offset = 1
            column_letters_ = column_letters_[:-1]
    return output_number

tex_good_color = "maxGreen"
tex_neutral_color = "white"
tex_bad_color = "red"

tex_figureStar_template = """
\\begin{{figure*}}[t]
\t{FUNCTIONS_STRING}
\t\\centering
\t{CONSTS_STRING}
{SUBFIGURES_STRING}
\t\\caption{{
\t\t\\centering
\t\t{CAPTION_STRING} 
\t}}
\t\\label{{fig:{FIGURE_LABEL_STRING}}}
\\end{{figure*}}
"""

tex_subfig_template = """\t\\begin{{subfigure}}[b]{{0.245\\linewidth}}
\t\t\\centering
\t\t\\begin{{tikzpicture}}[
\t\t\tNode/.style = {{minimum width=\\MinWidth cm, minimum height=\\MinWidth cm, inner sep=0,outer sep=0}},
\t\t]
{X_AXIS_LABELS_STRING}
{Y_AXIS_LABELS_STRING}
{SUBFIGURE_CONTEXT_STRING}
{X_AXIS_STRING}
{Y_AXIS_STRING}
\t\t\end{{tikzpicture}}
\t\t\\caption{{
\t\t\t\\centering
\t\t\t{SUBFIGURE_CAPTION_STRING}
\t\t}}
\t\t\\label{{fig:{SUBFIGURE_LABEL_STRING}}}
\t\\end{{subfigure}}"""

tex_subfigure_context_x_axis_label_template = "\\node[Node] at (\\baseX + {AXIS_IDX} * \\MinWidth,\\baseY + \\MinWidth) {{\\scalebox{{\\MinWidth}}{{{AXIS_LABEL}}}}};"
tex_subfigure_context_y_axis_label_template = "\\node[Node] at (\\baseX - \\MinWidth,\\baseY - {AXIS_IDX} * \\MinWidth) {{\\scalebox{{\\MinWidth}}{{{AXIS_LABEL}}}}};"

tex_subfigure_context_x_axis_template = """\\draw[->] (-1 * \\MinWidth / 2,\\MinWidth / 2) -- (\\MinWidth * {NUM} - \\MinWidth / 2,\\MinWidth / 2);
\t\t\t\\node[minimum height=\\MinWidth cm, inner sep=0,outer sep=0] at (\\MinWidth * {NUM} / 2 - \\MinWidth / 2,0.75) {{\\scalebox{{\\MinWidth}}{{Input Size}}}};"""

tex_subfigure_context_y_axis_template = """\\draw[->] (-1 * \\MinWidth / 2,\\MinWidth / 2) -- (-1 * \\MinWidth / 2,-1 * \\MinWidth * {NUM} + \\MinWidth / 2);
\t\t\t\\node[minimum height=\\MinWidth cm, inner sep=0,outer sep=0] at (-0.2,-1 * \\MinWidth * {NUM}) {{\\scalebox{{\\MinWidth}}{{Output Size}}}};"""

tex_figure_constants = [
    ("HighlighColor", "black"),
    ("MaxNumber", "{MAX}"),
    ("MidNumber", "{MID}"),
    ("MinNumber", "{MIN}"),
    ("MaxColor", "{MAX_COLOR}"),
    ("MidColor", "{MID_COLOR}"),
    ("MinColor", "{MIN_COLOR}"),
    ("baseX", "0"),
    ("baseY", "0"),
    ("MinWidth", "0.5"),
]
# pyright: reportInvalidStringEscapeSequence=false
tex_functions = [
    (
        "CreateGradientColorCell",
        "3",
        """
\t\t\\ifdim #1 pt > \\MidNumber pt
\t\t\t\\pgfmathsetmacro{\\PercentColor}{max(min(100.0*(#1 - \\MidNumber)/(\\MaxNumber-\\MidNumber),100.0),0.00)} %
\t\t\t\\node[draw=#3, Node, fill=\\MaxColor!\\PercentColor!\\MidColor] at #2 {\\scalebox{0.5}{#1}};
\t\t\\else
\t\t\t\\pgfmathsetmacro{\\PercentColor}{max(min(100.0*(\\MidNumber - #1)/(\\MidNumber-\\MinNumber),100.0),0.00)} %
\t\t\t\\node[draw=#3, Node, fill=\\MinColor!\\PercentColor!\\MidColor] at #2 {\\scalebox{0.5}{#1}};
\t\t\\fi
\t""",
        "\\CreateGradientColorCell{{{VALUE}}}{{(\\baseX + {X_IDX} * \\MinWidth,\\baseY - 0.0)}}{{{BORDER_STYLE}}}",
    ),
    (
        "CreateGradientColorCellWithName",
        "5",
        """
\t\t\\ifdim #1 pt > \\MidNumber pt
\t\t\t\\pgfmathsetmacro{\\PercentColor}{max(min(100.0*(#1 - \\MidNumber)/(\\MaxNumber-\\MidNumber),100.0),0.00)} %
\t\t\t\\node[draw=#3, #5, fill=\\MaxColor!\\PercentColor!\\MidColor] at #2 {\\setstretch{0.8}\\scalebox{0.6}{#4}\\\\\\scalebox{0.6}{#1}};
\t\t\\else
\t\t\t\\pgfmathsetmacro{\\PercentColor}{max(min(100.0*(\\MidNumber - #1)/(\\MidNumber-\\MinNumber),100.0),0.00)} %
\t\t\t\\node[draw=#3, #5, fill=\\MinColor!\\PercentColor!\\MidColor] at #2 {\\setstretch{0.8}\\scalebox{0.6}{#4}\\\\\\scalebox{0.6}{#1}};
\t\t\\fi
\t""",
        "",
    ),
    (
        "CreateLogTwoGradientColorCell",
        "3",
        """
\t\t\\ifdim #1 pt > \\MidNumber pt
\t\t\t\\pgfmathsetmacro{\\PercentColor}{max(min(100.0*(#1 - \\MidNumber)/(\\MaxNumber-\\MidNumber),100.0),0.00)} %
\t\t\t\\node[draw=#3, Node, fill=\\MaxColor!\\PercentColor!\\MidColor] at #2 {\\scalebox{0.5}{$2^{#1}$}};
\t\t\\else
\t\t\t\\pgfmathsetmacro{\\PercentColor}{max(min(100.0*(\\MidNumber - #1)/(\\MidNumber-\\MinNumber),100.0),0.00)} %
\t\t\t\\node[draw=#3, Node, fill=\\MinColor!\\PercentColor!\\MidColor] at #2 {\\scalebox{0.5}{$2^{#1}$}};
\t\t\\fi
\t""",
        "",
    ),
]
tex_method_list_sorted = [
    "I8-I4",
    "I4-I8",
    "I4-I4",
    "Ternary-Ternary",
    "Binary-Binary",
    'XNNPack',
    'No-Caching',
    'GEMMLOWP',
    'FP32',
    'XNNPack-FP32',
    'No-Caching-FP32',
    'Eigen',
    'ULPPACK-W1A1',
    'ULPPACK-W2A2',
    'ULPPACK-W3A3',
]
tex_method_to_subcaption = {
    "I8-I4": "\\ourmethod{} for $W4A8$",
    "I4-I8": "\\ourmethod{} for $W8A4$",
    "I4-I4": "\\ourmethod{} for $W4A4$",
    "Ternary-Ternary": "\\ourmethod{} for $W2A2$",
    "Binary-Binary": "\\ourmethod{} for $W1A1$",
    'XNNPack': "\\xnnpackint{}",
    'No-Caching': "\\tfliteint{}",
    'GEMMLOWP': "\\gemmlowpint{}",
    'FP32': "\\ruyfp{}",
    'XNNPack-FP32': "\\xnnpackfp{}",
    'No-Caching-FP32': "\\tflitefp{}",
    'Eigen': "\\eigenfp{}",
    'ULPPACK-W1A1': "\\ulppackWA{1}",
    'ULPPACK-W2A2': "\\ulppackWA{2}",
    'ULPPACK-W3A3': "\\ulppackWA{3}",
}
tex_method_to_label = {
    "I8-I4": "ourW4A8",
    "I4-I8": "w8a4",
    "I4-I4": "w4a4",
    "Ternary-Ternary": "w2a2",
    "Binary-Binary": "w1a1",
    'XNNPack': "xnnpack-w8a8",
    'No-Caching': "tflite-w8a8",
    'GEMMLOWP': "gemmlowp-w8a8",
    'FP32': "ruy-fp32",
    'XNNPack-FP32': "xnnpack-fp32",
    'No-Caching-FP32': "tflite-fp32",
    'Eigen': "eigen-fp32",
    'ULPPACK-W1A1': "ulppack-w1a1",
    'ULPPACK-W2A2': "ulppack-w2a2",
    'ULPPACK-W3A3': "ulppack-w3a3",
}

