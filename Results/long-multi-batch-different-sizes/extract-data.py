#!/usr/bin/env python
from os.path import isfile, join, splitext, isdir
from os import listdir, times
from pprint import pprint

result_files = listdir(".")
result_files = list(filter(isfile, result_files))
result_files = list(filter(lambda file_name: "result" in file_name and ".log" in file_name, result_files))
result_methods_files = list(map(lambda x: (x.split("result-")[1].split(".log")[0], x), result_files))
results = {}
methods = []
for result_method_file in result_methods_files:
    method, file_name = result_method_file
    methods.append(method)
    with open(file_name) as file:
        lines = list(map(lambda x: x[:-1], file.readlines()[1:]))
        models_size = list(map(lambda line: "{}x{}^{}".format(line.split("model-")[1].split("-")[0], line.split("x")[1].split(".tflite")[0], int(line.split("batch-")[1].split("x")[0]) + 1), lines[0::2]))
        models_time = list(map(lambda line: float(line.split("avg=")[1]), lines[1::2]))
        models_size_time = zip(models_size, models_time)
    for model_size_time in models_size_time:
        size, time = model_size_time
        if size in results:
            results[size][method] = time
        else:
            results[size] = { method: time }
# pprint(results)

print("Sizes", end=",")
for method in methods:
    print(method, end=",")
print()

for size in results.keys():
    print(size, end=",")
    for method in methods:
        print(results[size][method], end=",")
    print()

    
