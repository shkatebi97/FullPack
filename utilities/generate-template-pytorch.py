#!/usr/bin/env python
"""

"""

import os
from torch import rand, jit
from torch.nn import Sequential, ReLU, Linear, Flatten, Module
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import optparse

parser = optparse.OptionParser()

parser.add_option('-a', '--add-size',
    action="append", dest="sizes",
    help="Define model sizes", default=[])
parser.add_option('-z', '--single-size',
    action="store", dest="single_size",
    help="Define single size for all layers of model. Must be used with '--repeat-for' flag.", default="")
parser.add_option('-r', '--repeat-for',
    action="store", dest="repeat_for",
    help="Define number of repeats of single size. Must be used with '--single-size' flag.", default="")
parser.add_option('-i', '--input-size',
    action="store", dest="input_size",
    help="Define input size. Default is the first element of '--add-size'.", default="")
parser.add_option('-b', '--num-batches',
    action="store", dest="num_batches",
    help="Set number of batches", default=1)
parser.add_option('-d', '--deepspeech',
    action="store_true", dest="deepspeech",
    help="Activate the deepspeech mode", default=False)
parser.add_option('-t', '--resnet',
    action="store_true", dest="resnet",
    help="Activate the resnet mode", default=False)
parser.add_option('-o', '--output',
    action="store", dest="output_path",
    help="Set the path to store the output '.tflite' file.", default="model.tflite")

options, args = parser.parse_args()

device = "cpu"

model_sizes = []
input_size = 10

if not options.deepspeech and not options.resnet:
    if options.single_size and options.repeat_for:
        for i in range(int(options.repeat_for)):
            model_sizes.append(int(options.single_size))
        if options.input_size:
            input_size = int(options.input_size)
        else:
            input_size = int(options.single_size)
    else:
        if len(options.sizes) < 2 and not options.input_size:
            print(f"You must pass at least two sizes, one size and input size, or use single size with repeat.")
            exit(-1)
        if options.input_size:
            input_size = int(options.input_size)
        else:
            input_size = int(options.sizes[0])
        for i in options.sizes[0 if options.input_size else 1:]:
            model_sizes.append(int(i))
    

    model = Sequential().to(device=device)
    model_sizes = [input_size, *model_sizes]
    model_sizes = [(model_sizes[i], model_sizes[i + 1]) for i in range(len(model_sizes) - 1)]
    for size in model_sizes:
        model.append(Linear(size[0], size[1]).to(device=device))
        model.append(ReLU().to(device=device))
    print(model)
    sample_input = rand(int(options.num_batches), model_sizes[0][1]).to(device=device)
    ready_for_save_model = jit.trace(model, sample_input)
    print(ready_for_save_model)
    ready_for_save_model.save(options.output_path)
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # tflite_model = converter.convert()
    # with open(options.output_path, "wb") as f:
    #     f.write(tflite_model)
elif options.resnet:
    model = models.resnet18(pretrained=True)
    model.eval()
    example = rand(1, 3, 224, 224)
    traced_script_module = jit.trace(model, example)
    traced_script_module.save(options.output_path)
else:
    # model = Sequential()
    # model.add(Input(shape=(494), batch_size=int(options.num_batches)))
    # model.add(Dense(2048, activation='relu'))
    # model.add(Dense(2048, activation='relu'))
    # model.add(Dense(2048, activation='relu'))
    # model.add(Reshape((-1, 2048)))
    # model.add(LSTM(2048, activation='relu', batch_size=1))
    # model.add(Dense(2048, activation='relu'))
    # model.add(Dense(28, activation='softmax'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # tflite_model = converter.convert()
    # with open(options.output_path, "wb") as f:
    #     f.write(tflite_model)
    pass







