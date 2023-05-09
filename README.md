# FullPack

This repository contains FullPack codes for only ARMv8 (`aarch64`) architecture.

To use FullPack, you have two options:

1. Use FullPack as a shared library.
1. Use the provided tool to test and benchmark the FullPack.

We will start with the first options.

## Use FullPack as a shared library
To use FullPack as a library you should first build the shared library.

### Build

You should be able to build the shared library with the following commands (clone this repository first, and then build the shared library):
```bash
cd FullPack
make -j `nproc` libfullpack.so
```
To enable the debug mode (will build with `-g` options and no `-O3`) you can set the `DEBUG` variable to `1`:
```bash
make -j `nproc` DEBUG=1 libfullpack.so
```
Please remember to clean the build if you have already built with other options. If you already built with no `DEBUG` enabled, you should run this instead:
```bash
make clean
make -j `nproc` DEBUG=1 libfullpack.so
```

This will create some intermediate object files and a `libfullpack.so` in the main path.

### Usage
First, you should move the already built `.so` file to a path inside the `PATH` variable or add the path of the file to `LD_LIBRARY_PATH` environmental variable.
After that, if you want to build an executable you need to add the path of [low_precision_fully_connected.h](low_precision_fully_connected.h) file to your include path (or move it to a path the compiler can find it).

The following code will add the required files system-wide:
```bash
sudo cp libfullpack.so /usr/aarch64-linux-gnu/lib/
sudo cp low_precision_fully_connected.h /usr/aarch64-linux-gnu/include/
```

There will be eventually a document available, but unfortunately, there are not any available right now.
However, you can take a look at the [header](low_precision_fully_connected.h) file to figure out the API.

## Use the provided tool to test and benchmark the FullPack

To the tool, you need to install `bazel` to build `Ruy` which is used as the baseline.
We have successfully tested the Bazel version `4.2.2` and we recommend this version.
However, you may be able to work with higher versions as well but we don't provide any specific support for other versions except `4.2.2`.

### Build
You can build this tool (after cloning this repository) with the following commands:
```bash
cd FullPack
make -j `nproc`
```
To enable the debug mode (will build with `-g` options and no `-O3`) you can set the `DEBUG` variable to `1`:
```bash
make -j `nproc` DEBUG=1
```
Please remember to clean the build if you have already built with other options. If you already built with no `DEBUG` enabled, you should run this instead:
```bash
make clean
make -j `nproc` DEBUG=1
```

### Usage

The provided tool will be built statically, so you don't need to dynamically link it with other libraries and can be used as a standalone benchmark tool.
The supported flags and env. variables will be added here soon.