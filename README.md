# FullPack

This repository contains FullPack codes for only ARMv8 (`aarch64`) architecture.

To use FullPack, you have two options:

1. Use FullPack as a shared library.
1. Use the provided tool to test and benchmark the FullPack. (coming soon)

We will start with the first options.

## Use FullPack as a shared library
To use FullPack as a library, you should first clone this repository and then build the shared library:
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
