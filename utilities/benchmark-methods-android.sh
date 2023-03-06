#!/bin/bash
SRC_DIR=$PWD/$(dirname -- "${BASH_SOURCE[0]}")

exec &> >(tee  $PWD/benchmark-report.log)

echo -e "3-Layers VerySmall\n"
$SRC_DIR/debug-android.sh -v 2 -m float,int8,i8-binary,f32-binary,f16-binary,i8-ternary,i8-quaternary,i8-i4,i8-4shift -p -2 -n 1000
echo -e "\n********************\n3-Layers Smaller\n"
$SRC_DIR/debug-android.sh -v 2 -m float,int8,i8-binary,f32-binary,f16-binary,i8-ternary,i8-quaternary,i8-i4,i8-4shift -p -1 -n 1000
echo -e "\n********************\n3-Layers Small\n"
$SRC_DIR/debug-android.sh -v 2 -m float,int8,i8-binary,f32-binary,f16-binary,i8-ternary,i8-quaternary,i8-i4,i8-4shift -p 0 -n 1000
echo -e "\n********************\n3-Layers Medium\n"
$SRC_DIR/debug-android.sh -v 2 -m float,int8,i8-binary,f32-binary,f16-binary,i8-ternary,i8-quaternary,i8-i4,i8-4shift -p 1 -n 1000
echo -e "\n********************\n3-Layers Large\n"
$SRC_DIR/debug-android.sh -v 2 -m float,int8,i8-binary,f32-binary,f16-binary,i8-ternary,i8-quaternary,i8-i4,i8-4shift -p 2 -n 1000