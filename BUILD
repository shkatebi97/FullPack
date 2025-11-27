load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow:tensorflow.bzl", "tf_opts_nortti_if_android")

package(
    licenses = ["notice"],  # Apache 2.0
)

# config_setting(
#     name = "linux_arm64",
#     values = { "cpu": "aarch64", },
#     visibility = ["//visibility:private"],
# )

config_setting(
    name = "chromiumos_arm64",
    values = {
        "crosstool_top": "//external:chromiumos/crosstool",
        "cpu": "arm",
    },
    visibility = ["//visibility:private"],
)
# This is defined by the tensorflow:linux_aarch64 config_setting.
config_setting(
    name = "cpu_aarch64",
    values = {"cpu": "aarch64"},
    visibility = ["//visibility:private"],
)

# This is defined by some config_setting's in the wild and is a reasonable value to
# support anyway.
config_setting(
    name = "cpu_arm64",
    values = {"cpu": "arm64"},
    visibility = ["//visibility:private"],
)

# This is the value defined by --config=ios_arm64.
config_setting(
    name = "cpu_ios_arm64",
    values = {"cpu": "ios_arm64"},
    visibility = ["//visibility:private"],
)

# arm64e variants of the above two. See:
# https://stackoverflow.com/questions/52624308/xcode-arm64-vs-arm64e
config_setting(
    name = "cpu_arm64e",
    values = {"cpu": "arm64e"},
    visibility = ["//visibility:private"],
)

config_setting(
    name = "cpu_ios_arm64e",
    values = {"cpu": "ios_arm64e"},
    visibility = ["//visibility:private"],
)

# This is the value defined by --config=android_arm64
config_setting(
    name = "cpu_arm64_v8a",
    values = {"cpu": "arm64-v8a"},
    visibility = ["//visibility:private"],
)

config_setting(
    name = "linux_armhf",
    values = {"cpu": "armhf"},
    visibility = ["//visibility:public"],
)

# cc_library(
#     name = "tflite_with_ruy_default",
#     build_for_embedded = True,
#     select_deps = {
#         ":chromiumos_arm64": [":tflite_with_ruy_enabled"],
#         ":cpu_aarch64": [":tflite_with_ruy_enabled"],
#         ":cpu_arm64": [":tflite_with_ruy_enabled"],
#         ":cpu_arm64e": [":tflite_with_ruy_enabled"],
#         ":cpu_ios_arm64": [":tflite_with_ruy_enabled"],
#         ":cpu_ios_arm64e": [":tflite_with_ruy_enabled"],
#         ":cpu_arm64_v8a": [":tflite_with_ruy_enabled"],
#         "//tensorflow:android_arm": ["tflite_with_ruy_enabled"],
#         "//conditions:default": [],
#     },
#     visibility = ["//visibility:private"],
# )

IS_ARM_FLAGS = select({
    ":chromiumos_arm64": ["IS_ARM", "IS_ARM64"],
    ":cpu_aarch64": ["IS_ARM", "IS_ARM64"],
    ":cpu_arm64": ["IS_ARM", "IS_ARM64"],
    ":cpu_arm64e": ["IS_ARM", "IS_ARM64"],
    ":cpu_ios_arm64": ["IS_ARM", "IS_ARM64"],
    ":cpu_ios_arm64e": ["IS_ARM", "IS_ARM64"],
    ":cpu_arm64_v8a": ["IS_ARM", "IS_ARM64"],
    # ":linux_arm64": ["IS_ARM", "IS_ARM64"],
    "//tensorflow:android_arm": ["IS_ARM", "IS_ARM32"],
    "//tensorflow:android_arm64": ["IS_ARM", "IS_ARM64"],
    "//tensorflow:linux_armhf": ["IS_ARM", "IS_ARM32"],
    "//conditions:default": ["IS_X86_64", "IS_X86"],
})

ARCH_FLAGS = select({
    ":chromiumos_arm64": ["-march=armv8.2-a+fp16"],
    ":cpu_aarch64": ["-march=armv8.2-a+fp16"],
    ":cpu_arm64": ["-march=armv8.2-a+fp16"],
    ":cpu_arm64e": ["-march=armv8.2-a+fp16"],
    ":cpu_ios_arm64": ["-march=armv8.2-a+fp16"],
    ":cpu_ios_arm64e": ["-march=armv8.2-a+fp16"],
    ":cpu_arm64_v8a": ["-march=armv8.2-a+fp16"],
    # ":linux_arm64": ["-march=armv8.2-a+fp16"],
    "//tensorflow:android_arm": [],
    "//tensorflow:android_arm64": ["-march=armv8.2-a+fp16"],
    "//tensorflow:linux_armhf": [],
    "//conditions:default": ["-march=native"],
})

ULPPACK_copts = ["-flax-vector-conversions", "-lpthread", "-lstdc++", "-Wno-pointer-arith", "-std=c++14" ]

cc_library(
    name = "half",
    hdrs = ["common/half.hpp"],
    copts = tflite_copts() + tf_opts_nortti_if_android(),
    defines = [
        "TFLITE_BUILD"
    ] + IS_ARM_FLAGS,
)

cc_library(
    name = "types",
    hdrs = ["common/types.h"],
    copts = tflite_copts() + tf_opts_nortti_if_android(),
    deps = [
        ":half"
    ],
    defines = [
        "TFLITE_BUILD"
    ] + IS_ARM_FLAGS,
)

cc_library(
    name = "flags",
    hdrs = ["common/flags.h"],
    copts = tflite_copts() + tf_opts_nortti_if_android(),
    defines = [
        "TFLITE_BUILD"
    ] + IS_ARM_FLAGS,
)


cc_library(
    name = "asmutility",
    hdrs = ["common/asmutility.h"],
    copts = tflite_copts() + tf_opts_nortti_if_android(),
    defines = [
        "TFLITE_BUILD",
    ] + IS_ARM_FLAGS,
)

cc_library(
    name = "ULPPACK",
    hdrs = [
        "kernels-impl/ULPPACK/ULPPACK.h",
        "kernels-impl/ULPPACK/test.h",
    ],
    copts = tflite_copts() + tf_opts_nortti_if_android() + ULPPACK_copts,
    defines = [
        "TFLITE_BUILD",
    ] + IS_ARM_FLAGS,
)

cc_library(
    name = "low_precision_packing",
    srcs = ["ops-implementations/mul/LowPrecisionPacking.cc"],
    hdrs = ["ops-implementations/mul/LowPrecisionPacking.h"],
    copts = tflite_copts() + tf_opts_nortti_if_android(),
    defines = [
        "TFLITE_BUILD",
        "BAZEL_BUILD"
    ] + IS_ARM_FLAGS,
    deps = [
        ":asmutility"
    ],
)

cc_library(
    name = "low_precision_fully_connected",
    srcs = [
        "low_precision_fully_connected.cc",
        "kernels-impl/Int8-Int4.cc",
        "kernels-impl/Int4-Int8.cc",
        "kernels-impl/Int4-Int4.cc",
        "kernels-impl/Int8-Int4Shift.cc",
        "kernels-impl/Int8-Ternary.cc",
        "kernels-impl/Ternary-Int8.cc",
        "kernels-impl/Ternary-Ternary.cc",
        "kernels-impl/Int8-Binary.cc",
        "kernels-impl/Binary-Int8.cc",
        "kernels-impl/Binary-Binary.cc",
        "kernels-impl/Binary-Binary-XOR.cc",
        "kernels-impl/Int8-Quaternary.cc",
        "kernels-impl/Int3-Int3.cc",
        "kernels-impl/ULPPACK.cc",
        # BarrelShiftMultiplier sources
        "kernels-impl/BarrelShiftMultiplier.cc",
        "kernels-impl/BarrelShiftMultiplier-kernels/W8A8.cc",
        "kernels-impl/BarrelShiftMultiplier-kernels/W4A4.cc",
        # SelfDependent sources
        "kernels-impl/SelfDependent.cc",
        "kernels-impl/SelfDependent-kernels/W4A4.cc",
        "kernels-impl/SelfDependent-kernels/W4A8.cc",
        "kernels-impl/SelfDependent-kernels/W8A4.cc",
        "kernels-impl/SelfDependent-kernels/W2A2.cc",
        # Float32 sources
        "kernels-impl/Float32.cc",
    ],
    hdrs = [
        "low_precision_fully_connected.h",
        "kernels-interface/kernels.hh",
        "kernels-interface/Int8-Int4.hh",
        "kernels-interface/Int4-Int8.hh",
        "kernels-interface/Int4-Int4.hh",
        "kernels-interface/Int8-Int4Shift.hh",
        "kernels-interface/Int8-Ternary.hh",
        "kernels-interface/Ternary-Int8.hh",
        "kernels-interface/Ternary-Ternary.hh",
        "kernels-interface/Int8-Binary.hh",
        "kernels-interface/Binary-Int8.hh",
        "kernels-interface/Binary-Binary.hh",
        "kernels-interface/Binary-Binary-XOR.hh",
        "kernels-interface/Int8-Quaternary.hh",
        "kernels-interface/Int3-Int3.hh",
        "kernels-interface/ULPPACK.hh",
        # BarrelShiftMultiplier sources
        "kernels-interface/BarrelShiftMultiplier.hh",
        # SelfDependent sources
        "kernels-interface/SelfDependent.hh",
        # Float32 sources
        "kernels-interface/Float32.hh",
    ],
    copts = tflite_copts() + tf_opts_nortti_if_android() + ARCH_FLAGS + ["-lm"] + ULPPACK_copts,
    defines = [
        "TFLITE_BUILD",
        "BAZEL_BUILD"
    ] + IS_ARM_FLAGS,
    deps = [
        ":asmutility",
        ":types",
        ":low_precision_packing",
        ":flags",
        ":ULPPACK",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "Profiler",
    hdrs = ["profiler/profiler.h"],
    srcs = ["profiler/profiler.cc"],
    copts = tflite_copts() + tf_opts_nortti_if_android(),
    defines = [
        "TFLITE_BUILD",
    ] + IS_ARM_FLAGS,
)

# https://docs.bazel.build/versions/master/be/c-cpp.html#cc_binary
cc_binary(
    name = "low_precision_fully_connected_test",
    srcs = ["low_precision_fully_connected_test.cc"],
    copts = tflite_copts() + tf_opts_nortti_if_android() + ["-ldl"],
    defines = [
        "TFLITE_BUILD",
        "BAZEL_BUILD"
    ] + IS_ARM_FLAGS,
    deps = [":low_precision_fully_connected"],
)

# "ops-implementations/mul/4BitInt.cc",
# "ops-implementations/mul/Quaternary.cc",
# "ops-implementations/mul/Ternary.cc",
# "ops-implementations/mul/Binary.cc",
# "-march=armv8.2-a+fp16",


