package(
    licenses = ["notice"],  # Apache 2.0
)

# Usage examples.
cc_binary(
    name = "example",
    srcs = ["example.cc"],
    deps = ["//ruy"],
)

cc_binary(
    name = "parametrized_example",
    srcs = ["parametrized_example.cc"],
    deps = ["//ruy"],
)

config_setting(
    name = "linux_arm64",
    values = {
        "cpu": "aarch64",
    },
    visibility = ["//visibility:public"],
)

cc_library(
    name = "half",
    hdrs = ["common/half.hpp"],
    copts = [
        "-static", 
        "-g",
        "-std=c++11",
    ],
)

cc_library(
    name = "types",
    hdrs = ["common/types.h"],
    copts = [
        "-static", 
        "-g",
        "-std=c++11",
    ],
    deps = [
        ":half"
    ],
)

cc_library(
    name = "flags",
    hdrs = ["common/flags.h"],
    copts = [
        "-static", 
        "-g",
        "-std=c++11",
    ],
)

cc_library(
    name = "asmutility",
    hdrs = ["common/asmutility.h"],
    copts = [
        "-static", 
        "-g",
        "-std=c++11",
    ],
)

cc_library(
    name = "low_precision_packing",
    srcs = ["ops-implementations/mul/LowPrecisionPacking.cc"],
    hdrs = ["ops-implementations/mul/LowPrecisionPacking.h"],
    copts = [
        "-static", 
        "-g",
        "-std=c++11",
    ],
    defines = [
        "BAZEL_BUILD",
    ] + select({
        ":linux_arm64": ["IS_ARM"],
        "//conditions:default": [],
    }),
    deps = [
        ":asmutility"
    ],
)

cc_library(
    name = "Profiler",
    hdrs = ["profiler/profiler.h"],
    srcs = ["profiler/profiler.cc"],
    copts = [ 
        "-static", 
        "-g",
        "-std=c++11",
    ],
)

cc_library(
    name = "Tensor",
    hdrs = ["ops-implementations/mul/Tensor.h"],
    srcs = ["ops-implementations/mul/Tensor.cc"],
    copts = [ 
        "-static", 
        "-g",
        "-std=c++11",
    ],
    defines = [
        "BAZEL_BUILD",
    ] + select({
        ":linux_arm64": ["IS_ARM"],
        "//conditions:default": [],
    }),
    deps = [
        "//ruy",
        ":types",
        ":flags",
        ":Profiler",
    ],
)

cc_library(
    name = "Mul",
    srcs = [
        "ops-implementations/mul/4BitInt.cc",
        "ops-implementations/mul/Quaternary.cc",
        "ops-implementations/mul/Ternary.cc",
        "ops-implementations/mul/Binary.cc",
        "ops-implementations/mul/Mul.cc"
    ],
    hdrs = ["ops-implementations/mul/Mul.h"],
    copts = [ 
        "-static", 
        "-g",
        "-std=c++11",
        "-march=armv8.2-a+fp16",
    ],
    defines = [
        "BAZEL_BUILD",
    ] + select({
        ":linux_arm64": ["IS_ARM"],
        "//conditions:default": [],
    }),
    deps = [
        "//ruy:context",
        "//ruy",
        ":types",
        ":flags",
        ":asmutility",
        ":Profiler",
        ":Tensor",
        ":low_precision_packing",
    ],
)

cc_library(
    name = "fully-connected",
    hdrs = ["ops-implementations/fully-connected.h"],
    srcs = ["ops-implementations/fully-connected.cc"],
    copts = [ 
        "-static", 
        "-g",
        "-std=c++11",
    ],
    defines = [
        "BAZEL_BUILD",
    ] + select({
        ":linux_arm64": ["IS_ARM"],
        "//conditions:default": [],
    }),
    deps = [
        ":types",
        ":Profiler",
        ":Mul",
        ":Tensor",
        "//ruy:context",
        "//ruy",
    ],
)

cc_binary(
    name = "fully-connected-example",
    srcs = [
        "examples/fully-connected-example.cc",
    ],
    copts = [
        "-static", 
        "-g",
        "-std=c++11",
    ],
    defines = [
        "BAZEL_BUILD",
    ] + select({
        ":linux_arm64": ["IS_ARM"],
        "//conditions:default": [],
    }),
    deps = [
        ":types",
        ":flags",
        ":Profiler",
        ":fully-connected",
    ],
)

cc_binary(
    name = "multiplication-example",
    srcs = [
        "multiplication-example.cc",
    ],
    copts = [
        "-static", 
        "-g",
        "-std=c++11",
    ],
    defines = [
        "BAZEL_BUILD",
    ] + select({
        ":linux_arm64": ["IS_ARM"],
        "//conditions:default": [],
    }),
    deps = [
        "//ruy:context",
        "//ruy",
    ],
    linkstatic=True,
)

cc_binary(
    name = "test-4BitInt",
    srcs = [
        "examples/test-4BitInt.cc",
    ],
    copts = [
        "-static", 
        "-g",
        "-std=c++11",
    ],
    defines = [
        "BAZEL_BUILD",
    ] + select({
        ":linux_arm64": ["IS_ARM"],
        "//conditions:default": [],
    }),
    deps = [
        ":types",
        ":flags",
        ":Profiler",
        ":fully-connected",
    ],
    linkstatic=True,
)


