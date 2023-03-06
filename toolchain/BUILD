package(default_visibility = ["//visibility:public"])

load(":cc_toolchain_config.bzl", "cc_toolchain_config")

cc_toolchain_config(name = "aarch64_toolchain_config")
cc_toolchain_config(name = "armhf_toolchain_config")

cc_toolchain_suite(
    name = "aarch64_suite",
    toolchains = {
        "aarch64": ":aarch64_toolchain",
    },
)
cc_toolchain_suite(
    name = "armhf_suite",
    toolchains = {
        "armhf": ":armhf_toolchain",
    },
)

filegroup(name = "empty")

cc_toolchain(
    name = "aarch64_toolchain",
    toolchain_identifier = "aarch64-toolchain",
    toolchain_config = ":aarch64_toolchain_config",
    all_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 0,
)
cc_toolchain(
    name = "armhf_toolchain",
    toolchain_identifier = "armhf-toolchain",
    toolchain_config = ":armhf_toolchain_config",
    all_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 0,
)



