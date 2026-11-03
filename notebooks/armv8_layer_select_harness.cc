// Standalone ARMv8 tests for layer selection + AF32WI8 GEMM (qemu-aarch64).
#include "low_precision_fully_connected.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using LowPrecision::FullyConnected::AF32WI8ShapeOk;
using LowPrecision::FullyConnected::IsArmV8Runtime;
using LowPrecision::FullyConnected::IsLayerSelected;
using LowPrecision::FullyConnected::ParseLayerSelection;

static int g_failed = 0;

static void expect(bool cond, const char* name, const std::string& detail = "") {
  std::cout << (cond ? "[PASS] " : "[FAIL] ") << name;
  if (!detail.empty()) std::cout << " — " << detail;
  std::cout << std::endl;
  if (!cond) g_failed++;
}

static void set_env(const char* key, const char* val) {
  if (val == nullptr) unsetenv(key);
  else setenv(key, val, 1);
}

static LowPrecision::Shape make_shape(int r, int c) {
  int dims[2] = {r, c};
  return LowPrecision::get_shape(dims, 2);
}

static void test_parse_and_gates() {
  expect(ParseLayerSelection("").select_all, "parse_layers_all_empty");
  expect(ParseLayerSelection("ALL").select_all, "parse_layers_all");
  {
    auto s = ParseLayerSelection("0");
    expect(s.IsSelected(0) && !s.IsSelected(1), "parse_layers_single_index");
  }
  {
    auto s = ParseLayerSelection("0,2,5");
    expect(s.IsSelected(0) && s.IsSelected(2) && s.IsSelected(5) && !s.IsSelected(1),
           "parse_layers_multi_index");
  }
  {
    auto s = ParseLayerSelection("dense_1,fc2");
    expect(s.IsSelected(-1, "dense_1") && s.IsSelected(-1, "model/dense_1/x") &&
               !s.IsSelected(-1, "dense_2"),
           "parse_layers_by_name");
  }
  {
    auto s = ParseLayerSelection("conv1");
    expect(!s.IsSelected(-1, "conv10"), "name_exact_conv1_vs_conv10");
    expect(s.IsSelected(-1, "conv1"), "name_exact_conv1_vs_conv1");
    expect(s.IsSelected(-1, "model/conv1/Conv2D") && !s.IsSelected(-1, "model/conv10/Conv2D") &&
               !s.IsSelected(-1, "xconv1") && !s.IsSelected(-1, "conv1_bn"),
           "name_exact_component_boundaries");
  }
  {
    auto s = ParseLayerSelection("0,dense_1,3");
    expect(s.IsSelected(0) && s.IsSelected(3) && s.IsSelected(-1, "dense_1"),
           "parse_layers_mixed");
  }
  {
    auto s = ParseLayerSelection("0,!!!bad!!!,dense_1");
    expect(!s.invalid_tokens.empty() && s.IsSelected(0) && s.IsSelected(-1, "dense_1"),
           "parse_layers_invalid_token");
  }

  set_env("LowPrecisionFC_Layers", "0,2");
  expect(IsLayerSelected(2, ""), "should_apply_matched_layer");
  expect(!IsLayerSelected(1, ""), "should_apply_unmatched_layer");

  expect(AF32WI8ShapeOk(4, 8, 4) && !AF32WI8ShapeOk(5, 8, 4) &&
             !AF32WI8ShapeOk(4, 7, 4) && !AF32WI8ShapeOk(4, 8, 5),
         "af32wi8_rejects_bad_shapes");
  expect(IsArmV8Runtime(), "af32wi8_armv8_runtime_true");
}

static void test_af32wi8_gemm() {
  using namespace LowPrecision;
  const int M = 8, K = 16, N = 8;
  expect(AF32WI8ShapeOk(M, K, N), "gemm_shape_precheck");

  std::vector<float> input(M * K), output(M * N, 0.f), reference(M * N, 0.f);
  std::vector<int8_t> filter(K * N), filter_packed(K * N);
  std::vector<float> input_packed(M * K);

  for (int i = 0; i < M * K; ++i)
    input[i] = static_cast<float>(((i * 17) % 200) - 100) / 100.f;
  for (int i = 0; i < K * N; ++i)
    filter[i] = static_cast<int8_t>(((i * 13) % 254) - 127);

  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = 0.f;
      for (int k = 0; k < K; ++k)
        acc += input[m * K + k] * static_cast<float>(filter[k * N + n]);
      reference[m * N + n] = acc;
    }

  Shape in_shape = make_shape(M, K);
  Shape filt_shape = make_shape(K, N);
  Shape out_shape = make_shape(M, N);

  Status qs = FullyConnected::QuantizeFilter(
      Method::kAF32WI8, filter.data(), filt_shape, filter_packed.data(),
      MemLayout::kRowMajor);
  expect(mask_out_source(qs) == Status::Success, "gemm_quantize_filter",
         get_status_string(mask_out_source(qs)));

  Status qi = FullyConnected::QuantizeInput(
      Method::kAF32WI8, input.data(), in_shape, input_packed.data(),
      MemLayout::kRowMajor);
  expect(mask_out_source(qi) == Status::Success, "gemm_quantize_input",
         get_status_string(mask_out_source(qi)));

  Status gm = FullyConnected::MultiplyInt8MultiBatched<float, int8_t, float>(
      Method::kAF32WI8, input_packed.data(), in_shape, filter_packed.data(),
      filt_shape, output.data(), out_shape);
  expect(mask_out_source(gm) == Status::Success, "gemm_multiply",
         get_status_string(mask_out_source(gm)));

  float max_abs = 0.f;
  for (int i = 0; i < M * N; ++i)
    max_abs = std::max(max_abs, std::fabs(output[i] - reference[i]));
  // Match FullPack AF32WI8 absolute tolerance used in gemm api tests.
  expect(max_abs <= 1e-2f, "af32wi8_gemm_correctness_armv8",
         "max_abs=" + std::to_string(max_abs));
}

static void test_layer_policy_12_13() {
  set_env("LowPrecisionFC", "AF32WI8");
  set_env("LowPrecisionFC_Layers", "1");
  expect(IsLayerSelected(1, "dense_1") && !IsLayerSelected(0, "dense_0"),
         "tflite_fc_single_layer_selected");

  set_env("LowPrecisionFC_Layers", "ALL");
  expect(IsLayerSelected(0, "a") && IsLayerSelected(2, "b"),
         "tflite_fc_all_layers_selected");
}

int main() {
  std::cout << "IsArmV8Runtime=" << (IsArmV8Runtime() ? 1 : 0) << std::endl;
  test_parse_and_gates();
  test_af32wi8_gemm();
  test_layer_policy_12_13();
  std::cout << (g_failed ? "FAILED: " : "OK: ") << g_failed << " failures\n";
  return g_failed ? 1 : 0;
}
