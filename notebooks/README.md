# AF32WI8 layer selection (TFLite / FullPack)

Select which FullyConnected layers use the AF32WI8 (`float32` activations × `int8` weights) NEON kernel on **ARMv8 / aarch64**.

## Environment

| Variable | Values | Meaning |
|----------|--------|---------|
| `LowPrecisionFC` | `AF32WI8` | Enable AF32WI8 method (required) |
| `LowPrecisionFC_Layers` | `ALL` (default if unset/empty) | Apply to every applicable FC |
| | `0` | Only FC layer index `0` |
| | `0,2,5` | Specific indices (order of FC Prepare) |
| | `dense_1,fc2` | Match filter tensor name (exact or substring) |
| | `0,dense_1` | Mix of indices and names |
| `LowPrecisionFC_Debug` | `1` | Log id/name/selected/appliable per FC |

Invalid tokens in `LowPrecisionFC_Layers` are **ignored** (safe). If every token is invalid, **no** layer is selected.

## Shape constraints (AF32WI8)

- `M % 4 == 0`, `K % 8 == 0`, `N % 4 == 0`
- Non-ARMv8 hosts: kernel is not applied (`IsArmV8Runtime()` gate)

## Tests

Notebook: [`af32wi8_layer_select_tests.ipynb`](af32wi8_layer_select_tests.ipynb)

```bash
cd FullPack/notebooks
# Python unit checks (1–8, 10–11):
python3 ../python/test_layer_select_unit.py

# ARMv8 GEMM + layer policy under qemu (covers 9 / 12 / 13 on x86 hosts):
./run_armv8_tests.sh
```

On x86_64, the notebook invokes `run_armv8_tests.sh` (cross-compile + `qemu-aarch64-static`) instead of skipping tests 9/12/13.

## Build notes

- Sources live in `tensorflow/lite/kernels/FullPack/` (vendored from `FullPack/`).
- `BUILD` includes `kernels-impl/Float32.cc` for AF32WI8.
- Hook: `tensorflow/lite/kernels/fully_connected.cc` gates with `IsLayerSelected` before `IsAppliable`.

Example:

```bash
export LowPrecisionFC=AF32WI8
export LowPrecisionFC_Layers=1          # only second FC
# or
export LowPrecisionFC_Layers=dense_1
```
