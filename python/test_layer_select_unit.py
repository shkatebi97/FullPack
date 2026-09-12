"""Minimal unit checks mirroring notebook tests 1–8 for the C++ contract.

Run with: python3 python/test_layer_select_unit.py
(This validates the Python reference used by the notebook; C++ is source-of-truth for TFLite.)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer_select as ls


def main():
    assert ls.parse_layer_selection("ALL").select_all
    assert ls.parse_layer_selection("0").is_selected(0)
    assert not ls.parse_layer_selection("0").is_selected(1)
    assert ls.parse_layer_selection("0,2,5").is_selected(2)
    assert ls.parse_layer_selection("dense_1,fc2").is_selected(name="model/dense_1/x")
    assert ls.parse_layer_selection("0,dense_1").is_selected(0)
    assert ls.parse_layer_selection("0,dense_1").is_selected(name="dense_1")
    sel = ls.parse_layer_selection("0,!!!,dense_1")
    assert "!!!" in sel.invalid_tokens
    assert ls.should_apply_af32wi8(
        method_env="AF32WI8", layers_env="0", layer_index=0, require_arm=False
    )
    assert not ls.should_apply_af32wi8(
        method_env="AF32WI8", layers_env="0", layer_index=1, require_arm=False
    )
    print("section-a unit OK")


if __name__ == "__main__":
    main()
