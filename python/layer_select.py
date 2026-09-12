"""Layer selection for AF32WI8 / LowPrecisionFC (mirrors C++ API).

Env:
  LowPrecisionFC=AF32WI8
  LowPrecisionFC_Layers=ALL | 0 | 0,2,5 | dense_1,fc2 | 0,dense_1
  LowPrecisionFC_Debug=1  (optional)

Invalid tokens are ignored (safe fail).
Unset / empty / ALL => all layers selected.
"""

from __future__ import annotations

import os
import platform
import re
from dataclasses import dataclass, field
from typing import List, Optional, Set


_INDEX_RE = re.compile(r"^\d+$")


@dataclass
class LayerSelection:
    """Parsed LowPrecisionFC_Layers policy."""

    select_all: bool = True
    indices: Set[int] = field(default_factory=set)
    names: Set[str] = field(default_factory=set)
    invalid_tokens: List[str] = field(default_factory=list)

    def is_selected(self, index: Optional[int] = None, name: Optional[str] = None) -> bool:
        if self.select_all:
            return True
        if index is not None and index in self.indices:
            return True
        if name is not None and name in self.names:
            return True
        # Also match if name contains any selected name as substring (TFLite mangling)
        if name is not None:
            for n in self.names:
                if n and n in name:
                    return True
        return False


def parse_layer_selection(raw: Optional[str] = None) -> LayerSelection:
    """Parse LowPrecisionFC_Layers value. None reads from env."""
    if raw is None:
        raw = os.environ.get("LowPrecisionFC_Layers")

    if raw is None or raw.strip() == "" or raw.strip().upper() == "ALL":
        return LayerSelection(select_all=True)

    selection = LayerSelection(select_all=False)
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token.upper() == "ALL":
            return LayerSelection(select_all=True)
        if _INDEX_RE.match(token):
            selection.indices.add(int(token))
        elif re.match(r"^[A-Za-z_][A-Za-z0-9_./:-]*$", token):
            selection.names.add(token)
        else:
            selection.invalid_tokens.append(token)
            # invalid tokens ignored (safe)
    # If everything was invalid and nothing selected, treat as none selected
    return selection


def is_layer_selected(
    index: Optional[int] = None,
    name: Optional[str] = None,
    raw: Optional[str] = None,
) -> bool:
    return parse_layer_selection(raw).is_selected(index=index, name=name)


def is_arm_v8() -> bool:
    machine = platform.machine().lower()
    return machine in ("aarch64", "arm64")


def af32wi8_shape_ok(m: int, k: int, n: int) -> bool:
    """AF32WI8 requires M%4==0, K%8==0, N%4==0."""
    return m % 4 == 0 and k % 8 == 0 and n % 4 == 0


def should_apply_af32wi8(
    *,
    method_env: Optional[str] = None,
    layers_env: Optional[str] = None,
    layer_index: Optional[int] = None,
    layer_name: Optional[str] = None,
    m: int = 4,
    k: int = 8,
    n: int = 4,
    require_arm: bool = True,
) -> bool:
    """High-level gate used by tests 7–11."""
    if method_env is None:
        method_env = os.environ.get("LowPrecisionFC", "")
    if method_env != "AF32WI8":
        return False
    if require_arm and not is_arm_v8():
        return False
    if not af32wi8_shape_ok(m, k, n):
        return False
    if not is_layer_selected(index=layer_index, name=layer_name, raw=layers_env):
        return False
    return True
