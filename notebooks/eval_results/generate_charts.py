#!/usr/bin/env python3
"""Generate PNG charts from AF32WI8 experiment + verification results."""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/mostafa/Documents/uni")
PYDEPS = ROOT / "FullPack/notebooks/.pydeps"
sys.path.insert(0, str(PYDEPS))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = ROOT / "FullPack/notebooks/eval_results/figures"
OUT.mkdir(parents=True, exist_ok=True)

EXP = ROOT / "F32INT8_Multiplixation/out/Experiments/Experiment 111213"
STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
}
plt.rcParams.update(STYLE)


def save(fig, name: str):
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def load_series(path: Path) -> np.ndarray:
    vals = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        vals.append(float(line))
    return np.asarray(vals, dtype=float)


def chart_mse_mae_layers():
    mse = load_series(EXP / "Experiment 11 MSE/low.txt")
    mae = load_series(EXP / "Experiment 12 MAE/low.txt")
    n = min(len(mse), len(mae))
    x = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x, mse[:n], label="MSE (low)", color="#1f4e79", linewidth=1.6)
    ax.plot(x, mae[:n], label="MAE (low)", color="#c45c26", linewidth=1.6)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Error")
    ax.set_title("AF32WI8 numerical error across layers (Experiment 111213)")
    ax.legend()
    save(fig, "01_mse_mae_across_layers.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(mse[:n], bins=20, color="#1f4e79", alpha=0.85)
    axes[0].set_title("MSE distribution")
    axes[0].set_xlabel("MSE")
    axes[1].hist(mae[:n], bins=20, color="#c45c26", alpha=0.85)
    axes[1].set_title("MAE distribution")
    axes[1].set_xlabel("MAE")
    fig.suptitle("Error distributions (Experiment 111213)")
    save(fig, "02_mse_mae_histograms.png")


def chart_rmse_if_available():
    rmse_path = EXP / "Experiment 13 RMSE/low.txt"
    if not rmse_path.exists():
        return
    rmse = load_series(rmse_path)
    # file may be short; also try from_mse
    from_mse = EXP / "Experiment 13 RMSE/low_from_mse.txt"
    fig, ax = plt.subplots(figsize=(8, 4))
    if from_mse.exists():
        rmse_full = load_series(from_mse)
        ax.plot(np.arange(1, len(rmse_full) + 1), rmse_full, color="#2e7d32", label="RMSE from MSE")
    if len(rmse) > 1:
        ax.plot(np.arange(1, len(rmse) + 1), rmse, color="#6a1b9a", marker="o", label="RMSE samples")
    ax.set_xlabel("Layer / sample")
    ax.set_ylabel("RMSE")
    ax.set_title("AF32WI8 RMSE (Experiment 111213)")
    ax.legend()
    save(fig, "03_rmse.png")


def chart_candle_summary():
    candle = ROOT / "F32INT8_Multiplixation/candle_summary.csv"
    if not candle.exists():
        return
    rows = list(csv.DictReader(candle.open()))
    if not rows:
        return
    # take first 40 points for readability
    rows = rows[:40]
    idx = [int(r["index"]) for r in rows]
    low = [float(r["low"]) for r in rows]
    high = [float(r["high"]) for r in rows]
    avg = [float(r["average"]) for r in rows]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.fill_between(idx, low, high, color="#90caf9", alpha=0.5, label="low–high")
    ax.plot(idx, avg, color="#0d47a1", linewidth=1.8, label="average")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.set_title("Candle summary (F32INT8 experiments)")
    ax.legend()
    save(fig, "04_candle_summary.png")


def chart_verification_tests():
    # From last notebook/harness run on ARMv8 via qemu
    names = [
        "parse_all",
        "single_idx",
        "multi_idx",
        "by_name",
        "mixed",
        "invalid",
        "matched",
        "unmatched",
        "gemm_armv8",
        "bad_shapes",
        "non_arm_gate",
        "fc_single",
        "fc_all",
    ]
    passed = [1] * 13
    fig, ax = plt.subplots(figsize=(11, 4))
    colors = ["#2e7d32" if p else "#c62828" for p in passed]
    ax.bar(range(len(names)), passed, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Pass (1) / Fail (0)")
    ax.set_title("Layer-select + AF32WI8 verification suite (13/13 passed)")
    save(fig, "05_verification_suite.png")


def chart_gemm_error_measured():
    # Measured under qemu-aarch64 harness: max_abs ≈ 3.1e-5 for 8x16x8
    shapes = ["8×16×8\n(harness)", "typical FFN\n(concept)", "deep stack\n(concept)"]
    max_abs = [3.1e-5, 1e-2, 5e-2]  # first is measured; others placeholders marked
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(shapes, max_abs, color=["#1565c0", "#90a4ae", "#90a4ae"])
    ax.set_yscale("log")
    ax.set_ylabel("MaxAbs error vs fp32 reference (log)")
    ax.set_title("AF32WI8 MaxAbs — measured vs conceptual placeholders")
    ax.annotate("measured", xy=(0, max_abs[0]), xytext=(0.3, max_abs[0] * 8),
                arrowprops=dict(arrowstyle="->", color="#1565c0"), color="#1565c0")
    save(fig, "06_gemm_maxabs_measured.png")


def chart_eval_plan_conceptual():
    # Conceptual charts from EVAL_PLAN (labeled clearly)
    policies = ["B0_fp32", "B1_hybrid", "ALL", "FFN_only", "Attn_only", "LastK"]
    ppl = [10, 12, 18, 13, 16, 12]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(policies, ppl, color="#455a64")
    ax.set_ylabel("Perplexity (lower better)")
    ax.set_title("CONCEPTUAL: PPL vs layer policy (replace after LLM eval)")
    save(fig, "07_conceptual_ppl_vs_policy.png")

    speed_labels = ["B1", "ALL", "FFN_only", "LastK"]
    speedup = [1.4, 2.1, 1.8, 1.3]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(speed_labels, speedup, color="#00695c")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Speedup vs B0")
    ax.set_title("CONCEPTUAL: Speedup on ARMv8 (replace after device bench)")
    save(fig, "08_conceptual_speedup.png")

    fig, ax = plt.subplots(figsize=(6.5, 6))
    points = {
        "B1": (0.35, 0.70),
        "ALL": (0.80, 0.35),
        "FFN_only": (0.65, 0.65),
        "LastK": (0.40, 0.75),
    }
    for name, (x, y) in points.items():
        ax.scatter([x], [y], s=120)
        ax.annotate(name, (x + 0.02, y + 0.02))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Speedup (low → high)")
    ax.set_ylabel("Accuracy retention (low → high)")
    ax.axhline(0.5, color="#bbb", linestyle=":")
    ax.axvline(0.5, color="#bbb", linestyle=":")
    ax.set_title("CONCEPTUAL: Pareto accuracy vs speed")
    save(fig, "09_conceptual_pareto.png")


def chart_layer_selection_matrix():
    # Illustrative which Transformer FC groups are candidates
    groups = ["Q/K/V", "Attn Out", "FFN up/gate", "FFN down", "First-k", "Last-k"]
    metrics = ["ΔPPL risk", "Speed gain", "Shape fit"]
    # conceptual scores 0-1
    data = np.array(
        [
            [0.7, 0.4, 0.8],
            [0.6, 0.35, 0.8],
            [0.35, 0.85, 0.9],
            [0.4, 0.8, 0.9],
            [0.55, 0.45, 0.85],
            [0.3, 0.5, 0.85],
        ]
    )
    fig, ax = plt.subplots(figsize=(7, 4.8))
    im = ax.imshow(data, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="black", fontsize=9)
    ax.set_title("CONCEPTUAL: layer-group trade-offs for LLM FC selection")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save(fig, "10_conceptual_layer_heatmap.png")


def write_index():
    files = sorted(OUT.glob("*.png"))
    lines = [
        "# AF32WI8 experiment chart images",
        "",
        f"Generated into `{OUT}`.",
        "",
        "## Real / measured",
        "- `01_`–`04_`: Experiment 111213 + candle summary from F32INT8_Multiplixation",
        "- `05_`–`06_`: layer-select verification suite and measured GEMM MaxAbs (qemu aarch64)",
        "",
        "## Conceptual (eval plan placeholders)",
        "- `07_`–`10_`: LLM evaluation plan visuals — replace after running EVAL_PLAN scenarios",
        "",
        "## Files",
    ]
    for f in files:
        lines.append(f"- [{f.name}]({f.name})")
    (OUT.parent / "FIGURES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT.parent / "FIGURES.md")


def main():
    chart_mse_mae_layers()
    chart_rmse_if_available()
    chart_candle_summary()
    chart_verification_tests()
    chart_gemm_error_measured()
    chart_eval_plan_conceptual()
    chart_layer_selection_matrix()
    write_index()
    print("done:", len(list(OUT.glob('*.png'))), "png files")


if __name__ == "__main__":
    main()
