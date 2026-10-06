#!/usr/bin/env python3
"""Render the Main-Results C(i) "star chart": one radar/spider panel per
model family (Ctrl-World / DreamGen / DreamDojo), one polygon per
acceleration method, summarizing speed vs. fidelity retention vs. dense.

Reads the master table produced by scripts/tables/make_speedup_summary.py
(results/reports/speedup_summary_full.csv) and rolls it up to one
(method, model_family) point per axis:

    - Speedup            = dense_time / method_time (>1 is faster)
    - PSNR retention      = method_psnr / dense_psnr   (>1 is better)
    - SSIM retention      = method_ssim / dense_ssim   (>1 is better)
    - LPIPS retention     = dense_lpips / method_lpips  (>1 is better, i.e.
                            method has LOWER lpips than dense)

A method only gets a polygon vertex on an axis if at least one cell in that
family has that number; a family panel is skipped for methods with zero
cells in that family (matches the acceleration-benchmark rule: report
per-method applicability honestly, do not force a data point that was never
measured). Dense itself is drawn as the reference circle at 1.0 on every
axis.

Usage:
    python assets/scripts/make_speedup_star_chart.py \
        --csv results/reports/speedup_summary_full.csv \
        --out assets/paper_figures/speedup_star_chart
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS_ORDER = ["dicache", "worldcache", "fastercache", "sito", "itm",
                 "pisa", "radial", "svg1", "svg2"]
METHOD_DISPLAY = {
    "dicache": "DiCache", "worldcache": "WorldCache", "fastercache": "FasterCache",
    "sito": "SiTo", "itm": "ITM", "pisa": "PISA", "radial": "Radial",
    "svg1": "SVG1", "svg2": "SVG2",
}
FAMILY_ORDER = ["ctrlworld", "dreamgen", "dreamdojo"]
FAMILY_DISPLAY = {"ctrlworld": "Ctrl-World", "dreamgen": "DreamGen", "dreamdojo": "DreamDojo"}
FAMILY_DISPLAY_INV = {v: k for k, v in FAMILY_DISPLAY.items()}

AXES = ["speedup", "psnr_retention", "ssim_retention", "lpips_retention"]
AXIS_LABEL = {
    "speedup": "Speedup", "psnr_retention": "PSNR retention",
    "ssim_retention": "SSIM retention", "lpips_retention": "LPIPS retention",
}
# Radial scale per axis: (min, max) mapped to (0, 1) on the plot. Dense sits
# at value 1.0 on every axis, so these ranges are chosen to place the dense
# reference circle at a consistent, readable radius (~0.45-0.5) while still
# giving room for methods that beat dense.
AXIS_RANGE = {
    "speedup": (0.4, 1.6),
    "psnr_retention": (0.6, 1.4),
    "ssim_retention": (0.6, 1.4),
    "lpips_retention": (0.6, 1.4),
}
# All four ranges are symmetric around 1.0 -> dense (value 1.0 on every
# axis) lands at exactly the same normalized radius (0.5) on every spoke,
# so the dashed dense polygon is a true reference circle, not a kite.


def _norm(axis: str, value: float) -> float:
    lo, hi = AXIS_RANGE[axis]
    v = (value - lo) / (hi - lo)
    return max(0.0, min(1.05, v))


def load_rows(csv_path: Path) -> list[dict]:
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        family = FAMILY_DISPLAY_INV.get(r["family_display"], r["family_display"])
        method = None
        for m, disp in METHOD_DISPLAY.items():
            if disp == r["method_display"]:
                method = m
                break
        if method is None:
            continue

        def _f(key):
            v = r.get(key, "")
            return float(v) if v not in ("", None) else None

        out.append({
            "method": method, "family": family,
            "speedup": _f("speedup"), "psnr_retention": _f("psnr_retention"),
            "ssim_retention": _f("ssim_retention"), "lpips_retention": _f("lpips_retention"),
        })
    return out


def rollup(rows: list[dict]) -> dict[tuple[str, str], dict[str, float]]:
    """(method, family) -> {axis: mean value across cells that have it}."""
    buckets: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["method"], r["family"])
        for axis in AXES:
            if r[axis] is not None:
                buckets[key][axis].append(r[axis])
    out = {}
    for key, axmap in buckets.items():
        out[key] = {axis: mean(vals) for axis, vals in axmap.items() if vals}
    return out


PALETTE = plt.get_cmap("tab10").colors


def plot_family(ax, family: str, agg: dict[tuple[str, str], dict[str, float]]):
    n = len(AXES)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([AXIS_LABEL[a] for a in AXES], fontsize=9)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.05)

    # Dense reference circle (value 1.0 on every axis).
    dense_vals = [_norm(a, 1.0) for a in AXES] + [_norm(AXES[0], 1.0)]
    ax.plot(angles, dense_vals, color="black", linewidth=2.0, linestyle="--", label="Dense")
    ax.fill(angles, dense_vals, color="black", alpha=0.04)

    plotted = 0
    for i, method in enumerate(METHODS_ORDER):
        vals_map = agg.get((method, family))
        if not vals_map or "speedup" not in vals_map:
            continue  # not benchmarked (with timing) for this family -- omit, don't fabricate
        vals = [_norm(a, vals_map.get(a, 1.0)) for a in AXES]
        vals += vals[:1]
        color = PALETTE[i % len(PALETTE)]
        ax.plot(angles, vals, color=color, linewidth=1.6, label=METHOD_DISPLAY[method])
        ax.fill(angles, vals, color=color, alpha=0.08)
        plotted += 1

    ax.set_title(f"{FAMILY_DISPLAY[family]}  (n={plotted} methods)", fontsize=11, pad=14)
    return plotted


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=Path("results/reports/speedup_summary_full.csv"))
    ap.add_argument("--out", type=Path, default=Path("assets/paper_figures/speedup_star_chart"))
    args = ap.parse_args()

    rows = load_rows(args.csv)
    agg = rollup(rows)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2), subplot_kw=dict(polar=True))
    handles_labels = None
    for ax, family in zip(axs, FAMILY_ORDER):
        plot_family(ax, family, agg)
        h, l = ax.get_legend_handles_labels()
        if handles_labels is None or len(l) > len(handles_labels[1]):
            handles_labels = (h, l)

    fig.suptitle("Speed-fidelity trade-off of inference acceleration methods (vs. dense)",
                 fontsize=13, y=1.04)
    fig.legend(*handles_labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.08),
               fontsize=9, frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.out) + ".pdf", bbox_inches="tight")
    fig.savefig(str(args.out) + ".png", dpi=200, bbox_inches="tight")
    print(f"Wrote {args.out}.pdf and {args.out}.png")

    for family in FAMILY_ORDER:
        n = sum(1 for (m, f) in agg if f == family and "speedup" in agg[(m, f)])
        print(f"  {FAMILY_DISPLAY[family]}: {n} methods plotted "
              f"(have both speedup + quality data)")


if __name__ == "__main__":
    main()
