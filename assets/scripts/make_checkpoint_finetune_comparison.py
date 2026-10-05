#!/usr/bin/env python3
"""Checkpoint ablation (Main-Results B): base / pre-finetuning checkpoint
vs. the finetuned checkpoint already used everywhere else in this repo's
`dense` baseline, on the SAME episodes.

This is a small, curated comparison (not a full-dataset walk like
make_speedup_summary.py) because a genuine "base checkpoint" only exists,
and is architecture-compatible, for a subset of cells:

  - Ctrl-World bimanual / humanoid multiview: base = the released
    `checkpoint-10000.pt` (DROID pretrain, script's own fallback default).
  - DreamGen bimanual / humanoid multiview (2B): base =
    `checkpoints/base/nvidia/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt`.

Explicitly out of scope (see scripts/_run_checkpoint_ablation_base.sh header
and the paper appendix note for why):
  - Ctrl-World / DreamGen single_arm multiview: dense itself already runs
    the un-finetuned base checkpoint here, so there is nothing to compare.
  - DreamDojo (bimanual / single_arm singleview): blocked in this
    environment by a gated HuggingFace repo (`nvidia/Cosmos-Predict2.5-2B`)
    that `ActionConditionedSetupArguments` eagerly resolves regardless of
    `--checkpoints-dir`; confirmed this ALSO breaks the existing finetuned
    dense path right now, i.e. it is an access blocker, not something
    specific to the base checkpoint.
  - DreamDojo humanoid: only one checkpoint (2B_GR1_post-train) ever existed.
  - DreamGen humanoid/single_arm SINGLEVIEW: this pipeline never writes
    per-episode metrics.json (only .mp4 + .txt), so neither checkpoint has
    a score to compare.

Usage:
    python assets/scripts/make_checkpoint_finetune_comparison.py
"""
from __future__ import annotations

import argparse
import json
import csv
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]

# NOTE ON SCOPE (see assets/scripts/benchmark_cells.py).
# The two Ctrl-World cells below are measured on the 3-VIEW STRIP checkpoints,
# which the paper no longer reports as benchmark cells (superseded by the 4-view
# grid). They are deliberately KEPT here: this table is a base-vs-finetuned
# ablation, not a benchmark cell, and no such ablation was ever run on the grid
# checkpoints -- dropping these rows would delete the only evidence that
# finetuning the Ctrl-World backbone helps at all.
#
# `benchmark_cells.is_excluded` is therefore NOT applied to this generator. The
# appendix prose states the layout mismatch explicitly (the "Three caveats"
# sentence in 3_implemetation_details.tex). If a grid ablation is ever run,
# replace these two entries and drop that caveat.
#
# Each cell: (family, embodiment, view, label, [(base_metrics_path, finetuned_metrics_path), ...])
CELLS = [
    {
        "family": "Ctrl-World", "embodiment": "bimanual", "view": "multiview",
        "base_ckpt": "checkpoint-10000.pt (DROID pretrain)",
        "finetuned_ckpt": "bimanual_multiview_5fps/checkpoint-280000.pt",
        "pairs": [
            ("checkpoint_ablation/ctrlworld_bimanual_mv/base_ckpt/output/makovian/"
             "episode_find_insert_small_gear_shaft__000018/metrics.json",
             "sampling_dataset/dense/bimanual/output/multiview/ctrlworld/makovian/"
             "episode_find_insert_small_gear_shaft__000018/metrics.json"),
            ("checkpoint_ablation/ctrlworld_bimanual_mv/base_ckpt/output/makovian/"
             "episode_insert_lan_cable_into_the_hub__000014/metrics.json",
             "sampling_dataset/dense/bimanual/output/multiview/ctrlworld/makovian/"
             "episode_insert_lan_cable_into_the_hub__000014/metrics.json"),
        ],
    },
    {
        "family": "Ctrl-World", "embodiment": "humanoid", "view": "multiview",
        "base_ckpt": "checkpoint-10000.pt (DROID pretrain)",
        "finetuned_ckpt": "humanoid_multiview/checkpoint-200000.pt",
        "pairs": [
            ("checkpoint_ablation/ctrlworld_humanoid_mv/base_ckpt/output/makovian/"
             "episode_AIRBOT_MMK2_mobile_car__000019/metrics.json",
             "sampling_dataset/dense/humanoid/output/multiview/ctrlworld/makovian/"
             "episode_AIRBOT_MMK2_mobile_car__000019/metrics.json"),
            ("checkpoint_ablation/ctrlworld_humanoid_mv/base_ckpt/output/makovian/"
             "episode_AIRBOT_MMK2_mobile_car__000039/metrics.json",
             "sampling_dataset/dense/humanoid/output/multiview/ctrlworld/makovian/"
             "episode_AIRBOT_MMK2_mobile_car__000039/metrics.json"),
        ],
    },
    {
        "family": "DreamGen", "embodiment": "bimanual", "view": "multiview",
        "base_ckpt": "Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt",
        "finetuned_ckpt": "bimanual_multiview_seg16/iter_000120000",
        "pairs": [
            ("checkpoint_ablation/dreamgen_bimanual_mv/base_ckpt/output/makovian/base_ckpt/"
             "close_cardboard_box/episode_000026/metrics.json",
             "sampling_dataset/dense/bimanual/output/multiview/dreamgen/makovian/"
             "iter_000120000/close_cardboard_box/episode_000026/metrics.json"),
        ],
    },
    {
        "family": "DreamGen", "embodiment": "humanoid", "view": "multiview",
        "base_ckpt": "Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt",
        "finetuned_ckpt": "humanoid_multiview_seg16/iter_000100000",
        "pairs": [
            ("checkpoint_ablation/dreamgen_humanoid_mv/base_ckpt/output/makovian/base_ckpt/"
             "AIRBOT_MMK2_mobile_car/episode_000000/metrics.json",
             "sampling_dataset/dense/humanoid/output/multiview/dreamgen/makovian/"
             "iter_000100000/AIRBOT_MMK2_mobile_car/episode_000000/metrics.json"),
        ],
    },
]


def _load(p: str) -> dict:
    fp = ROOT / p
    return json.loads(fp.read_text())


def build_rows():
    rows = []
    for cell in CELLS:
        base_psnr, base_ssim, base_lpips = [], [], []
        ft_psnr, ft_ssim, ft_lpips = [], [], []
        for base_p, ft_p in cell["pairs"]:
            b = _load(base_p)
            f = _load(ft_p)
            base_psnr.append(b["psnr"]); base_ssim.append(b["ssim"]); base_lpips.append(b["lpips"])
            ft_psnr.append(f["psnr"]); ft_ssim.append(f["ssim"]); ft_lpips.append(f["lpips"])
        rows.append({
            "family": cell["family"], "embodiment": cell["embodiment"], "view": cell["view"],
            "base_ckpt": cell["base_ckpt"], "finetuned_ckpt": cell["finetuned_ckpt"],
            "n_episodes": len(cell["pairs"]),
            "base_psnr": round(mean(base_psnr), 3), "finetuned_psnr": round(mean(ft_psnr), 3),
            "base_ssim": round(mean(base_ssim), 4), "finetuned_ssim": round(mean(ft_ssim), 4),
            "base_lpips": round(mean(base_lpips), 4), "finetuned_lpips": round(mean(ft_lpips), 4),
            "delta_psnr": round(mean(ft_psnr) - mean(base_psnr), 3),
            "delta_ssim": round(mean(ft_ssim) - mean(base_ssim), 4),
            "delta_lpips": round(mean(ft_lpips) - mean(base_lpips), 4),
        })
    return rows


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["family", "embodiment", "view", "base_ckpt", "finetuned_ckpt", "n_episodes",
              "base_psnr", "finetuned_psnr", "delta_psnr",
              "base_ssim", "finetuned_ssim", "delta_ssim",
              "base_lpips", "finetuned_lpips", "delta_lpips"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})


def plot_bars(rows, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [f"{r['family']}\n{r['embodiment']}" for r in rows]
    metrics = [("psnr", "PSNR (dB) \u2191"), ("ssim", "SSIM \u2191"), ("lpips", "LPIPS \u2193")]

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    x = np.arange(len(rows))
    width = 0.35
    for ax, (key, ylabel) in zip(axs, metrics):
        base_vals = [r[f"base_{key}"] for r in rows]
        ft_vals = [r[f"finetuned_{key}"] for r in rows]
        b1 = ax.bar(x - width / 2, base_vals, width, label="Base (no finetuning)",
                     color="#9e9e9e")
        b2 = ax.bar(x + width / 2, ft_vals, width, label="Finetuned (dense)",
                     color="#1f77b4")
        for bars in (b1, b2):
            for rect in bars:
                h = rect.get_height()
                ax.annotate(f"{h:.2f}", (rect.get_x() + rect.get_width() / 2, h),
                            ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel.split(" ")[0], fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, lbls = axs[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06),
               frameon=False, fontsize=10)
    fig.suptitle("Effect of embodiment fine-tuning vs. the released base checkpoint\n"
                 "(same held-out episodes, matched dense pipeline)", fontsize=12, y=1.03)
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out_path) + ".png", dpi=200, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=ROOT / "metrics/reports/checkpoint_finetune_comparison.csv")
    ap.add_argument("--fig", type=Path, default=ROOT / "assets/paper_figures/checkpoint_finetune_comparison")
    args = ap.parse_args()

    rows = build_rows()
    write_csv(rows, args.csv)
    plot_bars(rows, args.fig)

    print(f"Wrote {len(rows)} cells -> {args.csv}")
    print(f"Wrote {args.fig}.pdf / .png")
    print()
    for r in rows:
        print(f"{r['family']:<12}{r['embodiment']:<10} PSNR {r['base_psnr']:>6.2f} -> {r['finetuned_psnr']:>6.2f} "
              f"(+{r['delta_psnr']:.2f})   SSIM {r['base_ssim']:.3f} -> {r['finetuned_ssim']:.3f}   "
              f"LPIPS {r['base_lpips']:.3f} -> {r['finetuned_lpips']:.3f}")


if __name__ == "__main__":
    main()
