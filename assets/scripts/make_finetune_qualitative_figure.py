#!/usr/bin/env python3
"""Emit the finetuning qualitative figure for the appendix.

The finetuning subsection reports a large PSNR gain from in-domain finetuning,
but a scalar cannot show *what* changes. This renders the comparison directly:
for matched episodes, the same frames from the ground-truth video, the
un-finetuned base checkpoint, and the finetuned checkpoint that the benchmark's
dense baseline uses.

Episodes are the ones for which a base-checkpoint rollout exists at the same
episode id as a finetuned rollout, so every triplet is frame-aligned and no
comparison is made across different content. PSNR is read from the per-episode
metrics of each rollout, never recomputed here, so the numbers printed on the
figure are the same ones the tables report.

    python3 assets/scripts/make_finetune_qualitative_figure.py

Writes:
    assets/paper/figures/fig_finetune_qualitative.{pdf,png}
    metrics/reports/finetune_qualitative.json
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_PDF = os.path.join(REPO, "assets/paper/figures/fig_finetune_qualitative.pdf")
OUT_PNG = os.path.join(REPO, "assets/paper/figures/fig_finetune_qualitative.png")
OUT_JSON = os.path.join(REPO, "metrics/reports/finetune_qualitative.json")
TMP = "/tmp/ft_qual_frames"

# Only Ctrl-World cells are used: they are the cells where a base and a finetuned
# checkpoint were both run on the same episode AND the rollout layout is
# identical, so the frames are directly comparable.
ROWS = [
    {
        "label": "humanoid, multi-view",
        "task": "mobile car",
        "base": "checkpoint_ablation/ctrlworld_humanoid_mv/base_ckpt/output/makovian",
        "ft": "sampling_dataset/dense/humanoid/output/multiview/ctrlworld/makovian",
        "episode": "episode_AIRBOT_MMK2_mobile_car__000019",
    },
    {
        "label": "bimanual, multi-view",
        "task": "insert gear shaft",
        "base": "checkpoint_ablation/ctrlworld_bimanual_mv/base_ckpt/output/makovian",
        "ft": "sampling_dataset/dense/bimanual/output/multiview/ctrlworld/makovian",
        "episode": "episode_find_insert_small_gear_shaft__000018",
    },
]

# Fractions through the rollout at which to sample frames. The first frame is
# excluded: it is conditioned on ground truth, so it shows no difference and
# would waste a column.
FRACS = [0.25, 0.55, 0.85]


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def n_frames(path):
    r = sh("ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames "
           "-of default=nw=1:nk=1 '%s'" % path)
    try:
        return int(r.stdout.strip())
    except ValueError:
        return 0


def extract(path, frac, dst, total):
    """Pull one frame at a fraction through the clip."""
    if total <= 0:
        return None
    idx = max(0, min(total - 1, int(round(frac * (total - 1)))))
    sh("ffmpeg -y -v error -i '%s' -vf \"select=eq(n\\,%d)\" -vsync 0 "
       "-frames:v 1 '%s'" % (path, idx, dst))
    return dst if os.path.exists(dst) and os.path.getsize(dst) > 0 else None


def psnr_of(metrics_path):
    if not os.path.exists(metrics_path):
        return None
    d = json.load(open(metrics_path))
    for k in ("psnr", "mean_psnr"):
        if isinstance(d.get(k), (int, float)):
            return float(d[k])
    s = d.get("summary") or {}
    return float(s["psnr"]) if isinstance(s.get("psnr"), (int, float)) else None


def collect():
    os.makedirs(TMP, exist_ok=True)
    out = []
    for row in ROWS:
        ep = row["episode"]
        base_dir = os.path.join(REPO, row["base"], ep)
        ft_dir = os.path.join(REPO, row["ft"], ep)
        # pred_all_views shows every camera, which is what reveals per-view
        # collapse; the primary-view video alone hides it.
        vids = {
            "gt": os.path.join(base_dir, "gt_all_views.mp4"),
            "base": os.path.join(base_dir, "pred_all_views.mp4"),
            "ft": os.path.join(ft_dir, "pred_all_views.mp4"),
        }
        missing = [k for k, v in vids.items() if not os.path.exists(v)]
        if missing:
            print("  skip %s (missing %s)" % (ep, ",".join(missing)))
            continue
        # Frames are sampled at the same fraction of the SHORTEST of the three
        # clips: rollout lengths can differ by a frame or two, and indexing each
        # clip by its own length would silently compare different timestamps.
        totals = {k: n_frames(v) for k, v in vids.items()}
        span = min(totals.values())
        frames = {}
        for kind, v in vids.items():
            frames[kind] = [
                extract(v, f, os.path.join(TMP, "%s_%s_%d.png" % (ep[:40], kind, j)), span)
                for j, f in enumerate(FRACS)]
        out.append({
            **row,
            "psnr_base": psnr_of(os.path.join(base_dir, "metrics.json")),
            "psnr_ft": psnr_of(os.path.join(ft_dir, "metrics.json")),
            "frames": frames,
            "n_frames": totals,
            "compared_span": span,
        })
    return out


KINDS = [("gt", "Ground truth", "#333333"),
         ("base", "Base checkpoint", "#c0392b"),
         ("ft", "Finetuned", "#1a6a3c")]


def plot(data):
    ncol = len(FRACS)
    nblock = len(data)

    # Each frame is a horizontal strip of all camera views, so its aspect ratio
    # (~5:1) -- not a guessed cell height -- must set the row height, or the
    # figure fills with whitespace and the headings collide.
    aspect = 5.0
    for row in data:
        for kind, _, _ in KINDS:
            for p in row["frames"][kind]:
                if p:
                    im = plt.imread(p)
                    aspect = im.shape[1] / float(im.shape[0])
                    break
            break
        break

    cell_w = 6.95 / ncol           # fill \linewidth of the one-column appendix
    cell_h = cell_w / aspect
    head_h = 0.40                  # room for a block heading (+ column titles on block 0)
    gap_h = 0.18                   # gap between the two episode blocks
    fig_w = 7.6
    fig_h = nblock * (3 * cell_h + head_h) + (nblock - 1) * gap_h + 0.20

    fig = plt.figure(figsize=(fig_w, fig_h))
    left = 0.172
    width = 0.995 - left
    y_cursor = 1.0 - 0.02 / fig_h

    for b, row in enumerate(data):
        y_cursor -= head_h / fig_h
        head_y = y_cursor
        for k, (kind, klabel, colour) in enumerate(KINDS):
            y_cursor -= cell_h / fig_h
            for c in range(ncol):
                ax = fig.add_axes([left + c * width / ncol, y_cursor,
                                   width / ncol * 0.985, cell_h / fig_h * 0.96])
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_edgecolor(colour); s.set_linewidth(1.2)
                p = row["frames"][kind][c]
                if p:
                    ax.imshow(plt.imread(p))
                else:
                    ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                            transform=ax.transAxes, fontsize=7)
                if b == 0 and k == 0:
                    ax.set_title("%d%% through rollout" % int(100 * FRACS[c]),
                                 fontsize=8.0, pad=3)
            # Row label goes in the left gutter as horizontal text, centred on
            # the row. A rotated y-label would be taller than these 5:1 strips
            # and would collide with the neighbouring row's label.
            lab = klabel
            if kind == "base" and row["psnr_base"]:
                lab += ", %.1f dB" % row["psnr_base"]
            elif kind == "ft" and row["psnr_ft"]:
                lab += ", %.1f dB" % row["psnr_ft"]
            fig.text(left - 0.008, y_cursor + cell_h / fig_h * 0.48, lab,
                     fontsize=7.0, color=colour, ha="right", va="center")
        # Headings sit in the reserved band above each block, so they cannot
        # overlap the column titles or the frames.
        # Block 0 also carries the column titles, so its heading sits higher.
        dy = 0.20 / fig_h if b == 0 else 0.03 / fig_h
        fig.text(left, head_y + dy, "%s \u2014 %s" % (row["label"], row["task"]),
                 fontsize=8.7, fontweight="bold", ha="left", va="bottom")
        fig.text(0.995, head_y + dy,
                 "+%.2f dB from finetuning" % (row["psnr_ft"] - row["psnr_base"]),
                 fontsize=8.2, ha="right", va="bottom", color="#1a6a3c")
        y_cursor -= gap_h / fig_h

    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=200)
    plt.close(fig)


def main():
    data = collect()
    if not data:
        print("no matched episodes found", file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    plot(data)
    payload = {
        "generated_by": "assets/scripts/make_finetune_qualitative_figure.py",
        "purpose": "Frame-level comparison of base vs finetuned checkpoints on "
                   "matched episodes, supporting the finetuning-effect claim.",
        "sampled_fractions": FRACS,
        "episodes": [{
            "cell": r["label"], "task": r["task"], "episode": r["episode"],
            "psnr_base": r["psnr_base"], "psnr_finetuned": r["psnr_ft"],
            "psnr_gain": round(r["psnr_ft"] - r["psnr_base"], 3),
            "n_frames": r["n_frames"], "compared_span": r["compared_span"],
        } for r in data],
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(payload, fh, indent=2)
    print("wrote", os.path.relpath(OUT_PDF, REPO))
    print("wrote", os.path.relpath(OUT_JSON, REPO))
    for e in payload["episodes"]:
        print("  %-22s %-20s base %.2f -> ft %.2f (+%.2f dB)" % (
            e["cell"], e["task"], e["psnr_base"], e["psnr_finetuned"],
            e["psnr_gain"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
