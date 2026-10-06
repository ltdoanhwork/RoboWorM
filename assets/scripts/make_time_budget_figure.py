#!/usr/bin/env python3
"""Per-backbone generation time budget, with the Amdahl ceiling of each method family.

This is the causal figure behind subsection C.2: it explains *why* the same
method wins on one backbone and loses on another, instead of leaving the reader
to take the speedup table on faith.

Left panel -- where generation time goes, per backbone. The DiT numbers are
aggregated over the per-run profiles in ``inference_times/*.json`` (fields
``average_dit_{self_attn,cross_attn,mlp}_seconds`` against
``average_denoise_time_seconds``), which is real instrumentation over tens of
thousands of runs rather than a one-off trace. The Ctrl-World U-Net column comes
from the profiled decomposition recorded in the pruning-insights rule, which is
the only profile that exists for that backbone; it is labelled as such.

Right panel -- the speedup ceiling each family can reach *even if its target
cost went to zero*, by Amdahl's law: a method that only touches self-attention
cannot exceed ``1 / (1 - self_attn_share)``. This is what makes the negative
results in the benchmark predictable rather than surprising: on the DreamDojo
singleview backbones self-attention is ~15-18% of generation, capping sparse
attention near 1.2x before any overhead, while on the DreamGen 2B DiT it is the
dominant term.

    python3 assets/scripts/make_time_budget_figure.py

Writes:
    assets/paper/figures/fig_time_budget.pdf
    results/reports/time_budget.json
"""

from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TIMES = os.path.join(REPO, "inference_times")
OUT_PDF = os.path.join(REPO, "assets/paper/figures/fig_time_budget.pdf")
OUT_JSON = os.path.join(REPO, "results/reports/time_budget.json")
CACHE = os.path.join(REPO, "results/reports/.time_budget_cache.json")

# Which resolution corresponds to which benchmark track. Both are the same 2B
# DiT; the sequence length differs, which is exactly the point of the figure.
RES_LABEL = {
    "432x768": "DreamGen 2B\nMV 16:9",
    "480x640": "DreamGen 2B\nSV 4:3",
}

# The Ctrl-World SVD U-Net has no per-run JSON instrumentation; its profile was
# measured once and recorded in .cursor/rules/ctrlworld-pruning-insights.mdc
# (episode_000255, single_arm multiview, 25 steps). Carried here verbatim, with
# provenance, and drawn in a visually distinct style so it is not mistaken for
# the aggregated DiT columns.
CTRLWORLD = {
    "label": "Ctrl-World\nSVD U-Net",
    "source": ".cursor/rules/ctrlworld-pruning-insights.mdc",
    "unet_share_of_wall": 0.64,
    "vae_decode_share": 0.08,
    # Shares *within* the U-Net.
    "within_unet": {
        "spatial_transformer": 0.16,
        "temporal_transformer": 0.22,
        "resnet": 0.31,
        "proj_time": 0.31,
    },
    # Of attention time, temporal attention (seq_len 11) is ~57%; spatial the
    # rest. Token pruning touches only spatial attention -> ~7% of wall time.
    "spatial_attn_share_of_wall": 0.07,
}


def aggregate_dit():
    """Mean per-step time budget of the DiT, per resolution.

    Every run contributes the same weight, so a resolution with more runs does
    not distort the other; the counts are reported so the reader can see the
    sample size behind each column.

    Scanning ~83k JSON files takes ~35 s, so the aggregate is cached and only
    recomputed when the number of profile files changes.
    """
    paths = glob.glob(os.path.join(TIMES, "*.json"))
    if os.path.exists(CACHE):
        try:
            with open(CACHE) as fh:
                cached = json.load(fh)
            if cached.get("n_files_scanned") == len(paths):
                return cached["by_resolution"]
        except Exception:
            pass

    acc = {}
    for path in paths:
        try:
            with open(path) as fh:
                d = json.load(fh)
        except Exception:
            continue
        if "average_dit_self_attn_seconds" not in d:
            continue
        denoise = d.get("average_denoise_time_seconds")
        if not denoise:
            continue
        res = d.get("resolution")
        if res not in RES_LABEL:
            continue
        a = acc.setdefault(res, {"n": 0, "denoise": 0.0, "self": 0.0,
                                 "cross": 0.0, "mlp": 0.0, "steps": set()})
        a["n"] += 1
        a["denoise"] += denoise
        a["self"] += d["average_dit_self_attn_seconds"]
        a["cross"] += d.get("average_dit_cross_attn_seconds", 0.0)
        a["mlp"] += d.get("average_dit_mlp_seconds", 0.0)
        a["steps"].add(d.get("num_sampling_steps"))

    out = {}
    for res, a in acc.items():
        n = a["n"]
        denoise = a["denoise"] / n
        self_t, cross_t, mlp_t = a["self"] / n, a["cross"] / n, a["mlp"] / n
        # The profiler instruments self-attention, cross-attention and the MLP;
        # whatever is left of the denoise step (norms, modulation, projections,
        # scheduler, Python overhead) is reported as `other` rather than dropped,
        # so each column sums to the measured step time.
        other = max(denoise - self_t - cross_t - mlp_t, 0.0)
        out[res] = {
            "n_runs": n,
            "sampling_steps": sorted(x for x in a["steps"] if x),
            "denoise_s": denoise,
            "self_attn_s": self_t,
            "cross_attn_s": cross_t,
            "mlp_s": mlp_t,
            "other_s": other,
            "self_attn_share": self_t / denoise,
            "cross_attn_share": cross_t / denoise,
            "mlp_share": mlp_t / denoise,
            "other_share": other / denoise,
        }
    with open(CACHE, "w") as fh:
        json.dump({"n_files_scanned": len(paths), "by_resolution": out}, fh)
    return out


def dreamdojo_profile():
    """DreamDojo self-attention share, from the DD_ATTN_PROFILE traces.

    These backbones are instrumented with an in-process attention profiler
    rather than the JSON writer, so the measured totals are carried from the
    per-cell insight rules that recorded them. Provenance is kept per entry: the
    point of the figure is the *ratio*, and these are the only measurements of it
    for this family.
    """
    return {
        "DreamDojo 2B\nhumanoid SV": {
            "self_attn_s": 8.90, "cross_attn_s": 1.71, "total_s": 58.04,
            "source": ".cursor/rules/dreamdojo-humanoid-singleview-"
                      "sparse-attention-insights.mdc",
            "shape": "S=4800, H=16, D=128",
        },
        "DreamDojo 2B\nsingle-arm SV": {
            "self_attn_s": 26.95, "cross_attn_s": 5.33, "total_s": 158.26,
            "source": ".cursor/rules/dreamdojo-single-arm-singleview-"
                      "sparse-attention-insights.mdc",
            "shape": "S=4800, H=16, D=128",
        },
        "DreamDojo 2B\nbimanual SV": {
            "self_attn_s": 71.98, "cross_attn_s": 13.82, "total_s": 403.63,
            "source": ".cursor/rules/dreamdojo-bimanual-singleview-"
                      "sparse-attention-insights.mdc",
            "shape": "S=4800, H=16, D=128",
        },
    }


def build_columns(dit, dd):
    """One column per backbone: (label, {component: share}, note)."""
    cols = []
    for res, label in RES_LABEL.items():
        if res not in dit:
            continue
        a = dit[res]
        cols.append((label, {
            "self-attention": a["self_attn_share"],
            "cross-attention": a["cross_attn_share"],
            "MLP / FFN": a["mlp_share"],
            "other (VAE, sched., Python)": a["other_share"],
        }, "$n{=}%s$ runs" % _thou(a["n_runs"]), a["self_attn_share"]))

    for label, p in dd.items():
        tot = p["total_s"]
        self_sh = p["self_attn_s"] / tot
        cross_sh = p["cross_attn_s"] / tot
        cols.append((label, {
            "self-attention": self_sh,
            "cross-attention": cross_sh,
            "MLP / FFN": float("nan"),
            "other (VAE, sched., Python)": 1.0 - self_sh - cross_sh,
        }, "attention trace", self_sh))

    cw = CTRLWORLD
    cols.append((cw["label"], {
        "self-attention": cw["spatial_attn_share_of_wall"],
        "cross-attention": float("nan"),
        "MLP / FFN": float("nan"),
        "other (VAE, sched., Python)": 1.0 - cw["spatial_attn_share_of_wall"],
    }, "single profile", cw["spatial_attn_share_of_wall"]))
    return cols


def _thou(n):
    s = str(int(n))
    out = []
    while len(s) > 3:
        out.insert(0, s[-3:])
        s = s[:-3]
    out.insert(0, s)
    return ",".join(out)


COMPONENTS = ["self-attention", "cross-attention", "MLP / FFN",
              "other (VAE, sched., Python)"]
COLORS = {
    "self-attention": "#c1121f",
    "cross-attention": "#e5989b",
    "MLP / FFN": "#457b9d",
    "other (VAE, sched., Python)": "#d9d9d9",
}


def plot(cols, path):
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(13.2, 4.5), gridspec_kw={"width_ratios": [1.75, 1.0]})

    labels = [c[0] for c in cols]
    x = np.arange(len(cols))

    # ---- left: stacked share of generation time -------------------------
    bottom = np.zeros(len(cols))
    for comp in COMPONENTS:
        vals = np.array([c[1].get(comp, 0.0) for c in cols], dtype=float)
        drawn = np.nan_to_num(vals, nan=0.0)
        ax.bar(x, drawn, bottom=bottom, width=0.62, label=comp,
               color=COLORS[comp], edgecolor="white", linewidth=0.7)
        bottom += drawn

    # Annotate the self-attention share, the quantity the whole argument turns on.
    for i, c in enumerate(cols):
        sh = c[3]
        ax.text(i, sh / 2.0, "%.0f%%" % (100 * sh), ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")
        ax.text(i, 1.015, c[2], ha="center", va="bottom", fontsize=7,
                color="#555555")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.2, linespacing=1.3)
    ax.set_ylim(0, 1.13)
    ax.set_ylabel("share of generation time")
    ax.set_title("(a) Where generation time goes", fontsize=11, loc="left")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=4,
              frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#eeeeee")

    # ---- right: Amdahl ceiling for a self-attention-only method ---------
    shares = np.array([c[3] for c in cols])
    ceilings = 1.0 / (1.0 - shares)
    order = np.argsort(ceilings)
    ax2.barh(np.arange(len(cols)), ceilings[order], height=0.6,
             color="#c1121f", alpha=0.85, edgecolor="white")
    ax2.axvline(1.0, color="#333333", lw=1.0, ls="--")
    # Place the reference label inside the axes, above the top bar, so it cannot
    # collide with the x tick labels.
    ax2.text(1.03, len(cols) - 0.35, "dense", fontsize=7.5, color="#333333",
             va="center")
    for i, v in enumerate(ceilings[order]):
        ax2.text(v + 0.03, i, r"%.2f$\times$" % v, va="center", fontsize=8.5)
    ax2.set_yticks(np.arange(len(cols)))
    ax2.set_yticklabels([labels[i].replace("\n", " ") for i in order], fontsize=7.5)
    ax2.set_xlim(0, max(ceilings) * 1.30)
    ax2.set_xlabel(r"max attainable speedup ($\times$)")
    ax2.set_title("(b) Ceiling if self-attention were free", fontsize=11, loc="left")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_axisbelow(True)
    ax2.grid(axis="x", color="#eeeeee")

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=190, bbox_inches="tight")
    plt.close(fig)


def main():
    dit = aggregate_dit()
    if not dit:
        print("ERROR: no profiled runs found under inference_times/", file=sys.stderr)
        return 1
    dd = dreamdojo_profile()
    cols = build_columns(dit, dd)
    plot(cols, OUT_PDF)

    payload = {
        "generated_by": "assets/scripts/make_time_budget_figure.py",
        "purpose": "Per-backbone generation time budget and the Amdahl ceiling it "
                   "implies for any method that only touches self-attention.",
        "dit_aggregated": dit,
        "dit_source": "inference_times/*.json "
                      "(average_dit_{self_attn,cross_attn,mlp}_seconds vs "
                      "average_denoise_time_seconds)",
        "dreamdojo_traces": dd,
        "ctrlworld_profile": CTRLWORLD,
        "amdahl_ceilings": {
            c[0].replace("\n", " "): {
                "self_attn_share": c[3],
                "max_speedup": 1.0 / (1.0 - c[3]),
            } for c in cols
        },
        "interpretation": (
            "A method that only removes self-attention work cannot exceed "
            "1/(1 - self-attention share), whatever its hyperparameters. This is "
            "why sparse attention is a strong win on the DreamGen 2B DiT but "
            "close to net-neutral on the DreamDojo single-view backbones and "
            "net-negative on the Ctrl-World U-Net, where token pruning reaches "
            "only spatial attention."),
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(payload, fh, indent=2)

    print("wrote %s" % os.path.relpath(OUT_PDF, REPO))
    print("wrote %s" % os.path.relpath(OUT_JSON, REPO))
    print("")
    print("%-42s %14s %12s" % ("backbone", "self-attn share", "Amdahl cap"))
    for c in cols:
        print("%-42s %13.1f%% %11.2fx" % (
            c[0].replace("\n", " "), 100 * c[3], 1.0 / (1.0 - c[3])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
