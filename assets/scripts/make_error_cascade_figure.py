#!/usr/bin/env python3
"""Temporal error cascade: do KIVA violations grow along the autoregressive horizon?

THE CLAIM THIS FIGURE IS ALLOWED TO MAKE
----------------------------------------
KIVA scores consecutive, non-overlapping 16-frame segments, so a rollout is a
*sequence* of two unitless ratios (rigidity, jerk), each already divided by a
threshold calibrated on real segments of the same source. That makes the horizon
axis directly readable: if chunk-based autoregression compounds structural error,
the ratio at the end of a rollout must exceed the ratio at its start on the same
clip.

We therefore report a WITHIN-CLIP statistic -- Spearman rank correlation between
segment index and ratio, computed per clip and then aggregated, plus the
last-third minus first-third difference on the same clip. A pooled curve across
clips cannot support the claim on its own, because clips have wildly different
lengths (4 to 104 segments here) and a pooled mean at late normalised time is
dominated by the longest clips.

WHAT IT IS NOT ALLOWED TO CLAIM
-------------------------------
The 300-video pool is *severity-stratified by construction*: the sampler
deliberately over-samples `severe` clips (45/100 for humanoid and single_arm,
13/100 for bimanual) so the human study would see enough failures. Absolute
violation rates from this pool are therefore NOT population estimates of any
method or cell, and the per-bundle breakdown is a description of the pool, not a
ranking of methods. Both facts are written into the emitted JSON so a table
generator cannot quietly promote them.

Data:
    sampling_dataset/data_for_web/hallucinate_bench/300-prune/{robot}/segments.json
        per clip: fps, n_frames, source, n_violated, segments[{start_frame,
        end_frame, n_frames, rigidity{value,ratio,violated}, jerk{...}}]
    sampling_dataset/data_for_web/hallucinate_bench/300-prune/manifest.json
        per clip: source_bundle (which method produced it), stratum

Writes:
    assets/paper/figures/fig_error_cascade.pdf / .png
    results/reports/error_cascade.json

Usage:
    python3 assets/scripts/make_error_cascade_figure.py
    python3 assets/scripts/make_error_cascade_figure.py --stats-only
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "sampling_dataset/data_for_web/hallucinate_bench/300-prune"
FIG = REPO / "assets/paper/figures/fig_error_cascade"
OUT_JSON = REPO / "results/reports/error_cascade.json"

EMB = ["bimanual", "humanoid", "single_arm"]
EMB_LABEL = {"bimanual": "Bimanual", "humanoid": "Humanoid",
             "single_arm": "Single-arm"}
TERMS = [("rigidity", "Rigidity ratio $r_\\tau$"), ("jerk", "Jerk ratio $j_\\tau$")]

# A clip needs enough segments for a within-clip trend to mean anything. 6 is the
# shortest length that still admits a rank correlation with a usable tie structure
# (4-segment clips exist and are kept in the pool-level counts but excluded from the
# trend statistic, which is stated in the JSON rather than silently applied).
MIN_SEG_FOR_TREND = 6

# `real_a1x_teleop` is real teleoperation video, not a generated rollout: it is the
# plausibility anchor for the ratios and must never be pooled into a model's cascade.
REAL_BUNDLE = "real_a1x_teleop"


def load_clips() -> list[dict]:
    manifest = json.loads((BENCH / "manifest.json").read_text())["videos"]
    clips = []
    for emb in EMB:
        d = json.loads((BENCH / emb / "segments.json").read_text())
        for vid_id, v in d["videos"].items():
            meta = manifest.get(f"{emb}/{vid_id}.mp4", {})
            segs = v["segments"]
            clips.append({
                "embodiment": emb,
                "video_id": vid_id,
                "source": v["source"],
                "source_bundle": meta.get("source_bundle", "unknown"),
                "stratum": meta.get("stratum", "unknown"),
                "fps": v["fps"],
                "n_frames": v["n_frames"],
                "n_segments": len(segs),
                "rigidity": [s["rigidity"]["ratio"] for s in segs],
                "jerk": [s["jerk"]["ratio"] for s in segs],
                "violated": [bool(s["rigidity"]["violated"])
                             or bool(s["jerk"]["violated"]) for s in segs],
            })
    return clips


def within_clip_trend(clips: list[dict], term: str) -> dict:
    """Per-clip Spearman(segment index, ratio) and last-third minus first-third."""
    rhos, deltas, used = [], [], 0
    for c in clips:
        y = c[term]
        if len(y) < MIN_SEG_FOR_TREND or len(set(y)) < 2:
            continue
        used += 1
        rho, _ = stats.spearmanr(np.arange(len(y)), y)
        if not np.isnan(rho):
            rhos.append(float(rho))
        k = max(1, len(y) // 3)
        deltas.append(float(np.mean(y[-k:]) - np.mean(y[:k])))
    if not rhos:
        return {"n_clips": 0}
    pos = sum(r > 0 for r in rhos)
    # Sign test on the per-clip trend: is a rising ratio more common than chance?
    binom = stats.binomtest(pos, len(rhos), 0.5)
    wil = stats.wilcoxon(deltas) if len(deltas) > 5 else None
    return {
        "n_clips": used,
        "median_spearman_index_vs_ratio": round(statistics.median(rhos), 4),
        "mean_spearman_index_vs_ratio": round(float(np.mean(rhos)), 4),
        "frac_clips_rising": round(pos / len(rhos), 4),
        "sign_test_p": float(f"{binom.pvalue:.3e}"),
        "median_last_minus_first_third": round(statistics.median(deltas), 4),
        "wilcoxon_p_last_vs_first_third": (None if wil is None
                                           else float(f"{wil.pvalue:.3e}")),
    }


def normalised_curve(clips: list[dict], term: str, nbins: int = 10) -> dict:
    """Median ratio per decile of normalised horizon, for the figure only."""
    bins = defaultdict(list)
    for c in clips:
        n = len(c[term])
        if n < 2:
            continue
        for i, v in enumerate(c[term]):
            b = min(nbins - 1, int(nbins * i / (n - 1))) if n > 1 else 0
            bins[b].append(v)
    return {
        "centres": [(b + 0.5) / nbins for b in range(nbins)],
        "median": [statistics.median(bins[b]) if bins[b] else float("nan")
                   for b in range(nbins)],
        "q25": [float(np.percentile(bins[b], 25)) if bins[b] else float("nan")
                for b in range(nbins)],
        "q75": [float(np.percentile(bins[b], 75)) if bins[b] else float("nan")
                for b in range(nbins)],
        "n": [len(bins[b]) for b in range(nbins)],
    }


def violation_curve(clips: list[dict], nbins: int = 10) -> dict:
    bins = defaultdict(list)
    for c in clips:
        n = len(c["violated"])
        if n < 2:
            continue
        for i, v in enumerate(c["violated"]):
            b = min(nbins - 1, int(nbins * i / (n - 1)))
            bins[b].append(1.0 if v else 0.0)
    return {
        "centres": [(b + 0.5) / nbins for b in range(nbins)],
        "rate": [float(np.mean(bins[b])) if bins[b] else float("nan")
                 for b in range(nbins)],
        "n": [len(bins[b]) for b in range(nbins)],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-only", action="store_true",
                    help="compute and write the JSON, skip the PDF")
    args = ap.parse_args()

    clips = load_clips()
    gen = [c for c in clips if c["source_bundle"] != REAL_BUNDLE]
    real = [c for c in clips if c["source_bundle"] == REAL_BUNDLE]

    per_emb = {e: [c for c in gen if c["embodiment"] == e] for e in EMB}

    bundle = {
        "generated_by": "assets/scripts/make_error_cascade_figure.py",
        "purpose": "Whether KIVA's per-segment physical violations grow along the "
                   "autoregressive horizon, measured within each clip.",
        "source": "sampling_dataset/data_for_web/hallucinate_bench/300-prune/"
                  "{robot}/segments.json + manifest.json",
        "unit": "consecutive non-overlapping 16-frame segments; each term is a "
                "ratio against a threshold calibrated on real segments of the "
                "same source, so 1.0 is the violation boundary",
        "scope_limits": [
            "The pool is severity-stratified by construction (severe: 45/100 "
            "humanoid, 45/100 single_arm, 13/100 bimanual), so absolute violation "
            "rates are NOT population estimates of any method or cell.",
            "The per-bundle breakdown describes the labeling pool, not a ranking "
            "of acceleration methods.",
            f"Real teleoperation clips (source_bundle={REAL_BUNDLE}) are held out "
            "of every generated-rollout statistic and reported as the anchor.",
            f"The within-clip trend needs >= {MIN_SEG_FOR_TREND} segments; shorter "
            "clips are counted in the pool but excluded from the trend.",
        ],
        "pool": {
            "n_clips": len(clips),
            "n_generated": len(gen),
            "n_real_anchor": len(real),
            "n_segments": sum(c["n_segments"] for c in clips),
            "segments_per_clip": {
                "min": min(c["n_segments"] for c in clips),
                "max": max(c["n_segments"] for c in clips),
                "median": statistics.median(c["n_segments"] for c in clips),
            },
            "by_embodiment": {
                e: {"n_clips": len(per_emb[e]),
                    "n_segments": sum(c["n_segments"] for c in per_emb[e]),
                    "median_segments_per_clip": (
                        statistics.median(c["n_segments"] for c in per_emb[e])
                        if per_emb[e] else None)}
                for e in EMB
            },
            "by_source_bundle": {
                b: sum(1 for c in clips if c["source_bundle"] == b)
                for b in sorted({c["source_bundle"] for c in clips})
            },
            "by_stratum": {
                s: sum(1 for c in clips if c["stratum"] == s)
                for s in sorted({c["stratum"] for c in clips})
            },
        },
        "within_clip_trend": {
            "all_generated": {t: within_clip_trend(gen, t) for t, _ in TERMS},
            "by_embodiment": {
                e: {t: within_clip_trend(per_emb[e], t) for t, _ in TERMS}
                for e in EMB
            },
            "real_anchor": {t: within_clip_trend(real, t) for t, _ in TERMS},
        },
        "violation_rate": {
            # The pooled generated rate is segment-weighted, and bimanual supplies
            # 6,049 of the 8,370 segments (72%) at a violation rate of ~0.05, so the
            # pool average says more about the length mix than about generation. It is
            # emitted for completeness and explicitly marked not comparable to the real
            # anchor, which is single_arm only. Read `vs_real_within_embodiment`.
            "all_generated_segment_weighted": round(
                float(np.mean([v for c in gen for v in c["violated"]])), 4),
            "all_generated_not_comparable_to_real": True,
            "real_anchor": (round(
                float(np.mean([v for c in real for v in c["violated"]])), 4)
                if real else None),
            "real_anchor_embodiments": sorted({c["embodiment"] for c in real}),
            "by_embodiment": {
                e: round(float(np.mean([v for c in per_emb[e]
                                        for v in c["violated"]])), 4)
                for e in EMB if per_emb[e]
            },
            "vs_real_within_embodiment": {
                e: {
                    "generated": round(float(np.mean(
                        [v for c in per_emb[e] for v in c["violated"]])), 4),
                    "real": round(float(np.mean(
                        [v for c in real if c["embodiment"] == e
                         for v in c["violated"]])), 4),
                    "n_real_clips": sum(1 for c in real if c["embodiment"] == e),
                    "ratio_generated_over_real": round(
                        float(np.mean([v for c in per_emb[e]
                                       for v in c["violated"]])
                              / np.mean([v for c in real if c["embodiment"] == e
                                         for v in c["violated"]])), 3),
                }
                for e in EMB
                if any(c["embodiment"] == e for c in real) and per_emb[e]
            },
        },
        "curves": {
            "normalised_horizon_deciles": {
                e: {t: normalised_curve(per_emb[e], t) for t, _ in TERMS}
                for e in EMB
            },
            "violation_rate_deciles": {
                e: violation_curve(per_emb[e]) for e in EMB
            },
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(bundle, indent=2) + "\n")
    print(f"[ok] {OUT_JSON.relative_to(REPO)}")
    for t, _ in TERMS:
        s = bundle["within_clip_trend"]["all_generated"][t]
        print(f"     {t:9s} median within-clip rho {s['median_spearman_index_vs_ratio']:+.3f}"
              f"  rising in {s['frac_clips_rising']:.0%} of {s['n_clips']} clips"
              f"  (sign test p={s['sign_test_p']:.1e})")
    print(f"     violation rate (segment-weighted, not comparable across pools): "
          f"generated {bundle['violation_rate']['all_generated_segment_weighted']:.3f}"
          f"  real anchor {bundle['violation_rate']['real_anchor']}")
    for e, v in bundle["violation_rate"]["vs_real_within_embodiment"].items():
        print(f"     {e}: generated {v['generated']:.3f} vs real {v['real']:.3f} "
              f"({v['ratio_generated_over_real']:.2f}x, {v['n_real_clips']} real clips)")

    if args.stats_only:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"bimanual": "#2471a3", "humanoid": "#c0392b",
               "single_arm": "#1e8449"}
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.1))

    for ax, (term, tlabel) in zip(axes[:2], TERMS):
        for e in EMB:
            c = bundle["curves"]["normalised_horizon_deciles"][e][term]
            x = np.array(c["centres"])
            ax.fill_between(x, c["q25"], c["q75"], color=colours[e], alpha=0.13,
                            linewidth=0)
            ax.plot(x, c["median"], "-o", ms=3.4, lw=1.5, color=colours[e],
                    label=EMB_LABEL[e])
        ax.axhline(1.0, color="black", lw=0.9, ls="--")
        ax.text(0.015, 1.0, "violation", fontsize=6.5, va="bottom",
                transform=ax.get_yaxis_transform())
        ax.set_xlabel("normalised rollout horizon")
        ax.set_ylabel(tlabel)
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.22, lw=0.5)

    ax = axes[2]
    for e in EMB:
        c = bundle["curves"]["violation_rate_deciles"][e]
        ax.plot(c["centres"], c["rate"], "-o", ms=3.4, lw=1.5, color=colours[e],
                label=EMB_LABEL[e])
    # The real anchor exists for single_arm only, so it is drawn in that colour and
    # labelled as such -- a black "real video" line across all three panels would
    # imply a cross-embodiment reference that this pool does not contain.
    vr = bundle["violation_rate"]["vs_real_within_embodiment"]
    for e, v in vr.items():
        ax.axhline(v["real"], color=colours[e], lw=0.9, ls=":",
                   label=f"real {EMB_LABEL[e].lower()}")
    ax.set_xlabel("normalised rollout horizon")
    ax.set_ylabel("segment violation rate")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.22, lw=0.5)
    ax.legend(fontsize=6.5, frameon=False, loc="best")

    for ax, tag in zip(axes, ["(a)", "(b)", "(c)"]):
        ax.set_title(tag, loc="left", fontsize=9, fontweight="bold")
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIG}.{ext}", dpi=200, bbox_inches="tight")
    print(f"[ok] {FIG.relative_to(REPO)}.pdf / .png")


if __name__ == "__main__":
    main()
