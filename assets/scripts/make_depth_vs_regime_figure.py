#!/usr/bin/env python3
"""Rollout depth vs the KIVA regime shift on the single-view configurations.

THE CLAIM THIS FIGURE IS ALLOWED TO MAKE
----------------------------------------
The two task-dynamics regimes of a configuration do not carry the same autoregressive
rollout depth, because they draw on different task pools. Ordering the six single-view
cells by the non-Markovian/Markovian depth ratio shows that among the four cells whose
non-Markovian rollout is actually deeper, KIVA degrades in the two deepest and not in the
two milder ones -- consistent with hallucination compounding over depth rather than with
an effect of the Markov property itself.

Both panels use the same dumbbell idiom (open marker = Markovian, arrow to
non-Markovian) so "depth grew" in (a) can be read against "violations grew" in (b) row by
row. Panel (b) shows the two LEVELS, not the delta, because the same 6-point improvement
means clearing violations entirely in one row (6.1 -> 0.0) and barely moving while still
failing a quarter of the segments in the next (23.6 -> 23.2).

WHAT IT IS NOT ALLOWED TO CLAIM
-------------------------------
- n = 6 cells. The depth ordering is an observation consistent with the within-clip
  cascade figure, not a hypothesis test of it.
- The claim is scoped to the four cells whose non-Markovian rollout is deeper. The two
  cells where it is not (humanoid, ratio 0.67x and 1.00x) are drawn ABOVE a divider, so
  humanoid DreamGen degrading on a SHORTER rollout is visibly outside the scope of the
  claim rather than a hidden counterexample. Depth is not a sufficient explanation on its
  own and is not presented as one.
- Visual Integrity is still computed into the JSON sidecar as a diagnostic, but it is NOT
  plotted: it is barred from model ranking (it fails the GT-is-ceiling audit), and mixing
  an appearance column into a physics figure invited the reader to treat the two as
  commensurable. The appearance-vs-physics argument is made elsewhere, on the surface
  where the evidence is not a 6-cell subgroup.

Data:
    metrics/reports/rq1_dense_baselines.json
        blocks.{embodiment}.{track}.kiva.dense.{mak,non_mak}
        blocks.{embodiment}.{track}.dense.visual_integrity.{mak,non_mak}
    Rollout depth: median chunks/clip per split, from the per-clip frame counts of
        sampling_dataset/dense/{emb}/output/singleview/{dreamgen,dreamdojo}, with the
        chunk stride identified by divisibility of the length distribution
        (DreamGen 92, DreamDojo bimanual+single_arm 12, DreamDojo humanoid 49).
        Transcribed below from assets/paper/appendix/tables/chunk_depth.tex so this
        figure and that table can never disagree silently.

Writes:
    assets/paper/figures/fig_depth_vs_regime.pdf / .png
    metrics/reports/depth_vs_regime.json

Usage:
    python3 assets/scripts/make_depth_vs_regime_figure.py
    python3 assets/scripts/make_depth_vs_regime_figure.py --stats-only
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "metrics/reports/rq1_dense_baselines.json"
FIG = REPO / "assets/paper/figures/fig_depth_vs_regime"
OUT_JSON = REPO / "metrics/reports/depth_vs_regime.json"

# Median chunks per clip, (Markovian, non-Markovian), and the chunk stride.
# Must stay identical to assets/paper/appendix/tables/chunk_depth.tex.
# Which RQ1 track key belongs to which model directory under output/singleview/.
TRACK = {"dreamgen": "Dreamgen SV", "dreamdojo": "DreamDojo SV"}

# Rollout depth is MEASURED, not transcribed: assets/scripts/measure_rollout_depth.py
# reads every generated clip and reports mean chunks/clip plus a Mann-Whitney test.
# It is read here so the figure cannot drift from the videos -- the earlier hand-copied
# table reported a spurious 0.67x on humanoid DreamGen (see that script's docstring).
DEPTH_JSON = REPO / "metrics" / "reports" / "rollout_depth.json"
SHORT = {"bimanual": "Bimanual", "humanoid": "Humanoid", "single_arm": "Single-arm"}
MODEL = {"Dreamgen SV": "DreamGen", "DreamDojo SV": "DreamDojo"}

# Quality colours. Deliberately not sign colours: KIVA is lower-is-better and
# Visual Integrity is higher-is-better, so a shared colour scale keyed on the
# arithmetic sign would hide the very inversion this figure reports.
C_WORSE = "#c0392b"
C_BETTER = "#2471a3"
C_DEPTH = "#7f8c8d"


def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(xs, ys):
    a, b = _rank(xs), _rank(ys)
    ma, mb = statistics.mean(a), statistics.mean(b)
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
    den = math.sqrt(sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b))
    return num / den if den else float("nan")


def load_depth():
    if not DEPTH_JSON.exists():
        raise SystemExit(f"missing {DEPTH_JSON}; run assets/scripts/measure_rollout_depth.py")
    doc = json.loads(DEPTH_JSON.read_text())
    return {(c["embodiment"], TRACK[c["model"]]): c for c in doc["cells"]}


def collect():
    blocks = json.loads(SRC.read_text())["blocks"]
    rows = []
    for (emb, trk), d in load_depth().items():
        cell = blocks[emb][trk]
        kiva = cell["kiva"]["dense"]
        vi = cell["dense"]["visual_integrity"]
        # Mean chunks/clip is the depth statistic: depth is a count of re-conditioning
        # events, and the median is degenerate on the two-valued DreamGen length
        # distribution (it flips 3->2 across a near-50/50 split).
        dm = d["makovian"]["chunks_mean"]
        dn = d["non_makovian"]["chunks_mean"]
        rows.append(dict(
            embodiment=emb, model=MODEL[trk], stride=d["stride"],
            depth_mak=dm, depth_non_mak=dn,
            depth_ratio=round(dn / dm, 3),
            depth_ratio_median=d["depth_ratio_median"],
            depth_p=d["mannwhitney_p"],
            depth_deeper=d["deeper_significant"],
            n_clips=[d["makovian"]["n_clips"], d["non_makovian"]["n_clips"]],
            kiva_mak=kiva["mak"], kiva_non_mak=kiva["non_mak"],
            d_kiva=kiva["non_mak"] - kiva["mak"],
            vi_mak=vi["mak"], vi_non_mak=vi["non_mak"],
            d_vi=vi["non_mak"] - vi["mak"],
        ))
    for r in rows:
        # KIVA is lower-is-better, Visual Integrity is higher-is-better.
        r["kiva_quality"] = "worse" if r["d_kiva"] > 0 else "better"
        r["vi_quality"] = "worse" if r["d_vi"] < 0 else "better"
        r["opposed"] = r["kiva_quality"] != r["vi_quality"]
    # Group by whether the depth difference survives its test, THEN order by ratio, so
    # the two statistically indistinguishable cells stay contiguous above the divider
    # instead of being interleaved with cells the claim is actually about.
    rows.sort(key=lambda r: (r["depth_deeper"], r["depth_ratio"]))
    return rows


def stats(rows):
    ratio = [r["depth_ratio"] for r in rows]
    dk = [r["d_kiva"] for r in rows]
    dv = [r["d_vi"] for r in rows]
    deeper = [r for r in rows if r["depth_deeper"]]
    n_opp = sum(1 for r in rows if r["opposed"])
    return dict(
        n_cells=len(rows),
        spearman_depth_vs_dkiva=round(spearman(ratio, dk), 3),
        spearman_depth_vs_dvi=round(spearman(ratio, dv), 3),
        spearman_dkiva_vs_dvi=round(spearman(dk, dv), 3),
        spearman_depth_vs_dkiva_deeper_only=round(
            spearman([r["depth_ratio"] for r in deeper], [r["d_kiva"] for r in deeper]), 3),
        n_deeper_significant=len(deeper),
        deeper_criterion="Mann-Whitney p < 1e-3 on chunks/clip AND mean ratio > 1",
        kiva_degrades_among_deeper=(
            f"{sum(1 for r in deeper if r['kiva_quality'] == 'worse')}/{len(deeper)}"),
        opposed_quality_direction=f"{n_opp}/{len(rows)}",
        opposed_sign_test_p=round(2 * 0.5 ** len(rows), 4) if n_opp == len(rows) else None,
        median_abs_d_vi=round(statistics.median(abs(r["d_vi"]) for r in rows), 2),
        median_abs_d_kiva=round(statistics.median(abs(r["d_kiva"]) for r in rows), 2),
    )


def plot(rows, st):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fmt = lambda v: f"{v:g}"
    # Left column carries the configuration name ONLY: rollout depth gets its own
    # dumbbell panel in the same visual idiom as KIVA, so the reader compares "depth grew"
    # against "violations grew" row by row instead of decoding a ratio in the label.
    labels = [f"{SHORT[r['embodiment']]} {r['model']}" for r in rows]
    y = list(range(len(rows)))
    # The claim is scoped to the cells whose depth difference SURVIVES ITS TEST. Drawing
    # that boundary in the figure keeps the two indistinguishable rows -- one of which
    # degrades anyway -- from reading as counterexamples to a claim never made about them.
    n_shallow = sum(1 for r in rows if not r["depth_deeper"])

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True,
                             gridspec_kw=dict(width_ratios=[1.0, 1.45], wspace=0.13))

    def dumbbell(ax, mk, nk, qkey, log=False):
        for i, r in enumerate(rows):
            a, b = r[mk], r[nk]
            col = C_WORSE if r[qkey] == "worse" else C_BETTER
            if abs(b - a) > 1e-9:
                ax.annotate("", xy=(b, i), xytext=(a, i),
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.9,
                                            shrinkA=3.5, shrinkB=0.0, mutation_scale=9))
            ax.plot([a], [i], "o", ms=5.2, mfc="white", mec="#555555", mew=1.3, zorder=3)
            ax.plot([b], [i], "o", ms=5.2, color=col, zorder=4)
        return [r[mk] for r in rows] + [r[nk] for r in rows]

    # ---- (a) rollout depth --------------------------------------------------
    ax = axes[0]
    for i, r in enumerate(rows):
        a, b = r["depth_mak"], r["depth_non_mak"]
        grew = r["depth_deeper"]
        col = "#2c3e50" if grew else C_DEPTH
        if abs(b - a) > 1e-9:
            ax.annotate("", xy=(b, i), xytext=(a, i),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.9,
                                        shrinkA=3.5, shrinkB=0.0, mutation_scale=9))
        ax.plot([a], [i], "o", ms=5.2, mfc="white", mec="#555555", mew=1.3, zorder=3)
        ax.plot([b], [i], "o", ms=5.2, color=col, zorder=4)
        # An indistinguishable cell gets "n.s." instead of a ratio: printing 0.97x there
        # would invite exactly the over-reading the significance test rules out.
        tag = f"{r['depth_ratio']:.2f}$\\times$" if grew else "n.s."
        ax.text(0.985, i, tag, transform=ax.get_yaxis_transform(), va="center",
                ha="right", fontsize=7.6, fontweight="bold" if grew else "normal",
                color="#2c3e50" if grew else "#999999")
    ax.set_xscale("log")
    ax.set_xlim(1.4, 132)
    ax.set_xticks([2, 5, 10, 20, 40])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("chunks per clip (log)", fontsize=8)
    ax.set_title("(a) rollout depth", loc="left", fontsize=8.5, fontweight="bold")

    # ---- (b) KIVA -----------------------------------------------------------
    ax = axes[1]
    vals = dumbbell(ax, "kiva_mak", "kiva_non_mak", "kiva_quality")
    pad = 12.0
    for i, r in enumerate(rows):
        a, b = r["kiva_mak"], r["kiva_non_mak"]
        col = C_WORSE if r["kiva_quality"] == "worse" else C_BETTER
        lo, hi, lo_is_m = min(a, b), max(a, b), a <= b
        ax.text(lo - pad * 0.10, i, f"{lo:.1f}", va="center", ha="right", fontsize=8.0,
                color="#555555" if lo_is_m else col,
                fontweight="normal" if lo_is_m else "bold")
        ax.text(hi + pad * 0.10, i, f"{hi:.1f}", va="center", ha="left", fontsize=8.0,
                color=col if lo_is_m else "#555555",
                fontweight="bold" if lo_is_m else "normal")
    ax.set_xlim(min(vals) - pad, max(vals) + pad)
    ax.set_xlabel("invalid 16-frame segments (%)", fontsize=8)
    ax.set_title("(b) KIVA   $\\downarrow$ better", loc="left", fontsize=8.5,
                 fontweight="bold")

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=7.8)
    axes[0].set_ylim(-0.6, len(rows) - 0.4)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.axhline(n_shallow - 0.5, color="#aaaaaa", lw=0.8, ls=(0, (4, 3)))
        ax.tick_params(axis="x", labelsize=7, pad=1.5)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", alpha=0.18, lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    # Band annotation sits just under the divider inside panel (a), where the log axis
    # leaves empty space at low chunk counts, so it cannot collide with a marker or label.
    axes[0].text(1.55, n_shallow - 0.42, "non-Markovian rollout is deeper",
                 va="top", ha="left", fontsize=6.4, style="italic", color="#999999")
    axes[0].text(1.55, n_shallow - 0.58, "depth difference not significant",
                 va="bottom", ha="left", fontsize=6.4, style="italic", color="#999999")

    h = [plt.Line2D([], [], marker="o", ls="none", ms=5.2, mfc="white",
                    mec="#555555", mew=1.3),
         plt.Line2D([], [], marker="o", ls="none", ms=5.2, color=C_WORSE),
         plt.Line2D([], [], marker="o", ls="none", ms=5.2, color=C_BETTER)]
    fig.legend(h, ["Markovian (arrow tail)", "non-Markovian: worse",
                   "non-Markovian: better"],
               loc="lower center", ncol=3, frameon=False, fontsize=7.2,
               bbox_to_anchor=(0.56, -0.05), handlelength=1.0, columnspacing=1.6)

    fig.subplots_adjust(left=0.155, right=0.995, top=0.88, bottom=0.27)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIG}.{ext}", dpi=220, bbox_inches="tight")
    print(f"[ok] {FIG.relative_to(REPO)}.pdf / .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args()

    rows = collect()
    st = stats(rows)

    print(f"{'cell':24s} {'stride':>6s} {'depth':>9s} {'ratio':>7s} "
          f"{'dKIVA':>8s} {'dVI':>8s}  quality direction")
    for r in rows:
        print(f"{SHORT[r['embodiment']] + ' ' + r['model']:24s} {r['stride']:6d} "
              f"{f'{r[chr(100)+chr(101)+chr(112)+chr(116)+chr(104)+chr(95)+chr(109)+chr(97)+chr(107)]}->{r[chr(100)+chr(101)+chr(112)+chr(116)+chr(104)+chr(95)+chr(110)+chr(111)+chr(110)+chr(95)+chr(109)+chr(97)+chr(107)]}':>9s} "
              f"{r['depth_ratio']:6.2f}x {r['d_kiva']:+8.2f} {r['d_vi']:+8.2f}  "
              f"KIVA {r['kiva_quality']:6s} / VI {r['vi_quality']:6s}"
              f"{'  <- opposed' if r['opposed'] else ''}")
    print()
    for k, v in st.items():
        print(f"  {k:38s} {v}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(dict(
        generated_by="assets/scripts/make_depth_vs_regime_figure.py",
        purpose=("Rollout depth (autoregressive chunks) against the Markovian -> "
                 "non-Markovian change in KIVA and in Visual Integrity, per "
                 "single-view configuration."),
        sources=[
            "metrics/reports/rq1_dense_baselines.json",
            "metrics/reports/rollout_depth.json (measured rollout depth)",
        ],
        polarity_note=("KIVA is lower-is-better (% invalid 16-frame segments); Visual "
                       "Integrity is higher-is-better (CLIP-IQA+). The two deltas share "
                       "the same arithmetic sign in every cell, which means they move in "
                       "OPPOSITE quality directions. Bars are coloured by quality."),
        scope_limits=[
            "n = 6 cells: the depth ordering is an observation, not a hypothesis test.",
            "Depth is reported as MEAN chunks/clip and every 'deeper' claim is gated on a "
            "two-sided Mann-Whitney test. The median is degenerate here: it reports a "
            "spurious 0.67x on humanoid DreamGen, whose clips take only two lengths "
            "(2 or 3 chunks) split 60/70 vs 69/61, i.e. means 2.54 vs 2.47 and p=0.27.",
            "The claim covers only the 4 cells whose depth difference is significant. "
            "Humanoid DreamGen degrades with NO measurable depth increase, so depth is "
            "not a sufficient explanation on its own.",
            "Visual Integrity is barred from model ranking (GT-is-ceiling audit failure); "
            "it is used here only as a within-configuration regime delta.",
            "Sign agreement is 6/6 but magnitude agreement is not: VI +0.08 on humanoid "
            "DreamGen and KIVA -0.45 on bimanual DreamGen are negligible moves.",
        ],
        stats=st,
        cells=rows,
    ), indent=1) + "\n")
    print(f"[ok] {OUT_JSON.relative_to(REPO)}")

    if not args.stats_only:
        plot(rows, st)


if __name__ == "__main__":
    main()
