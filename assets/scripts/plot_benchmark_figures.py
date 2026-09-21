#!/usr/bin/env python3
"""Paper figures for the benchmark space: coverage matrices + composition bars.

Replaces the earlier bubble/violin set, which was unreadable and (worse) read its
rollout counts from a raw directory walk that split the two Ctrl-World tree
variants into separate columns and included stale output trees. Everything here
reads the AUDITED counts instead, so no figure can disagree with the audit:

    python3 assets/scripts/audit_sampling_distribution.py --json metrics/reports/sampling_audit.json
    python3 assets/scripts/make_benchmark_stats.py --json metrics/reports/benchmark_stats.json
    python3 assets/scripts/make_dataset_appendix_tables.py
    python3 assets/scripts/plot_benchmark_figures.py

Design follows the robotics-dataset convention (DROID / Open X-Embodiment /
BridgeData V2): dot-matrix coverage grids and horizontal long-tail bars, with NO
pie charts -- a matrix carries the (embodiment x view x model x method) structure
in far less area than any proportional chart, and it shows the GAPS, which is
what a reviewer actually wants to see.

Writes assets/paper/figures/fig_{coverage,composition,diversity}.{pdf,png}
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "assets/paper/figures")
AUDIT = os.path.join(REPO, "metrics/reports/sampling_audit.json")
STATS = os.path.join(REPO, "metrics/reports/benchmark_stats.json")
EVID = os.path.join(REPO, "metrics/reports/dataset_appendix_evidence.json")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.6,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})

# Okabe-Ito palette (colourblind-safe).
MODEL = {"dreamgen": ("DreamGen", "#0072B2"),
         "dreamdojo": ("DreamDojo", "#D55E00"),
         "ctrlworld": ("Ctrl-World", "#009E73")}
MODEL_ORDER = ["dreamgen", "dreamdojo", "ctrlworld"]
MODEL_SHORT = {"dreamgen": "DG", "dreamdojo": "DD", "ctrlworld": "CW"}
EMB_ORDER = ["humanoid", "bimanual", "single_arm"]
EMB_LABEL = {"humanoid": "Humanoid", "bimanual": "Bimanual",
             "single_arm": "Single-arm"}
EMB_COLOR = {"humanoid": "#009E73", "bimanual": "#D55E00",
             "single_arm": "#0072B2"}
VIEW_ORDER = ["MV", "SV"]
VIEW_LABEL = {"MV": "multi-view", "SV": "single-view"}

# Method families, in the order the paper introduces them.
FAMILIES = [
    ("Dense", ["dense"], "#444444"),
    ("Token red.", ["sito", "itm"], "#CC79A7"),
    ("Caching", ["worldcache", "dicache", "fastercache"], "#0072B2"),
    ("Sparse attn.", ["pisa", "radial", "svg1", "svg2"], "#009E73"),
]
METHOD_LABEL = {
    "dense": "Dense", "sito": "SiTo", "itm": "ITM", "worldcache": "WorldCache",
    "dicache": "DiCache", "fastercache": "FasterCache", "pisa": "PISA",
    "radial": "Radial", "svg1": "SVG-1", "svg2": "SVG-2",
}


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"),
                    dpi=400 if ext == "png" else None)
    plt.close(fig)
    print(f"wrote figures/{name}.pdf + .png")


def load():
    audit = json.load(open(AUDIT))
    stats = json.load(open(STATS))
    evid = json.load(open(EVID))
    # Refuse to draw a figure the audit does not back: a pretty plot of wrong
    # numbers is worse than no plot.
    if not audit.get("scope_matches_design"):
        raise SystemExit("audit does not match the design scope -- fix the data, "
                         "not the plot")
    dense = {(c["embodiment"], c["view"], c["model"]): c
             for c in audit["dense_cells"]}
    meth = {(c["method"], c["embodiment"], c["view"], c["model"]): c
            for c in audit["method_cells"]}
    return audit, stats, evid, dense, meth


def fig_coverage(audit, dense, meth):
    """(a) dense grid over embodiment x view x model, (b) method x configuration
    dot matrix that shows exactly which (method, model) pairs are unsupported."""
    rows = [(e, v) for e in EMB_ORDER for v in VIEW_ORDER]
    fig = plt.figure(figsize=(7.1, 5.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.30], hspace=0.30)

    # ---------------- (a) dense grid ----------------
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    for j, m in enumerate(MODEL_ORDER):
        for i, (e, v) in enumerate(rows):
            c = dense.get((e, v, m))
            if c is None or c["total"] == 0:
                ax.add_patch(Rectangle((j - 0.46, i - 0.38), 0.92, 0.76,
                                       facecolor="#F4F4F4", edgecolor="#DCDCDC",
                                       lw=0.5, hatch="///", zorder=1))
                ax.text(j, i, "not applicable", ha="center", va="center",
                        fontsize=6.4, color="#9A9A9A", zorder=3)
                continue
            col = MODEL[m][1]
            ax.add_patch(Rectangle((j - 0.46, i - 0.38), 0.92, 0.76,
                                   facecolor=col, alpha=0.12, edgecolor=col,
                                   lw=0.8, zorder=1))
            ax.text(j, i - 0.10, f"{c['total']}", ha="center", va="center",
                    fontsize=11.5, color=col, fontweight="bold", zorder=3)
            ax.text(j, i + 0.21, f"{c['makovian']} + "
                                 f"{c['non_makovian']}",
                    ha="center", va="center", fontsize=6.5, color="#666666",
                    zorder=3)
    ax.set_xticks(range(3))
    ax.set_xticklabels([MODEL[m][0] for m in MODEL_ORDER], fontsize=9)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{EMB_LABEL[e]}  {VIEW_LABEL[v]}" for e, v in rows],
                       fontsize=8.2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("(a)  Dense rollouts: embodiment $\\times$ camera track "
                 "$\\times$ world model\n"
                 "cell = total, below = Markovian + non-Markovian",
                 fontsize=8.8, pad=22, loc="left")
    for i, (e, v) in enumerate(rows):
        tot = sum(dense[(e, v, m)]["total"] for m in MODEL_ORDER
                  if (e, v, m) in dense)
        ax.text(2.60, i, f"{tot}", ha="left", va="center", fontsize=8.4,
                color="#333333")
    ax.text(2.60, -0.70, "row total", ha="left", va="center", fontsize=6.6,
            color="#888888")

    # ---------------- (b) method x config dot matrix ----------------
    ax2 = fig.add_subplot(gs[1])
    cfgs = [(e, v, m) for e in EMB_ORDER for v in VIEW_ORDER
            for m in MODEL_ORDER if (e, v, m) in dense]
    methods, mcolors, spans = [], {}, []
    for fname, members, col in FAMILIES:
        start = len(methods)
        for m in members:
            methods.append(m)
            mcolors[m] = col
        spans.append((fname, start, len(methods) - 1, col))

    ax2.set_xlim(-0.5, len(cfgs) - 0.5)
    ax2.set_ylim(len(methods) - 0.5, -0.5)
    for i in range(len(methods)):
        ax2.axhline(i, color="#F2F2F2", lw=0.5, zorder=0)
    for i, m in enumerate(methods):
        for j, (e, v, mo) in enumerate(cfgs):
            c = meth.get((m, e, v, mo))
            n = c["total"] if c else 0
            scope = c["scope_total"] if c else 0
            if n == 0:
                ax2.plot(j, i, marker="x", ms=4.4, mew=1.0, color="#CBCBCB",
                         zorder=2)
            else:
                full = scope > 0 and n >= scope
                ax2.plot(j, i, marker="o", ms=7.4, zorder=2,
                         mfc=mcolors[m] if full else "white",
                         mec=mcolors[m], mew=1.0)
    ax2.set_xticks(range(len(cfgs)))
    ax2.set_xticklabels([MODEL_SHORT[mo] for _, _, mo in cfgs], fontsize=6.8)
    # One centred (embodiment, view) caption under each block of model columns,
    # so the 12 columns never print overlapping text.
    blocks = {}
    for j, (e, v, mo) in enumerate(cfgs):
        blocks.setdefault((e, v), []).append(j)
    for (e, v), js in blocks.items():
        xmid = (sum(js) / len(js) + 0.5) / len(cfgs)
        ax2.text(xmid, -0.105, f"{EMB_LABEL[e]}\n{VIEW_LABEL[v]}",
                 ha="center", va="top", fontsize=6.6, color="#444444",
                 transform=ax2.transAxes)
        if js[0] > 0:
            ax2.axvline(js[0] - 0.5, color="#D0D0D0", lw=0.7, zorder=1)
    ax2.set_yticks(range(len(methods)))
    ax2.set_yticklabels([METHOD_LABEL[m] for m in methods], fontsize=8)
    for i, lab in enumerate(ax2.get_yticklabels()):
        lab.set_color(mcolors[methods[i]])
    for s in ax2.spines.values():
        s.set_visible(False)
    ax2.tick_params(length=0)
    for fname, a, b, col in spans:
        ycen = 1 - ((a + b) / 2 + 0.5) / len(methods)
        if a == b:                      # single-row family: no room to rotate
            ax2.text(-0.150, ycen, fname, va="center", ha="center",
                     fontsize=6.6, color=col, transform=ax2.transAxes)
        else:
            ax2.text(-0.150, ycen, fname, rotation=90, va="center",
                     ha="center", fontsize=6.6, color=col,
                     transform=ax2.transAxes)
    ax2.set_title("(b)  Acceleration coverage: every method against every "
                  "configuration\n"
                  "DG = DreamGen,  DD = DreamDojo,  CW = Ctrl-World",
                  fontsize=8.8, pad=8, loc="left")
    ax2.legend(handles=[
        plt.Line2D([], [], marker="o", ls="", mfc="#666666", mec="#666666",
                   ms=6.4, label="complete"),
        plt.Line2D([], [], marker="o", ls="", mfc="white", mec="#666666",
                   ms=6.4, label="partial"),
        plt.Line2D([], [], marker="x", ls="", color="#CBCBCB", ms=5,
                   label="not supported by the model"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=3, frameon=False,
        fontsize=7.3, handletextpad=0.3, columnspacing=1.6)
    save(fig, "fig_coverage")


def fig_composition(audit, evid):
    """(a) source episodes split by task dynamics, (b) rollout budget by family."""
    cells = {(c["embodiment"], c["view"], c["split"]): c for c in evid["cells"]}
    V = {"MV": "multiview", "SV": "singleview"}
    rows = [(e, v) for e in EMB_ORDER for v in VIEW_ORDER]

    fig = plt.figure(figsize=(7.1, 2.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.30, 1.0], wspace=0.46)

    ax = fig.add_subplot(gs[0])
    mk = [cells[(e, V[v], "makovian")]["episodes_selected"] for e, v in rows]
    nm = [cells[(e, V[v], "non_makovian")]["episodes_selected"] for e, v in rows]
    y = list(range(len(rows)))
    ax.barh(y, mk, color="#0072B2", alpha=0.85, height=0.62, label="Markovian")
    ax.barh(y, nm, left=mk, color="#E69F00", alpha=0.92, height=0.62,
            label="non-Markovian")
    for i, (a, b) in enumerate(zip(mk, nm)):
        ax.text(a + b + 6, i, f"{a + b}", va="center", fontsize=7.6,
                color="#333333")
        ax.text(a / 2, i, f"{a}", va="center", ha="center", fontsize=6.7,
                color="white")
        ax.text(a + b / 2, i, f"{b}", va="center", ha="center", fontsize=6.7,
                color="white")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{EMB_LABEL[e]}  {VIEW_LABEL[v]}" for e, v in rows],
                       fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(a + b for a, b in zip(mk, nm)) * 1.18)
    ax.set_xlabel("source episodes", fontsize=8)
    ax.tick_params(labelsize=7.4, length=2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
              frameon=False, fontsize=7.2, handlelength=1.1,
              handletextpad=0.4, columnspacing=1.4)
    ax.set_title(f"(a)  Source corpus: {evid['totals']['episode_instances']:,} "
                 f"episode instances "
                 f"({evid['totals']['distinct_source_episodes']:,} distinct "
                 f"trajectories)", fontsize=8.5, loc="left", pad=20)

    ax2 = fig.add_subplot(gs[1])
    groups = {g["name"]: g["total"] for g in audit["groups"]}
    order = ["Dense (no acceleration)", "Token reduction",
             "Acceleration caching", "Sparse attention"]
    short = {"Dense (no acceleration)": "Dense", "Token reduction": "Token red.",
             "Acceleration caching": "Caching",
             "Sparse attention": "Sparse attn."}
    cols = {"Dense (no acceleration)": "#444444", "Token reduction": "#CC79A7",
            "Acceleration caching": "#0072B2", "Sparse attention": "#009E73"}
    nmeth = {"Dense (no acceleration)": 1, "Token reduction": 2,
             "Acceleration caching": 3, "Sparse attention": 4}
    vals = [groups[k] for k in order]
    ax2.barh(range(len(order)), vals, height=0.6,
             color=[cols[k] for k in order], alpha=0.88)
    for i, k in enumerate(order):
        ax2.text(groups[k] + 200, i, f"{groups[k]:,}", va="center", fontsize=7.6,
                 color="#333333")

    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels(
        [f"{short[k]}\n{nmeth[k]} method{'s' if nmeth[k] > 1 else ''}"
         for k in order], fontsize=7.6)
    ax2.invert_yaxis()
    ax2.set_xlim(0, max(vals) * 1.26)
    ax2.set_xlabel("generated rollouts", fontsize=8)
    ax2.tick_params(labelsize=7.4, length=2)
    for s in ("top", "right", "left"):
        ax2.spines[s].set_visible(False)
    ax2.set_title(f"(b)  Rollout budget: {audit['grand_total']:,} videos",
                  fontsize=8.8, loc="left", pad=20)
    save(fig, "fig_composition")


def fig_diversity(stats, evid):
    """(a) long-tail primitive distribution, (b) horizon spread per embodiment."""
    fig = plt.figure(figsize=(7.1, 2.85))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.36)

    ax = fig.add_subplot(gs[0])
    sk = sorted(stats["skills_primary"].items(), key=lambda kv: -kv[1])
    labs = [k for k, _ in sk][::-1]
    vals = [v for _, v in sk][::-1]
    ax.barh(range(len(vals)), vals, color="#0072B2", alpha=0.85, height=0.68)
    for i, v in enumerate(vals):
        ax.text(v * 1.14, i, f"{v}", va="center", fontsize=6.6, color="#333333")
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels(labs, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlim(0.7, max(vals) * 2.6)
    ax.set_xlabel("distinct instructions (log scale)", fontsize=8)
    ax.tick_params(labelsize=7, length=2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"(a)  {len(labs)} manipulation primitives across "
                 f"{stats['distinct_prompts']} instructions",
                 fontsize=8.8, loc="left", pad=6)

    ax2 = fig.add_subplot(gs[1])
    per_emb = {e: [] for e in EMB_ORDER}
    for c in evid["cells"]:
        fps = (c["source_fps"] or [15.0])[0] or 15.0
        per_emb[c["embodiment"]].extend(n / fps for n in c["lengths"])
    bp = ax2.boxplot([per_emb[e] for e in EMB_ORDER], vert=False, widths=0.5,
                     patch_artist=True, showfliers=False,
                     medianprops=dict(color="#222222", lw=1.1),
                     whiskerprops=dict(lw=0.7, color="#888888"),
                     capprops=dict(lw=0.7, color="#888888"),
                     boxprops=dict(lw=0.6, edgecolor="#666666"))
    for patch, e in zip(bp["boxes"], EMB_ORDER):
        patch.set_facecolor(EMB_COLOR[e])
        patch.set_alpha(0.45)
    for i, e in enumerate(EMB_ORDER):
        ax2.text(max(per_emb[e]) * 1.10, i + 1, f"n={len(per_emb[e])}",
                 va="center", fontsize=6.6, color="#555555")
    ax2.set_yticks(range(1, len(EMB_ORDER) + 1))
    ax2.set_yticklabels([EMB_LABEL[e] for e in EMB_ORDER], fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xscale("log")
    ax2.set_xlabel("episode duration (s, log scale)", fontsize=8)
    ax2.tick_params(labelsize=7.4, length=2)
    for s in ("top", "right", "left"):
        ax2.spines[s].set_visible(False)
    allv = [x for v in per_emb.values() for x in v]
    ax2.set_xlim(min(allv) * 0.7, max(allv) * 2.0)
    ax2.set_title(f"(b)  Horizon: {min(allv):.1f}\u2013{max(allv):.0f}s, "
                  f"{sum(allv) / 3600:.1f} h of video",
                  fontsize=8.8, loc="left", pad=6)
    save(fig, "fig_diversity")


def main():
    audit, stats, evid, dense, meth = load()
    fig_coverage(audit, dense, meth)
    fig_composition(audit, evid)
    fig_diversity(stats, evid)
    print(f"\ndense {audit['dense_total']:,} | total {audit['grand_total']:,} "
          f"rollouts | source {evid['totals']['episode_instances']:,} episode "
          f"instances ({evid['totals']['distinct_source_episodes']:,} trajectories)")


if __name__ == "__main__":
    main()
