#!/usr/bin/env python3
"""Why the two single-arm configurations are the ones whose KIVA degrades.

THE CLAIM THIS FIGURE IS ALLOWED TO MAKE
----------------------------------------
On the two single-arm configurations the Markovian and non-Markovian clip-length
distributions are DISJOINT: every non-Markovian clip is longer, hence deeper in
autoregressive chunks, than every Markovian clip (common-language effect 0.000,
Mann-Whitney p < 1e-41 on 119-120 clips per regime).

    DreamGen    M 185-277 frames = 2-3 chunks    N 369-645 frames = 4-7 chunks
    DreamDojo   M 133-181 frames = 11-15 chunks  N 193-577 frames = 16-48 chunks

That separation, not the Markov property, is what these two cells actually vary, and they
are exactly the two cells where KIVA degrades (59.4 -> 70.7 and 21.3 -> 27.0 percent of
invalid 16-frame segments). A mean or median ratio cannot convey this; only the
distributions can, which is why the figure plots them.

WHAT IT IS NOT ALLOWED TO CLAIM
-------------------------------
- These are 2 of 6 single-view cells, selected BECAUSE they are the deepest. The figure
  explains the mechanism behind an already-reported result; it is not independent
  evidence for it, and it is not a test of depth against KIVA across configurations.
- Disjoint supports mean regime and depth are perfectly confounded HERE, so this figure
  cannot separate "deeper rollout" from "non-Markovian task" on its own. What licenses
  the depth reading is the within-clip cascade (fig:error_cascade), where depth varies
  with the regime held fixed.
- Clip length is a property of the source episodes, not a model choice.

Data:
    results/reports/rollout_depth.json   (per-clip frame/chunk counts; measured by
                                          assets/scripts/measure_rollout_depth.py)
    results/reports/rq1_dense_baselines.json  (blocks.single_arm.*.kiva.dense)

Usage:
    python3 assets/scripts/make_singlearm_depth_figure.py
"""

import json
import pathlib
import statistics as st
from collections import Counter

REPO = pathlib.Path(__file__).resolve().parents[2]
DEPTH = REPO / "results" / "reports" / "rollout_depth.json"
KIVA = REPO / "results" / "reports" / "rq1_dense_baselines.json"
FIG = REPO / "assets" / "paper" / "figures" / "fig_singlearm_depth"
OUT_JSON = REPO / "results" / "reports" / "singlearm_depth.json"
# Long-format CSV of exactly what the figure draws, one row per bar, so the plot can be
# rebuilt (or redrawn in another tool) without re-reading the videos.
OUT_CSV = REPO / "results" / "reports" / "singlearm_depth_bars.csv"

TRACK = {"dreamgen": "Dreamgen SV", "dreamdojo": "DreamDojo SV"}
MODEL = {"dreamgen": "DreamGen", "dreamdojo": "DreamDojo"}

# KIVA's colour in the paper is purple; the Markovian level is the same hue lightened so
# the pair reads as one metric at two depths rather than as two different quantities.
C_KIVA = "#753ca3"
C_KIVA_LIGHT = "#c9aede"
C_MAK = "#9aa5ad"
C_GAP = "#f2e9f7"   # retained for callers; no longer used as a band (see plot())


def collect():
    cells = {(c["embodiment"], c["model"]): c for c in json.loads(DEPTH.read_text())["cells"]}
    blocks = json.loads(KIVA.read_text())["blocks"]
    rows = []
    for model in ("dreamgen", "dreamdojo"):
        c = cells[("single_arm", model)]
        k = blocks["single_arm"][TRACK[model]]["kiva"]["dense"]
        m, n = c["makovian"], c["non_makovian"]
        rows.append(dict(
            model=MODEL[model], stride=c["stride"],
            frames_mak=m["frames_all"], frames_non_mak=n["frames_all"],
            chunks_mak=m["chunks_all"], chunks_non_mak=n["chunks_all"],
            depth_ratio=c["depth_ratio_mean"], p=c["mannwhitney_p"], cl=c["cl_effect"],
            kiva_mak=k["mak"], kiva_non_mak=k["non_mak"],
            # Disjoint supports are the whole point, so they are asserted from the data
            # rather than eyeballed off the plot.
            disjoint=not (set(m["chunks_all"]) & set(n["chunks_all"])),
            gap=[max(m["frames_all"]), min(n["frames_all"])],
        ))
    return rows


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import Counter

    # HALF-PAGE FIGURE. Authored at the final physical size (2.75 in = 0.5 * 5.5 in
    # textwidth) so \includegraphics[width=0.5\linewidth] renders it 1:1 and the fonts
    # below are the fonts the reader sees. Authoring wide and scaling down would put ~3 pt
    # text on the page.
    fig, axes = plt.subplots(2, 1, figsize=(2.75, 2.85),
                             gridspec_kw=dict(hspace=1.05))

    for r_i, r in enumerate(rows):
        ax = axes[r_i]
        stride = r["stride"]
        # x is CHUNK COUNT, the quantity the cascade argument is about; clip length is the
        # same axis rescaled by the stride, so a twin axis carries both without extra room.
        cm, cn = Counter(r["chunks_mak"]), Counter(r["chunks_non_mak"])
        am, an = np.array(r["chunks_mak"]), np.array(r["chunks_non_mak"])
        top = max(max(cm.values()), max(cn.values()))
        ymax = top * 1.42          # headroom for the ratio lane and the KIVA badge
        lo, hi = min(am.min(), an.min()), max(am.max(), an.max())
        span = hi - lo
        width = max(span * 0.022, 0.42)

        g0, g1 = am.max(), an.min()
        # Bars carry RAW CLIP COUNTS on a labelled axis. Normalised heights hid the fact
        # that e.g. DreamGen has 60 clips at 4 chunks but only 6 at 6 chunks.
        for cnt, col in ((cm, C_MAK), (cn, C_KIVA)):
            xs = sorted(cnt)
            ax.bar(xs, [cnt[x] for x in xs], width=width,
                   color=col, alpha=0.9, lw=0, zorder=2)

        # The supports are disjoint but ADJACENT: max(M) and min(N) are consecutive chunk
        # counts in both cells, so the empty interval is exactly 1 chunk. Shading it as a
        # band was misleading -- that same one-chunk fact covered 20% of the DreamGen axis
        # (stride 92) and 3% of the DreamDojo one (stride 12). A single divider states the
        # separation without implying a void whose width means anything.
        ax.axvline(0.5 * (g0 + g1), 0, 0.72 / 1.42, color=C_KIVA, lw=0.8,
                   ls=(0, (2.5, 2)), alpha=0.8, zorder=3)

        # Depth ratio arrow between the two means, in its own lane above the bars.
        lane = top * 1.12
        ax.annotate("", xy=(an.mean(), lane), xytext=(am.mean(), lane),
                    arrowprops=dict(arrowstyle="-|>", color=C_KIVA, lw=1.2,
                                    shrinkA=0.8, shrinkB=0.8, mutation_scale=7),
                    zorder=5)
        for x, c in ((am.mean(), C_MAK), (an.mean(), C_KIVA)):
            ax.plot([x], [lane], "o", ms=2.8, color=c, zorder=6)
        # Label the arrow with the two means it connects, not with the word "deeper":
        # the arrow already shows the direction, and "2.18x deeper on average" reads as if
        # "on average" qualified "deeper" rather than naming the statistic being compared.
        ax.text(0.5 * (am.mean() + an.mean()), lane + top * 0.05,
                f"{am.mean():.1f} $\\to$ {an.mean():.1f} chunks "
                f"({r['depth_ratio']:.1f}$\\times$)",
                ha="center", va="bottom", fontsize=6.4, color=C_KIVA,
                fontweight="bold", zorder=6)

        # KIVA as a badge inside the axes: at this width a separate panel would leave no
        # room for the distribution, which is the thing only a figure can show. Rendered
        # via mathtext bold (\\mathbf), since \\textbf does not exist in mathtext.
        # Placed INSIDE the axes, right of the arrow head: the region right of the
        # non-Markovian mean is empty in both cells, and above the axes it would collide
        # with the frame ticks.
        ax.text(0.90, 0.66,
                f"KIVA {r['kiva_mak']:.1f}$\\to\\mathbf{{{r['kiva_non_mak']:.1f}}}$",
                transform=ax.transAxes, ha="right", va="center", fontsize=6.2,
                color=C_KIVA, zorder=7,
                bbox=dict(boxstyle="round,pad=0.20", fc="white", ec=C_KIVA, lw=0.5))

        ax.set_xlim(lo - 2.0 * width, hi + 2.0 * width)
        ax.set_ylim(0, ymax)
        ax.set_yticks([0, top])
        ax.set_yticklabels(["0", str(top)], fontsize=5.8)
        ax.tick_params(axis="y", length=1.8, pad=1.0, colors="#7a8085")
        ax.set_ylabel("clips", fontsize=5.8, color="#7a8085", labelpad=1.0)
        ax.tick_params(axis="x", labelsize=6.2, pad=1.2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_bounds(0, top)
        ax.spines["left"].set_linewidth(0.6)
        ax.set_title(f"Single-arm {r['model']}", loc="left", fontsize=7.0,
                     fontweight="bold", pad=13)

        # Twin top axis: the same positions labelled in frames (length = stride * chunks
        # + conditioning frames), so one axis reports depth and length at once.
        axt = ax.twiny()
        axt.set_xlim(ax.get_xlim())
        # Label the four support endpoints in frames, but drop any label closer than 7% of
        # the axis to the previous one: on DreamDojo, 181 and 193 frames are 1 chunk apart
        # and would print on top of each other.
        cand = sorted({am.min(), am.max(), an.min(), an.max()})
        ticks, min_sep = [], 0.07 * (hi - lo)
        for c in cand:
            if not ticks or c - ticks[-1] >= min_sep:
                ticks.append(c)
        axt.set_xticks(ticks)
        axt.set_xticklabels([str(int(round(c * stride + 1))) for c in ticks],
                            fontsize=5.6)
        axt.tick_params(axis="x", pad=0.8, length=1.8, colors="#7a8085")
        for sp in ("top", "right", "left", "bottom"):
            axt.spines[sp].set_visible(False)

    axes[0].set_xlabel("chunks per clip", fontsize=6.8, labelpad=1.5)
    axes[1].set_xlabel("chunks per clip", fontsize=6.8, labelpad=1.5)
    for ax in axes:
        ax.annotate("frames", xy=(-0.012, 1.19), xycoords="axes fraction", ha="right",
                    va="bottom", fontsize=5.8, color="#7a8085", style="italic")

    h = [plt.Rectangle((0, 0), 1, 1, color=C_MAK, alpha=0.9),
         plt.Rectangle((0, 0), 1, 1, color=C_KIVA, alpha=0.9),
         plt.Line2D([], [], color=C_KIVA, lw=0.8, ls=(0, (2.5, 2)), alpha=0.75)]
    fig.legend(h, ["Markovian", "non-Markovian", "regimes separate here"],
               loc="lower center", ncol=3, frameon=False, fontsize=6.2,
               bbox_to_anchor=(0.5, -0.07), handlelength=1.1, handleheight=0.85,
               columnspacing=1.1, handletextpad=0.5)

    FIG.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIG}.{ext}", dpi=300, bbox_inches="tight")
    print(f"[ok] {FIG.relative_to(REPO)}.pdf / .png")


def main():
    rows = collect()
    for r in rows:
        print(f"\nSingle-arm {r['model']}  stride={r['stride']}")
        for tag, f, c in (("M  ", r["frames_mak"], r["chunks_mak"]),
                          ("N  ", r["frames_non_mak"], r["chunks_non_mak"])):
            print(f"  {tag} n={len(f):3d}  frames {min(f)}-{max(f)} "
                  f"(mean {st.mean(f):.0f})  chunks {min(c)}-{max(c)} "
                  f"(mean {st.mean(c):.2f})")
        print(f"  disjoint={r['disjoint']}  empty band {r['gap'][0]}-{r['gap'][1]} frames "
              f"| ratio {r['depth_ratio']:.2f}x  CL={r['cl']}  p={r['p']:.1g}")
        print(f"  KIVA {r['kiva_mak']:.2f} -> {r['kiva_non_mak']:.2f} "
              f"({r['kiva_non_mak'] - r['kiva_mak']:+.2f})")

    OUT_JSON.write_text(json.dumps(dict(
        generated_by="assets/scripts/make_singlearm_depth_figure.py",
        purpose=("Clip-length distributions and rollout-depth ratio for the two "
                 "single-arm configurations, the two whose KIVA degrades."),
        sources=["results/reports/rollout_depth.json",
                 "results/reports/rq1_dense_baselines.json"],
        headline=("On both configurations the regime supports are DISJOINT: every "
                  "non-Markovian clip is deeper than every Markovian clip "
                  "(common-language effect 0.000)."),
        scope_limits=[
            "2 of 6 single-view cells, selected because they are the deepest. This figure "
            "explains a mechanism behind an already-reported result, it is not independent "
            "evidence for it.",
            "Disjoint supports mean regime and depth are perfectly confounded here, so "
            "this figure alone cannot separate 'deeper rollout' from 'non-Markovian "
            "task'. The within-clip cascade figure is what varies depth at fixed regime.",
            "Clip length is a property of the source episodes, not a model choice.",
        ],
        bar_counts_note=("chunks_histogram gives the number of clips behind every bar; "
                         "the figure normalises bar height per panel, so these counts "
                         "cannot be read off the plot and are recorded here."),
        cells=[{k: v for k, v in r.items()
                if k not in ("frames_mak", "frames_non_mak",
                             "chunks_mak", "chunks_non_mak")}
               | dict(chunks_histogram_mak=dict(sorted(
                          (str(k), v) for k, v in Counter(r["chunks_mak"]).items())),
                      chunks_histogram_non_mak=dict(sorted(
                          (str(k), v) for k, v in Counter(r["chunks_non_mak"]).items())))
               | dict(frames_range_mak=[min(r["frames_mak"]), max(r["frames_mak"])],
                      frames_range_non_mak=[min(r["frames_non_mak"]),
                                            max(r["frames_non_mak"])],
                      chunks_range_mak=[min(r["chunks_mak"]), max(r["chunks_mak"])],
                      chunks_range_non_mak=[min(r["chunks_non_mak"]),
                                            max(r["chunks_non_mak"])],
                      n_clips=[len(r["frames_mak"]), len(r["frames_non_mak"])])
               for r in rows],
    ), indent=1) + "\n")
    print(f"\n[ok] {OUT_JSON.relative_to(REPO)}")

    lines = ["model,stride,regime,chunks,frames,n_clips,mean_chunks,depth_ratio,"
             "kiva_mak,kiva_non_mak,mannwhitney_p,cl_effect"]
    for r in rows:
        for regime, chunks in (("makovian", r["chunks_mak"]),
                               ("non_makovian", r["chunks_non_mak"])):
            mean = st.mean(r["chunks_mak" if regime == "makovian" else "chunks_non_mak"])
            for ch, n in sorted(Counter(chunks).items()):
                lines.append(
                    f"{r['model']},{r['stride']},{regime},{ch},{ch * r['stride'] + 1},{n},"
                    f"{mean:.3f},{r['depth_ratio']},{r['kiva_mak']},{r['kiva_non_mak']},"
                    f"{r['p']},{r['cl']}")
    OUT_CSV.write_text("\n".join(lines) + "\n")
    print(f"[ok] {OUT_CSV.relative_to(REPO)}  ({len(lines) - 1} bars)")
    plot(rows)


if __name__ == "__main__":
    main()
