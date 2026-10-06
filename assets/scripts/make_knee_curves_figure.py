#!/usr/bin/env python3
"""Quality-versus-speedup sweep curves, with the shipped operating point marked.

This is the figure that justifies *which* configuration each method ships. The
appendix tables give one row per cell; this shows the sweep that row was chosen
from, so a reader can see that the shipped point sits at the knee rather than
being asserted to.

Data source: the per-cell tuning-insight rules under ``.cursor/rules/*-insights.mdc``.
Those files are the primary record of every sweep run in this project -- each
contains a markdown table of (method, configuration, time, speedup, PSNR, SSIM,
LPIPS, verdict). They are parsed here rather than retyped, so the figure cannot
drift from the record.

Two honest properties of this data are preserved rather than hidden:

  * Sweeps were run per cell with that cell's own episode count, so points are
    comparable *within* a panel and not across panels. Each panel is therefore
    normalised to its own dense baseline and the axes are relative.
  * A sweep point slower than dense (speedup $<1$) is kept and drawn to the left
    of the dense line. Several methods genuinely lose on several backbones, and
    deleting those points would misrepresent the benchmark.

    python3 assets/scripts/make_knee_curves_figure.py

Writes:
    assets/paper/figures/fig_knee_curves.pdf
    results/reports/knee_curves.json
"""

from __future__ import annotations

import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RULES = os.path.join(REPO, ".cursor/rules")
OUT_PDF = os.path.join(REPO, "assets/paper/figures/fig_knee_curves.pdf")
OUT_JSON = os.path.join(REPO, "results/reports/knee_curves.json")

METHOD_CANON = {
    "dense": "dense", "sito": "SiTo", "itm": "ITM",
    "worldcache": "WorldCache", "dicache": "DiCache", "fastercache": "FasterCache",
    "pisa": "PISA", "svg": "SVG", "svg1": "SVG", "svg2": "SVG2",
    "radial": "Radial",
}
COLOR = {
    "SiTo": "#e76f51", "ITM": "#f4a261",
    "WorldCache": "#2a9d8f", "DiCache": "#264653", "FasterCache": "#8ab17d",
    "PISA": "#c1121f", "SVG": "#457b9d", "SVG2": "#6a4c93", "Radial": "#b5179e",
}
MARKER = {
    "SiTo": "o", "ITM": "s", "WorldCache": "^", "DiCache": "v",
    "FasterCache": "D", "PISA": "*", "SVG": "P", "SVG2": "X", "Radial": "h",
}

# Panels to draw, in order: (rule stem, panel title). Chosen to span the three
# backbones and both the "sparse attention wins" and "sparse attention loses"
# regimes, which is the comparison the figure exists to make.
PANELS = [
    ("dreamgen-bimanual-multiview-sparse-attention-insights",
     "DreamGen 2B, bimanual MV"),
    ("dreamgen-humanoid-multiview-caching-insights",
     "DreamGen 2B, humanoid MV"),
    ("dreamdojo-humanoid-singleview-sparse-attention-insights",
     "DreamDojo 2B, humanoid SV"),
    # Ctrl-World is shown on the 4-view GRID, not the 3-view strip: the strip is
    # not a reported cell for humanoid (see assets/scripts/benchmark_cells.py),
    # so a panel drawn from its sweep would illustrate a cell absent from every
    # table in the paper.
    ("ctrlworld-humanoid-multiview-4view-grid-caching-insights",
     "Ctrl-World U-Net, humanoid MV (4-view)"),
]

BOLD = re.compile(r"\*\*")
NUM = re.compile(r"-?\d+\.?\d*")
BREAK_WORDS = re.compile(r"CORRUPT|BREAK|MELTED|DESTROYED|corrupt|melt", re.I)
GOOD_WORDS = re.compile(r"BEST|DEFAULT|OPTIMAL|knee|KNEE")


def strip(cell):
    return BOLD.sub("", cell).strip()


def parse_tables(path):
    """Every markdown table in a rule file, as a list of dicts."""
    tables = []
    rows, header = [], None
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith("|"):
            cells = [strip(c) for c in line.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells) and cells:
                continue  # separator
            if header is None:
                header = [c.lower() for c in cells]
            else:
                if len(cells) == len(header):
                    rows.append(dict(zip(header, cells)))
        else:
            if header and rows:
                tables.append((header, rows))
            header, rows = None, []
    if header and rows:
        tables.append((header, rows))
    return tables


def pick_column(header, *candidates):
    for cand in candidates:
        for h in header:
            if cand in h:
                return h
    return None


def extract_sweep(path):
    """Pull (method, config, speedup, quality) points out of a rule file.

    A table qualifies if it has a speedup column and at least one quality column.
    PSNR is preferred as the quality axis because every sweep reports it; SSIM is
    used when PSNR is absent.
    """
    best = None
    for header, rows in parse_tables(path):
        sp = pick_column(header, "speedup")
        if not sp:
            continue
        q = pick_column(header, "psnr", "meanpsnr", "ssim")
        if not q:
            continue
        m = pick_column(header, "method")
        cfg = pick_column(header, "config")
        verdict = pick_column(header, "quality", "verdict")
        pts = []
        for r in rows:
            sp_v = NUM.search(r.get(sp, "") or "")
            q_v = NUM.search(r.get(q, "") or "")
            if not (sp_v and q_v):
                continue
            raw_m = (r.get(m, "") or "").lower().strip()
            canon = METHOD_CANON.get(raw_m)
            if canon is None:
                continue
            note = r.get(verdict, "") if verdict else ""
            pts.append({
                "method": canon,
                "config": r.get(cfg, "") if cfg else "",
                "speedup": float(sp_v.group()),
                "quality": float(q_v.group()),
                "quality_metric": "PSNR" if "psnr" in q else "SSIM",
                "note": note,
                "is_break": bool(BREAK_WORDS.search(note)),
                "is_chosen": bool(GOOD_WORDS.search(note)),
            })
        if pts and (best is None or len(pts) > len(best[0])):
            best = (pts, q)
    return best[0] if best else []


def plot(panels, path):
    n = len(panels)
    ncol = 2
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(11.0, 4.1 * nrow))
    axes = axes.ravel() if n > 1 else [axes]

    for ax, (title, pts, metric) in zip(axes, panels):
        dense = next((p for p in pts if p["method"] == "dense"), None)
        dense_q = dense["quality"] if dense else None

        # Dense reference lines: the reader should be able to see at a glance
        # which points are actually faster AND not worse.
        ax.axvline(1.0, color="#444444", lw=1.0, ls="--", zorder=1)
        if dense_q is not None:
            ax.axhline(dense_q, color="#444444", lw=1.0, ls=":", zorder=1)
            ax.text(0.015, dense_q, "dense", fontsize=7, color="#444444",
                    va="bottom", ha="left", transform=ax.get_yaxis_transform())

        by_method = {}
        for p in pts:
            if p["method"] == "dense":
                continue
            by_method.setdefault(p["method"], []).append(p)

        for method, ps in sorted(by_method.items()):
            ps = sorted(ps, key=lambda p: p["speedup"])
            col = COLOR.get(method, "#888888")
            # Deliberately NOT connected: the points of one method are distinct
            # configurations, not samples along a single monotone parameter, so a
            # line would imply an ordering the sweep does not have.
            for p in ps:
                if p["is_break"]:
                    # A configuration whose output visibly degraded: drawn hollow
                    # with a cross so it is never read as a usable operating point.
                    ax.scatter(p["speedup"], p["quality"], s=64, facecolors="none",
                               edgecolors=col, linewidths=1.3, marker="X", zorder=3)
                else:
                    ax.scatter(p["speedup"], p["quality"], s=48, color=col,
                               marker=MARKER.get(method, "o"), zorder=3,
                               edgecolors="white", linewidths=0.5)
                if p["is_chosen"]:
                    # The shipped operating point.
                    ax.scatter(p["speedup"], p["quality"], s=235,
                               facecolors="none", edgecolors=col, linewidths=1.9,
                               zorder=4)
            ax.plot([], [], marker=MARKER.get(method, "o"), color=col, lw=0,
                    label=method, markersize=5)

        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("speedup vs. dense ($\\times$)")
        ax.set_ylabel(metric)
        ax.legend(fontsize=7.5, ncol=2, loc="lower left", frameon=True,
                  framealpha=0.92, edgecolor="none")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(color="#f0f0f0")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=185, bbox_inches="tight")
    plt.close(fig)


def main():
    panels, payload = [], []
    for stem, title in PANELS:
        path = os.path.join(RULES, stem + ".mdc")
        if not os.path.exists(path):
            print("WARNING: missing rule %s" % stem, file=sys.stderr)
            continue
        pts = extract_sweep(path)
        if not pts:
            print("WARNING: no sweep parsed from %s" % stem, file=sys.stderr)
            continue
        metric = pts[0]["quality_metric"]
        panels.append((title, pts, metric))
        payload.append({"panel": title, "source": ".cursor/rules/%s.mdc" % stem,
                        "quality_metric": metric, "n_points": len(pts),
                        "points": pts})

    if not panels:
        print("ERROR: no sweep data found", file=sys.stderr)
        return 1

    plot(panels, OUT_PDF)

    with open(OUT_JSON, "w") as fh:
        json.dump({
            "generated_by": "assets/scripts/make_knee_curves_figure.py",
            "purpose": "Quality-versus-speedup sweeps per backbone, with the "
                       "shipped operating point and the configurations that "
                       "visibly degraded both marked.",
            "source": "the per-cell tuning-insight rules under .cursor/rules/, "
                      "which are the primary record of every sweep run",
            "caveat": "sweeps were run per cell with that cell's own episode "
                      "count, so points are comparable within a panel but not "
                      "across panels; configurations slower than dense are kept "
                      "rather than dropped",
            "panels": payload,
        }, fh, indent=2)

    print("wrote %s" % os.path.relpath(OUT_PDF, REPO))
    print("wrote %s" % os.path.relpath(OUT_JSON, REPO))
    print("")
    for title, pts, metric in panels:
        chosen = [p for p in pts if p["is_chosen"]]
        broke = [p for p in pts if p["is_break"]]
        print("%-34s %2d points, %d shipped, %d degraded (%s)" % (
            title, len(pts), len(chosen), len(broke), metric))
    return 0


if __name__ == "__main__":
    sys.exit(main())
