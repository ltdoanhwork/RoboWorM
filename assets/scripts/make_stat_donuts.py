#!/usr/bin/env python3
"""Headline donut charts for the benchmark's axes, plus a composite panel.

Writes ONE FILE PER CHART with a fully transparent background, so each can be
dropped onto a slide or a web page without a white box around it:

    assets/paper/figures/stat_embodiment.{png,pdf}
    assets/paper/figures/stat_view.{png,pdf}
    assets/paper/figures/stat_regime.{png,pdf}
    assets/paper/figures/stat_acceleration.{png,pdf}

and ONE COMBINED figure that lays the four donuts out 2x2 above a strip of
ground-truth frames, i.e. the "Benchmark Statistics" panel:

    assets/paper/figures/fig_stats_panel.{png,pdf}

Every number is read from the measured evidence, never hardcoded:
  * embodiment / view mix  <- metrics/reports/dataset_appendix_evidence.json
  * Markovian split        <- the same file's per-cell `split`
  * rollouts per strategy  <- metrics/reports/sampling_audit.json `groups`
  * primitive frames       <- metrics/reports/_hero_frames/primitives/index.tsv

The composite exists because this panel was previously assembled by hand, which
let it go stale: it shipped 1,608 episodes (the pre-filter count), a 45/55
Markovian split (the real split is 48/52) and labelled 29,343 ROLLOUTS as
"episodes". Regenerating is now the only supported way to update it.

Run:  python3 assets/scripts/make_stat_donuts.py
"""
from __future__ import annotations

import collections
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "assets/paper/figures"
EVID = REPO / "metrics/reports/dataset_appendix_evidence.json"
AUDIT = REPO / "metrics/reports/sampling_audit.json"
FRAMES = REPO / "metrics/reports/_hero_frames/primitives"

# Blue ramp, darkest = largest slice (the reference figure's convention).
RAMP = ["#1A56DB", "#2F6FE4", "#5A8DEC", "#93B4F3", "#C7D8F9", "#E4ECFC"]

ROBOT_LABEL = {
    "Franka": "Franka",
    "RoboChallenge": "ARX-5 / Franka",
    "ALOHA": "ALOHA",
    "GR1T2": "GR1-T2",
    "Airbot_MMK2": "AIRBOT MMK2",
    "discover_robotics_aitbot_mmk2": "AIRBOT MMK2",
}


def draw_donut(fig, rect, slices: list[tuple[str, int]], centre_big: str,
               centre_small: str, heading: str, *, scale: float = 1.0) -> None:
    """Draw one donut + hand-aligned legend into ``rect`` of ``fig``.

    ``rect`` is ``(x, y, w, h)`` in figure coordinates and covers the donut AND
    its legend, so the same routine serves the standalone files and the 2x2
    composite. The donut axes is forced SQUARE IN INCHES (not in figure
    fractions) -- otherwise a wide composite canvas squashes the ring and the
    centre label spills over it. A pie legend cannot align a right-hand value
    column, hence the manual legend.
    """
    total = sum(v for _, v in slices)
    x, y, w, h = rect
    fw, fh = fig.get_size_inches()

    # Square donut axes, as tall as the rect allows.
    d_h = h * 0.96
    d_w = d_h * fh / fw
    ax = fig.add_axes([x, y + (h - d_h) / 2, d_w, d_h])
    ax.patch.set_alpha(0.0)
    ax.set_aspect("equal")
    ax.pie([v for _, v in slices], startangle=90, counterclock=False,
           colors=RAMP[: len(slices)],
           wedgeprops=dict(width=0.30, edgecolor="none"))
    # The ring hole has radius 0.70; long numbers ("29,343") must be shrunk to
    # fit inside it, otherwise the label spills over the wedges.
    big_fs = 15.0 * scale * min(1.0, 4.0 / max(4, len(centre_big)))
    ax.text(0, 0.16, centre_big, ha="center", va="center",
            fontsize=big_fs, fontweight="bold", color="#1F2937")
    small_fs = 6.6 * scale * min(
        1.0, 11.0 / max(11, max(len(l) for l in centre_small.split("\n"))))
    ax.text(0, -0.22, centre_small, ha="center", va="center",
            fontsize=small_fs, color="#6B7280", linespacing=1.2)

    # Legend: heading on top, one row per slice, values right-aligned at x+w.
    lx = x + d_w + 0.035 * w
    rows = len(slices)
    dy = 0.132 * h * (1.0 if rows <= 3 else 0.86)
    ly = y + h / 2 + (rows - 1) * dy / 2
    fig.text(lx, ly + dy * 0.95, heading, fontsize=9.6 * scale,
             fontweight="bold", color="#1F2937", va="center")
    for i, (name, v) in enumerate(slices):
        yy = ly - i * dy
        fig.patches.append(plt.Rectangle(
            (lx, yy - 0.022 * h), 0.030 * w * 0.9, 0.044 * h,
            transform=fig.transFigure, facecolor=RAMP[i], edgecolor="none"))
        fig.text(lx + 0.046 * w, yy, name, fontsize=8.0 * scale, va="center",
                 color="#374151")
        fig.text(x + w, yy, f"{100 * v / total:.0f}%", fontsize=8.0 * scale,
                 va="center", ha="right", color="#111827")


def donut(slices: list[tuple[str, int]], centre_big: str, centre_small: str,
          heading: str, stem: str, unit: str = "episodes") -> None:
    """Standalone single-donut file on a transparent canvas."""
    fig = plt.figure(figsize=(4.4, 2.05))
    fig.patch.set_alpha(0.0)
    draw_donut(fig, (0.02, 0.04, 0.965, 0.92), slices, centre_big,
               centre_small, heading)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=400 if ext == "png" else None,
                    transparent=True)
    plt.close(fig)
    total = sum(v for _, v in slices)
    pct = ", ".join(f"{n} {100 * v / total:.0f}%" for n, v in slices)
    print(f"  {stem}: {total:,} {unit} -> {pct}")


def stats_panel(donuts: list[dict], title: str) -> None:
    """The combined 'Benchmark Statistics' panel: 2x2 donuts + frame strip.

    Frames and their primitive labels come from the hero-frame index, so the
    strip always shows the primitives in measured rank order. Missing frames
    degrade to a donuts-only panel rather than failing.
    """
    strip: list[tuple[Path, str]] = []
    idx = FRAMES / "index.tsv"
    if idx.exists():
        with open(idx) as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                p = FRAMES / row["file"]
                if p.exists():
                    strip.append((p, row["primitive"]))

    has_strip = bool(strip)
    fig_h = 4.55 if has_strip else 3.30
    fig = plt.figure(figsize=(8.6, fig_h))
    fig.patch.set_facecolor("white")

    # Title plate.
    top = 0.975
    plate_h = 0.075 if has_strip else 0.10
    fig.patches.append(plt.Rectangle((0.045, top - plate_h), 0.47, plate_h,
                                     transform=fig.transFigure,
                                     facecolor="#1E3A5F", edgecolor="none"))
    fig.text(0.068, top - plate_h / 2, title, fontsize=11.5,
             fontweight="bold", color="white", va="center", family="monospace")

    # 2x2 donut grid. The strip, when present, owns the bottom ~30%.
    band_top = top - plate_h - (0.045 if has_strip else 0.06)
    band_bot = 0.30 if has_strip else 0.04
    row_h = (band_top - band_bot - 0.03) / 2
    gap_x, col_w = 0.03, 0.46
    for i, d in enumerate(donuts):
        r, c = divmod(i, 2)
        x = 0.045 + c * (col_w + gap_x)
        y = band_top - (r + 1) * row_h - r * 0.03
        draw_donut(fig, (x, y, col_w, row_h), d["slices"], d["centre_big"],
                   d["centre_small"], d["heading"], scale=0.92)

    if has_strip:
        n = len(strip)
        sw = 0.91 / n
        for i, (p, label) in enumerate(strip):
            ax = fig.add_axes([0.045 + i * sw, 0.105, sw * 0.90, 0.155])
            ax.imshow(mpimg.imread(str(p)))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            fig.text(0.045 + i * sw + sw * 0.45, 0.055,
                     label.replace(" / ", "/\n").replace(" & ", " &\n"),
                     fontsize=6.8, ha="center", va="center",
                     fontweight="bold", color="#1F2937", linespacing=1.3)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_stats_panel.{ext}",
                    dpi=300 if ext == "png" else None,
                    facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  fig_stats_panel: {len(donuts)} donuts + {len(strip)} frames")


def main() -> int:
    evid = json.loads(EVID.read_text())
    cells = evid["cells"]

    # (1) embodiment mix, by episode instances
    emb = collections.Counter()
    robots: set[str] = set()
    for c in cells:
        emb[c["embodiment"]] += c["episodes_selected"]
        for r in c["robot_types"]:
            robots.add(ROBOT_LABEL.get(r, r))
    EMB_LABEL = {"bimanual": "Bimanual (ALOHA)",
                 "single_arm": "Single-arm (Franka, ARX-5)",
                 "humanoid": "Humanoid (GR1, MMK2)"}
    emb_slices = [(EMB_LABEL[k], v) for k, v in emb.most_common()]
    donut(emb_slices, f"{len(robots)}", "robot\nplatforms", "Embodiment",
          "stat_embodiment")

    # (2) camera tracks, by view videos (= episodes x views per episode)
    view = collections.Counter()
    rigs = set()
    for c in cells:
        view[c["view"]] += c["episodes_selected"] * c["views_per_episode"]
        rigs.add((c["embodiment"], c["view"]))
    view_slices = [("Multi-view (3-4 cameras)", view["multiview"]),
                   ("Single-view (1 camera)", view["singleview"])]
    donut(view_slices, f"{len(rigs)}", "camera\nconfigurations", "Camera view",
          "stat_view", unit="view videos")

    # (3) task-dynamics regime, by episode instances
    reg = collections.Counter()
    for c in cells:
        reg[c["split"]] += c["episodes_selected"]
    total_eps = sum(reg.values())
    reg_slices = [("Markovian", reg["makovian"]),
                  ("Non-Markovian", reg["non_makovian"])]
    donut(reg_slices, f"{total_eps:,}", "episode\ninstances", "Episode type",
          "stat_regime")

    # (4) acceleration strategy, by ROLLOUTS -- deliberately labelled
    # "rollouts", not "episodes": there are 29,343 rollouts over 1,605
    # episodes, and conflating the two is what the hand-made panel did.
    audit = json.loads(AUDIT.read_text())
    accel_slices = [(g["name"], g["total"]) for g in
                    sorted(audit["groups"], key=lambda g: -g["total"])]
    donut(accel_slices, f"{audit['grand_total']:,}", "rollouts",
          "Acceleration method", "stat_acceleration", unit="rollouts")

    # Combined panel, in the reading order of the hand-made original.
    panel = [
        dict(slices=reg_slices, centre_big=f"{total_eps:,}",
             centre_small="episode\ninstances", heading="Episode type"),
        dict(slices=view_slices, centre_big=f"{len(rigs)}",
             centre_small="camera\nconfigurations", heading="Camera view"),
        dict(slices=accel_slices, centre_big=f"{audit['grand_total']:,}",
             centre_small="rollouts", heading="Acceleration method"),
        dict(slices=emb_slices, centre_big=f"{len(robots)}",
             centre_small="robot\nplatforms", heading="Embodiment"),
    ]
    stats_panel(panel, "RoboWorM: Benchmark Statistics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
