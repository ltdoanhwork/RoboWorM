#!/usr/bin/env python3
"""KIVA audit numbers (NEW detector) as a checked, in-repo artifact.

WHY THIS FILE EXISTS
--------------------
`sampling_dataset/data_for_web/BENCHMARK_NR_main.xlsx` is AUTO-GENERATED
(`export_paper_grid_xlsx.py`), and its `KIVA-main` tab is copied verbatim out of
the older reference workbook. Typing the new numbers into that tab would be
overwritten the next time the grid is exported. So the new KIVA pass is
transcribed here, once, and every table reads it from
`results/reports/kiva_audit.csv`.

THIS REPLACES THE OLD KIVA COLUMN -- IT IS NOT THE SAME NUMBER
--------------------------------------------------------------
The detector changed, and so did the quantity. The two are not comparable and must
never be mixed in one column:

    embodiment   old KIVA-main tab        new pass (here)
    bimanual     real 0.5427, dense 0.7822   real 5.05,  dense 23.63
    humanoid     real 0.4566, dense 0.4898   real 4.98,  dense  4.65
    single_arm   real 0.6552, dense 1.2052   real 10.12, dense 59.40

The ratio between old and new is not even constant across embodiments (9.3x, 10.9x,
15.4x), which is the arithmetic proof that this is a different measurement rather
than a rescaling of the same one. Consequences that callers must respect:

  * every prose claim quoting an old KIVA figure (e.g. "1.44x above real video on
    bimanual") has to be recomputed from this file;
  * the old tab stays in the workbook as provenance but is no longer read.

WHAT IS NEW BESIDES THE SCALE
-----------------------------
DreamDojo is scored for the first time: the old tab had it empty in all three
embodiments, which is why the dense table printed `n/s` there. ITM and SiTo remain
unrendered for DreamDojo (`---`), which is a rendering gap, not a zero.

POLARITY: lower is better (KIVA = Kinematic In-VAlidity, share of segments whose
kinematics are invalid). Real video is NOT zero -- it sits at 5.0/5.0/10.1 -- so it
is a reference level, not a floor.

Usage:
    python3 assets/scripts/make_kiva_audit_data.py           # write + verify
    python3 assets/scripts/make_kiva_audit_data.py --check   # verify only
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_CSV = REPO / "results/reports/kiva_audit.csv"
OUT_JSON = REPO / "results/reports/kiva_audit.json"

# Method labels exactly as the benchmark workbook spells them, in its row order, so
# a reader can diff this block against the pasted sheet line by line.
METHODS = ["Dense", "itm", "sito", "World-cache", "Faster-cache", "Di-cache",
           "SVG2", "SVG1", "Radial Attention", "PISA"]

# (markovian, non_makovian); None = "---" in the sheet = never rendered, NOT a zero.
# Transcribed 2026-09-23 from the KIVA pass supplied by the metric author.
DATA = {
    "bimanual": {
        "real_video": (5.05, 5.05),
        "Dreamgen": [(23.63, 23.18), (60.87, 63.36), (67.18, 79.69),
                     (45.46, 47.02), (28.64, 29.09), (55.42, 57.93),
                     (46.36, 44.15), (47.27, 47.28), (80.53, 75.87),
                     (45.88, 58.11)],
        "DreamDojo": [(6.11, 0.0), None, None,
                      (2.56, 0.65), (2.35, 0.96), (2.07, 0.50),
                      (4.36, 0.65), (3.28, 0.79), (3.25, 0.69),
                      (3.41, 0.77)],
    },
    "humanoid": {
        "real_video": (4.98, 4.98),
        "Dreamgen": [(4.65, 12.30), (9.34, 14.28), (9.70, 17.59),
                     (9.76, 11.47), (7.47, 12.31), (6.31, 10.33),
                     (5.35, 8.31), (6.46, 11.28), (12.07, 14.33),
                     (7.47, 12.20)],
        "DreamDojo": [(13.04, 3.31), None, None,
                      (4.69, 18.62), (5.58, 16.19), (5.76, 17.75),
                      (10.54, 32.20), (7.26, 21.32), (5.93, 22.45),
                      (7.71, 22.63)],
    },
    "single_arm": {
        "real_video": (10.12, 10.12),
        "Dreamgen": [(59.40, 70.65), (69.77, 79.06), (80.40, 74.17),
                     (55.26, 69.46), (73.37, 85.77), (57.09, 67.88),
                     (60.63, 66.72), (66.17, 62.60), (62.96, 58.64),
                     (60.30, 55.09)],
        "DreamDojo": [(21.26, 26.98), None, None,
                      (14.45, 27.07), (13.75, 28.32), (17.09, 26.83),
                      (18.03, 24.69), (16.39, 28.84), (16.32, 30.03),
                      (16.47, 25.51)],
    },
}

# KIVA needs a body lifted from a single view, so it applies to the single-view
# tracks only. The sheet's short label -> the track name the tables use.
TRACK = {"Dreamgen": "Dreamgen SV", "DreamDojo": "DreamDojo SV"}


def verify():
    """Structural checks on the transcription. Raises on anything suspicious."""
    problems = []
    for emb, blk in DATA.items():
        rv = blk["real_video"]
        # Real video is one measurement of one clip pool; the sheet prints it twice
        # (once per regime) and the two copies must agree, or the transcription
        # picked up two different rows.
        if abs(rv[0] - rv[1]) > 1e-9:
            problems.append(f"{emb}: real video differs across regimes {rv}")
        for model in ("Dreamgen", "DreamDojo"):
            rows = blk[model]
            if len(rows) != len(METHODS):
                problems.append(
                    f"{emb}/{model}: {len(rows)} rows for {len(METHODS)} methods")
            for name, cell in zip(METHODS, rows):
                if cell is None:
                    continue
                for reg, v in zip(("M", "N"), cell):
                    # A share of segments: outside 0-100 the transcription slipped a
                    # decimal comma (the source sheet uses `,` as separator).
                    if not 0.0 <= v <= 100.0:
                        problems.append(f"{emb}/{model}/{name}/{reg}: {v} out of range")
            if rows[0] is None:
                problems.append(f"{emb}/{model}: no Dense row -- nothing to compare to")
    if problems:
        raise SystemExit("KIVA transcription FAILED:\n  " + "\n  ".join(problems))
    n = sum(1 for blk in DATA.values() for m in ("Dreamgen", "DreamDojo")
            for c in blk[m] if c is not None) * 2
    print(f"[ok] {n} KIVA values transcribed, {len(DATA)} real-video references, "
          f"all in 0-100 and regime-consistent")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only, write nothing")
    args = ap.parse_args()
    verify()
    if args.check:
        return

    rows = []
    for emb, blk in DATA.items():
        rows.append({"embodiment": emb, "model": "real_video", "track": "",
                     "method": "real_video", "makovian": blk["real_video"][0],
                     "non_makovian": blk["real_video"][1], "rendered": 1})
        for model in ("Dreamgen", "DreamDojo"):
            for name, cell in zip(METHODS, blk[model]):
                rows.append({
                    "embodiment": emb, "model": model, "track": TRACK[model],
                    "method": name,
                    "makovian": "" if cell is None else cell[0],
                    "non_makovian": "" if cell is None else cell[1],
                    "rendered": 0 if cell is None else 1,
                })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    OUT_JSON.write_text(json.dumps({
        "generated_by": "assets/scripts/make_kiva_audit_data.py",
        "detector": "new KIVA pass, transcribed 2026-09-23",
        "unit": "percentage of 16-frame segments whose kinematics are invalid",
        "polarity": "lower is better; real video is a reference level, not zero",
        "supersedes": (
            "the KIVA-main tab of sampling_dataset/data_for_web/"
            "BENCHMARK_NR_main.xlsx, which holds an EARLIER detector on a different "
            "scale (old/new ratio is 9.3x / 10.9x / 15.4x across embodiments, i.e. "
            "not a rescaling). Old values must not be mixed with these."),
        "applies_to": "single-view body tracks only (KIVA lifts a body from one view)",
        "not_rendered": "itm and sito on DreamDojo (all embodiments)",
        "new_since_old_tab": "DreamDojo, which the old tab left empty everywhere",
        "rows": rows,
    }, indent=2) + "\n")
    print(f"[ok] {OUT_CSV.relative_to(REPO)}   {len(rows)} rows")
    print(f"[ok] {OUT_JSON.relative_to(REPO)}")


if __name__ == "__main__":
    main()
