#!/usr/bin/env python3
"""Matched real-video anchors per (embodiment, model, camera track, split).

WHY THIS FILE EXISTS
--------------------
The RQ1 table reads every generated row against real video. Which real-video
number is the *correct* comparison is not a free choice: Actor Activity is an
optical-flow statistic in px/frame, so it is only comparable on a matched grid
(the `phyworld-benchmark` protocol). Concretely, on bimanual Markovian clips the
GT pools disagree by a factor of three:

    gt_dreamgen (2x2 grid)          n=590   0.911 px/frame
    gt_ctrlworld_4view_grid         n=593   2.986
    gt_ctrlworld (3-view strip)     n=220   2.740

so a single pooled "Real Video" row (which `table_paper_grid.py` builds by pooling
every model's GT inside an embodiment, median 1.775) is not the anchor of any
individual row. Reading DreamGen MV's dense 0.961 against the pooled 1.775 makes
it look like it loses 46% of the motion; read against its OWN grid it is level
with real video. This script therefore emits the per-track anchor, and the table
reads each row against the anchor of its own geometry.

Two GT pools are deliberately EXCLUDED from being used as anchors:

  * `gt` with no model suffix -- these are per-view GT files taken from the input
    tree (`.../input/multiview/<model>/.../view_2.mp4`), i.e. one camera at a time,
    not the multi-view grid the predictions are rendered as. Flow magnitude is
    resolution- and layout-dependent, so a single view is not a matched reference
    for a grid. (It is also the pool that inflates the pooled row above.)
  * anything failing `comparable_pair` against the dense row it would anchor.

Coverage is reported honestly rather than back-filled: single-view GT was not
scored for bimanual/humanoid in this run, so those rows get no anchor and the
table shows the gap instead of borrowing a number from another geometry.

Cost: loads every clip-level record of the run (~82k), about 5 minutes. Output is
a small CSV so the table generator does not repeat it.

Usage:
    python3 assets/scripts/make_rq1_matched_anchors.py
    python3 assets/scripts/make_rq1_matched_anchors.py \
        --run metrics/WorldScore/results/dense_nr_full
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import table_dense_nr as T      # noqa: E402
import table_paper_grid as G    # noqa: E402

COLS = ["visual_integrity", "text_condition_alignment", "actor_motion_raw",
        "active_motion_continuity"]
SPLITS = ["makovian", "non_makovian"]
OUT = REPO / "results/reports/rq1_matched_anchors.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="metrics/WorldScore/results/dense_nr_full")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rows = T.load_records(REPO / args.run if not Path(args.run).is_absolute()
                          else Path(args.run))

    # (embodiment, model, track, split, metric) -> values, for GT and dense alike.
    # dense is carried too so the table's own numbers can be re-derived from the
    # same pass and checked against the workbook.
    acc = collections.defaultdict(list)
    for r in rows:
        track = G.track_of(r)
        if track is None:
            continue
        method = r.get("method") or "gt"
        if method not in ("gt", "dense"):
            continue
        model = r["model"]
        if method == "gt":
            # Only model-scoped GT pools are matched references; see module docstring.
            if not model.startswith("gt_"):
                continue
            model = model[3:]
        for col in COLS:
            v = T.gated(r, col)
            if v is not None:
                acc[(r["embodiment"], model, track, r["split"], method, col)].append(v)

    out_rows = []
    keys = sorted({k[:4] for k in acc})
    for emb, model, track, split in keys:
        for method in ("gt", "dense"):
            rec = {"embodiment": emb, "model": model, "track": track,
                   "split": split, "role": "matched_anchor" if method == "gt"
                   else "dense"}
            any_val = False
            for col in COLS:
                vals = acc.get((emb, model, track, split, method, col), [])
                enough = len(vals) >= T.MIN_PAIRS_FOR_CLAIM
                rec[col] = round(T.median(vals), 4) if enough else ""
                rec[f"{col}_n"] = len(vals)
                any_val |= enough
            if any_val:
                out_rows.append(rec)

    fields = (["embodiment", "model", "track", "split", "role"]
              + [c for col in COLS for c in (col, f"{col}_n")])
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    n_anchor = sum(r["role"] == "matched_anchor" for r in out_rows)
    print(f"[ok] {dest.relative_to(REPO)}: {len(out_rows)} rows "
          f"({n_anchor} matched anchors, {len(out_rows) - n_anchor} dense)")
    for r in out_rows:
        if r["role"] == "matched_anchor" and r["split"] == "makovian":
            print(f"     anchor {r['embodiment']:11s} {r['model']:22s} "
                  f"{r['track']:10s} actor={r['actor_motion_raw']} "
                  f"(n={r['actor_motion_raw_n']})")


if __name__ == "__main__":
    main()
