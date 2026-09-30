#!/usr/bin/env python3
"""Export per-video scores for the two activity metrics into two JSON files.

  * Active Rate            -> `actor_motion_active_rate`
  * Active Motion Continuity -> `active_motion_continuity`

Both already exist per clip in `<run>/results/<id>.json`; this only collects them.

Two things this is careful about, because both silently corrupt an export:

1. `active_motion_continuity` is legitimately `null` on many clips -- AMC is only
   defined when a clip has enough ACTIVE actor pairs to weight (a static clip has
   nothing to be continuous about). Nulls are kept in the file with an explicit
   `value: null` plus the reason, and counted separately, rather than dropped
   (dropping them would make the file look complete while silently excluding
   exactly the clips the metric could not score).
2. One clip id per FILE, but a video can appear under several `view_index`
   values (multi-view grids are scored per view). The key is therefore the
   result id, and `relative` / `view` are carried so a reader can regroup.

    python3 assets/scripts/export_activity_scores.py
    python3 assets/scripts/export_activity_scores.py --run dense_nr_full --out-dir metrics/reports
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "metrics/WorldScore/results"

FIELDS = ("id", "relative", "path", "model", "method", "embodiment", "split",
          "view", "view_index", "layout", "camera_mount", "n_views")


def collect(run_dir: Path, key: str, extra: tuple[str, ...] = ()):
    files = sorted((run_dir / "results").glob("*.json"))
    if not files:
        raise SystemExit(f"no result json under {run_dir/'results'}")
    rows, n_null, n_missing, n_bad = [], 0, 0, 0
    for fp in files:
        try:
            d = json.loads(fp.read_text())
        except Exception:
            n_bad += 1
            continue
        if key not in d:
            n_missing += 1
            continue
        v = d[key]
        rec = {f: d.get(f) for f in FIELDS}
        rec["value"] = v
        for e in extra:
            rec[e] = d.get(e)
        if v is None:
            n_null += 1
            # why it is null, so the gap is auditable rather than mysterious
            rec["null_reason"] = (
                "metric undefined for this clip: no/insufficient ACTIVE actor "
                "pairs to weight (weight_sum="
                f"{d.get('active_motion_continuity_weight_sum')!r}, "
                f"n_active_pairs={d.get('actor_n_active_pairs')!r})")
        rows.append(rec)
    return rows, dict(n_files=len(files), n_rows=len(rows), n_null=n_null,
                      n_missing_key=n_missing, n_unreadable=n_bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="dense_nr_full",
                    help="run dir under metrics/WorldScore/results/")
    ap.add_argument("--out-dir", default=str(ROOT / "metrics/reports"))
    args = ap.parse_args()

    run_dir = RESULTS / args.run
    if not run_dir.is_dir():
        raise SystemExit(f"no such run dir: {run_dir}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("actor_motion_active_rate", "active_rate_per_video.json",
         "Active Rate: fraction of valid actor pairs moving at >= the active "
         "threshold (px/source-frame). Range [0,1]; higher = more active.",
         ("actor_n_active_pairs", "actor_motion_valid_pair_frac")),
        ("active_motion_continuity", "active_motion_continuity_per_video.json",
         "Active Motion Continuity (AMC): activity-weighted VFI gain over frame "
         "copy x100. Range [0,100]; higher = smoother active motion. null when "
         "the clip has too little active motion to score.",
         ("active_motion_continuity_weight_sum", "actor_n_active_pairs")),
    ]

    for key, fname, desc, extra in specs:
        rows, stats = collect(run_dir, key, extra)
        vals = [r["value"] for r in rows if r["value"] is not None]
        payload = {
            "metric": key,
            "description": desc,
            "source_run": str(run_dir.relative_to(ROOT)),
            "n_videos": len(rows),
            "n_scored": len(vals),
            "n_null": stats["n_null"],
            "summary": ({"min": min(vals), "max": max(vals),
                         "mean": sum(vals) / len(vals)} if vals else None),
            "scan": stats,
            "videos": rows,
        }
        out = out_dir / fname
        out.write_text(json.dumps(payload, indent=2, allow_nan=False))
        miss = f", {stats['n_missing_key']} missing key" if stats["n_missing_key"] else ""
        print(f"{key}")
        print(f"  -> {out}")
        print(f"     {len(rows)} videos, {len(vals)} scored, {stats['n_null']} null{miss}")
        if vals:
            print(f"     min={min(vals):.4f} mean={sum(vals)/len(vals):.4f} max={max(vals):.4f}")


if __name__ == "__main__":
    main()
