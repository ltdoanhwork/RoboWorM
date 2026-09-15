"""Build a deduplicated visual-validation cohort from completed NR results.

The output is JSONL: one clip per line plus independent human targets. It contains
no aggregate judgment; three raters should annotate every row externally.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


TARGETS = {
    "visual_integrity": "image_collapse_0_2",
    "entity_presence": "expected_entities_visible_0_2",
    "task_lock": "text_condition_match_0_2",
    "actor_motion_raw": "robot_activity_0_2",
    "active_motion_continuity": "active_motion_smoothness_0_2",
    "actor_flicker_raw": "subject_shimmer_0_2",
    "camera_drift_raw": "external_camera_drift_0_2",
}


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def key(row):
    # Same source video may have several views; a view is valid independent unit.
    return row["id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--per-metric", type=int, default=24,
                    help="stratified clips per available metric")
    ap.add_argument("--seed", type=int, default=20260914)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = []
    for path in args.results.glob("*.json"):
        try:
            row = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if row.get("actor_validity") == "invalid":
            continue
        rows.append(row)
    if not rows:
        raise SystemExit("No valid result JSON files")

    selected, provenance = {}, {}
    for metric, target in TARGETS.items():
        candidates = [r for r in rows if finite(r.get(metric))]
        if metric in {"actor_motion_raw", "active_motion_continuity", "actor_flicker_raw"}:
            candidates = [r for r in candidates if r.get("actor_validity") == "robot_detected"]
        if metric == "camera_drift_raw":
            candidates = [r for r in candidates if "wrist" not in r.get("view", "").lower()]
        if metric in {"entity_presence", "task_lock"}:
            candidates = [r for r in candidates if r.get("text_condition_applicable", True)]
        if not candidates:
            continue
        candidates.sort(key=lambda r: r[metric])
        for i in range(args.per_metric):
            # One random item per quantile bin; prevents all samples being extremes.
            lo = i * len(candidates) // args.per_metric
            hi = max(lo + 1, (i + 1) * len(candidates) // args.per_metric)
            row = candidates[rng.randrange(lo, hi)]
            selected[key(row)] = row
            provenance.setdefault(key(row), []).append(metric)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for ident, row in sorted(selected.items()):
            targets = [TARGETS[m] for m in provenance[ident]]
            payload = {
                "id": ident,
                "video": row["path"], "view": row["view"],
                "model": row["model"], "embodiment": row["embodiment"],
                "split": row["split"], "instruction": row.get("instruction"),
                "sampling_metrics": provenance[ident],
                "human_targets": targets,
                "rater_fields": {t: None for t in sorted(set(targets))},
            }
            f.write(json.dumps(payload) + "\n")
    print(f"Wrote {len(selected)} clips to {args.out}")


if __name__ == "__main__":
    main()
