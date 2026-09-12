#!/usr/bin/env python3
"""Build webpage/labeling/metrics.multiview.json from data_for_web.csv.

Catalog ids (00000–00299) follow assets/all_videos.md order, which matches
the CSV row order one-to-one. That correspondence is checked against the
rendered clips before anything is written: it silently broke once, and a
score then describes a different video than the rater is watching.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CSV_PATH = REPO / "sampling_dataset" / "data_for_web" / "data_for_web.csv"
OUT_PATH = REPO / "webpage" / "labeling" / "metrics.multiview.json"
CLIPS = REPO / "sampling_dataset" / "data_for_web" / "assessment" / "multiview_static2" / "clips"


def check_clip_alignment(rows: list[dict]) -> list[dict]:
    if not CLIPS.is_dir():
        raise SystemExit(f"{CLIPS} not found — render the batch before building metrics")
    metas = []
    for i, row in enumerate(rows):
        meta_path = CLIPS / f"{i:05d}" / "meta.json"
        if not meta_path.is_file():
            raise SystemExit(f"clip {i:05d} is missing {meta_path}")
        meta = json.loads(meta_path.read_text())
        if meta.get("source_path") != row["video_path"]:
            raise SystemExit(
                f"clip {i:05d} was rendered from {meta.get('source_path')!r} but CSV row {i} "
                f"scores {row['video_path']!r} — rebuild the batch"
            )
        metas.append(meta)
    return metas


def main() -> None:
    rows = list(csv.DictReader(CSV_PATH.open()))
    metas = check_clip_alignment(rows)
    items = []
    for i, (row, meta) in enumerate(zip(rows, metas)):
        score = int(float(row["score"]))
        items.append(
            {
                "id": f"{i:05d}",
                "index": i,
                "check": "geometry",
                "video_path": row["video_path"],
                "score": score,
                "machine_score": score,
                "rating": {1: "bad", 2: "medium", 3: "good"}[score],
                "geometric_consistency": float(row["geometric_consistency"]),
                "pose_stability": float(row["pose_stability"]),
                "num_disjoint_frames": float(row["num_disjoint_frames"]),
                "disjoint_frame_ratio": float(row["disjoint_frame_ratio"]),
                "robot_type": row["robot_type"],
                "embodiment": row["robot_type"],
                "split": meta.get("split", ""),
                "task": meta.get("task", ""),
                "combined_score": float(row["combined_score"]),
            }
        )
    payload = {
        "source": "sampling_dataset/data_for_web/data_for_web.csv",
        "n": len(items),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": "Machine multiview consistency metrics; id matches catalog.json order.",
        "items": items,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT_PATH} ({len(items)} items)")


if __name__ == "__main__":
    main()
