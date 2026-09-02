#!/usr/bin/env python3
"""Build webpage/labeling/metrics.multiview.json from data_for_web.csv.

Catalog ids (00000–00299) follow assets/all_videos.md order, which matches
the CSV row order one-to-one.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CSV_PATH = REPO / "sampling_dataset" / "data_for_web" / "data_for_web.csv"
OUT_PATH = REPO / "webpage" / "labeling" / "metrics.multiview.json"


def main() -> None:
    rows = list(csv.DictReader(CSV_PATH.open()))
    items = []
    for i, row in enumerate(rows):
        items.append(
            {
                "id": f"{i:05d}",
                "video_path": row["video_path"],
                "score": int(float(row["score"])),
                "geometric_consistency": float(row["geometric_consistency"]),
                "pose_stability": float(row["pose_stability"]),
                "num_disjoint_frames": float(row["num_disjoint_frames"]),
                "disjoint_frame_ratio": float(row["disjoint_frame_ratio"]),
                "robot_type": row["robot_type"],
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
