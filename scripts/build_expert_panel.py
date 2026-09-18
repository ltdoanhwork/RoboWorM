#!/usr/bin/env python3
"""Build the per-segment panel doanh_check sees while re-rating.

The expert pass shows, for every segment in the queue, what the metric said and
what the current human label is, BEFORE the expert scores it. This is the user's
explicit decision: the expert observes both and decides.

What that costs, recorded here so nobody has to rediscover it: the overlap
computed from the expert's labels can no longer be read as an independent
measurement of the metric. The expert sees our band first, so any pull toward it
raises overlap for reasons unrelated to the metric's quality, and the control
stratum cannot detect that pull (it acts in the same direction in both strata).
The expert labels remain a legitimate CORRECTED reference for error analysis and
for showing where the pipeline cuts sit wrong; they are not a clean benchmark
number for "how well does our metric agree with humans".

Panel content per segment id:
  ours        band + label_ratio + per-detector ratios + violated flag
  human       current resolved score, who settled it, and both raters' scores
  agreement   whether they match, and how they differ

Written to webpage/labeling/expert_panel.hallucinate.json.

Usage:
    python3 scripts/build_expert_panel.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "sampling_dataset/data_for_web/hallucinate_bench"
METRICS = BENCH / "metrics.segment.json"
SITE = REPO / ".main-edit/webpage/labeling"
QUEUE = SITE / "expert_queue.hallucinate.json"
EXPORT = REPO / "sampling_dataset/data_for_web/hallucinate_final_labels.json"
OUT = SITE / "expert_panel.hallucinate.json"

# humans rate quality (3=Good); the bucket rates amount of hallucination
H2L = {3: "low", 2: "medium", 1: "high"}
SCORE_LABEL = {1: "Bad", 2: "Medium", 3: "Good"}


def main() -> None:
    metrics = json.loads(METRICS.read_text())
    by_id = {it["id"]: it for it in metrics["items"]}
    thresholds = metrics.get("segment_threshold", {})
    detectors = metrics.get("detectors_used", {})

    queue = set(json.loads(QUEUE.read_text())["segments"])
    exp = json.loads(EXPORT.read_text())

    panel = {}
    for v in exp["videos"]:
        for s in v["segments"]:
            uid = s["item_id"]
            if uid not in queue:
                continue
            m = by_id.get(uid)
            if m is None:
                continue
            hs = s["human_score"]
            ours = None
            ratio = m.get("ratio")
            if ratio is not None:
                ours = "low" if ratio < 1.0 else ("medium" if ratio < 1.5 else "high")
            panel[uid] = {
                "ours_band": ours,
                "ours_score": {"low": 3, "medium": 2, "high": 1}.get(ours),
                "ratio": ratio,
                "ratio_rigidity": m.get("ratio_rigidity"),
                "ratio_jerk": m.get("ratio_jerk"),
                "violated": m.get("violated"),
                "detectors": detectors.get(m.get("embodiment"), []),
                "threshold": thresholds.get(m.get("embodiment"), {}),
                "human_score": hs,
                "human_label": SCORE_LABEL.get(hs),
                "human_band": H2L.get(hs),
                "settled_by": s["consensus"],
                "rater_scores": s["rater_scores"],
                "checker_scores": s["checker_scores"],
                "match": (ours == H2L.get(hs)) if (ours and hs) else None,
            }

    OUT.write_text(json.dumps({
        "check": "hallucinate",
        "role": "expert",
        "n": len(panel),
        "note": ("Shown to doanh_check BEFORE scoring, by explicit request: the metric's "
                 "band/ratio and the current human label side by side. Consequence: the "
                 "overlap recomputed from the expert labels is not an independent "
                 "measurement of the metric (the expert saw the metric first), so cite it "
                 "as a corrected reference for error analysis, not as a benchmark score."),
        "scale": {"score": SCORE_LABEL, "band_for_score": H2L},
        "segments": panel,
    }, indent=2) + "\n")

    n_match = sum(1 for v in panel.values() if v["match"])
    print(f"panel entries : {len(panel)} (queue {len(queue)})")
    print(f"missing metric: {len(queue) - len(panel)}")
    print(f"ours == human : {n_match}  (control stratum)")
    print(f"ours != human : {len(panel) - n_match}  (disagree stratum)")
    print(f"\nwrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
