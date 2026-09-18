#!/usr/bin/env python3
"""Export the gold hallucinate label for every segment — one final score each.

Precedence, as resolved by the deployed VGP.resolveSegments and carried through
`hallucinate_final_labels.json`:

    expert (doanh_check)  >  checker  >  the two raters' consensus

So `gold` here is the expert's score wherever the expert has rated, and the
earlier resolution everywhere else. Nothing is recomputed in this script; it
flattens what the export already decided.

Provenance is kept per segment (`source`) because the two halves are not the same
kind of label. The expert re-rated with our metric's band and the prior human
label visible on screen, and on the segments measured so far their score matched
our band in 97% of cases (see results/hallucinate_final/EXPERT_PASS_AUDIT.md).
Any overlap statistic computed against the `expert` rows therefore reflects that
display, not independent human agreement — the `source` column is what lets a
later reader separate the two.

Each row also carries the individual scores behind the gold value: both raters
(every segment has exactly two), the checker who adjudicated if there was one, and
the expert's score if the expert reached that segment. `raters_agree` is 1 when the
two raters matched, which is also the condition under which no checker was needed.

Outputs (results/hallucinate_final/):
  gold_segments.csv    gold score + level + source, plus rater1/rater2/checker/expert
  gold_segments.json   the same, plus counts by source and by class
  gold_segments.jsonl  one compact record per line, for streaming consumers

Usage:
    python3 scripts/export_gold_segment_labels.py
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXPORT = REPO / "sampling_dataset/data_for_web/hallucinate_final_labels.json"
CATALOG = REPO / ".main-edit/webpage/labeling/catalog.segment.json"
OUT = REPO / "results/hallucinate_final"

# humans rate quality (3 = Good); the bench speaks in amount of hallucination
SCORE_LABEL = {1: "Bad", 2: "Medium", 3: "Good"}
SCORE_LEVEL = {3: "low", 2: "medium", 1: "high"}


def bucket_ids() -> dict[str, str]:
    """item_id -> the bucket's segment_id (video_path#index), for joining."""
    cat = json.loads(CATALOG.read_text())
    return {
        f"{it['id']}_s{s['index']:03d}": f"{it['video_path']}#{s['index']}"
        for it in cat["items"]
        for s in it["segments"]
    }


def main() -> None:
    exp = json.loads(EXPORT.read_text())
    keymap = bucket_ids()

    rows = []
    for v in exp["videos"]:
        for s in v["segments"]:
            score = s["human_score"]
            if score is None:
                continue
            # every segment carries exactly two raters and at most one checker
            # (verified over all 8093); the expert is a separate pass on top
            raters = [(k, x) for k, x in s["rater_scores"].items() if x is not None]
            checkers = [(k, x) for k, x in s["checker_scores"].items() if x is not None]
            r1, r2 = (raters + [(None, None), (None, None)])[:2]
            ck = (checkers + [(None, None)])[0]
            rows.append({
                "segment_id": keymap.get(s["item_id"]),
                "item_id": s["item_id"],
                "video_id": v["id"],
                "embodiment": v.get("embodiment"),
                "split": v.get("split"),
                "segment_index": s["index"],
                "start_frame": s.get("start_frame"),
                "end_frame": s.get("end_frame"),
                "score": score,
                "label": SCORE_LABEL[score],
                "level": SCORE_LEVEL[score],
                "source": s["consensus"],
                "rater1_name": r1[0],
                "rater1_score": r1[1],
                "rater2_name": r2[0],
                "rater2_score": r2[1],
                "raters_agree": (
                    None if r1[1] is None or r2[1] is None else int(r1[1] == r2[1])
                ),
                "checker_name": ck[0],
                "checker_score": ck[1],
                "expert_score": s.get("expert_score"),
            })

    rows.sort(key=lambda r: (r["video_id"], r["segment_index"]))
    by_source = collections.Counter(r["source"] for r in rows)
    by_score = collections.Counter(r["score"] for r in rows)

    OUT.mkdir(parents=True, exist_ok=True)

    cols = ["segment_id", "item_id", "video_id", "embodiment", "split",
            "segment_index", "start_frame", "end_frame", "score", "label",
            "level", "source", "rater1_name", "rater1_score", "rater2_name",
            "rater2_score", "raters_agree", "checker_name", "checker_score",
            "expert_score"]
    with (OUT / "gold_segments.csv").open("w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join("" if r[c] is None else str(r[c]) for c in cols) + "\n")

    with (OUT / "gold_segments.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps({
                "segment_id": r["segment_id"], "item_id": r["item_id"],
                "score": r["score"], "level": r["level"], "source": r["source"],
                "raters": {r["rater1_name"]: r["rater1_score"],
                           r["rater2_name"]: r["rater2_score"]},
                "checker": ({r["checker_name"]: r["checker_score"]}
                            if r["checker_name"] else {}),
                "expert_score": r["expert_score"],
            }) + "\n")

    (OUT / "gold_segments.json").write_text(json.dumps({
        "check": "hallucinate",
        "n_segments": len(rows),
        "n_videos": len({r["video_id"] for r in rows}),
        "precedence": ["expert", "checker", "raters"],
        "scale": {"score": SCORE_LABEL, "level_for_score": SCORE_LEVEL},
        "by_source": dict(by_source),
        "by_score": {str(k): v for k, v in sorted(by_score.items())},
        "note": ("gold = expert (doanh_check) score where present, otherwise the "
                 "checker/rater resolution. The expert rated with our metric's band "
                 "and the prior human label on screen, so statistics computed only "
                 "over source='expert' rows are not independent of our metric; keep "
                 "the source column when reporting."),
        "segments": rows,
    }, indent=2) + "\n")

    print(f"segments        : {len(rows)}  ({len({r['video_id'] for r in rows})} videos)")
    print(f"missing bucket id: {sum(1 for r in rows if not r['segment_id'])}")
    print("\nby source (gold provenance):")
    for k, n in by_source.most_common():
        print(f"  {k:9s} {n:5d}  {n / len(rows):6.1%}")
    print("\nby score:")
    for k in (1, 2, 3):
        n = by_score.get(k, 0)
        print(f"  {k} {SCORE_LABEL[k]:6s} ({SCORE_LEVEL[k]:6s}) {n:5d}  {n / len(rows):6.1%}")
    agree = [r for r in rows if r["raters_agree"] == 1]
    print(f"\ntwo raters agreed: {len(agree)}/{len(rows)} = {len(agree) / len(rows):.1%}")
    print(f"checker present  : {sum(1 for r in rows if r['checker_name'])}")
    print(f"expert present   : {sum(1 for r in rows if r['expert_score'] is not None)}")
    print(f"\nwrote {(OUT / 'gold_segments.csv').relative_to(REPO)}")
    print(f"      {(OUT / 'gold_segments.json').relative_to(REPO)}")
    print(f"      {(OUT / 'gold_segments.jsonl').relative_to(REPO)}")


if __name__ == "__main__":
    main()
