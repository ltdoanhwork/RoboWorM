#!/usr/bin/env python3
"""Build the expert (doanh_check) re-review queue for hallucinate segments.

Why a control group exists
--------------------------
The segments worth re-checking are the ones where our band disagrees with the
human label. But re-rating ONLY those makes the resulting overlap meaningless:
every flip there can only raise our score, and the segments we already match are
never revisited, so the number goes up whether or not the original labels were
wrong (flipping all 2468 would read 0.560 -> 0.865).

So the queue mixes two strata and the reviewer cannot tell them apart:

  disagree - our band != the human label (our band is medium or high)
  control  - our band == the human label, sampled to match the disagree stratum
             on embodiment x our band x who settled the label

Flips in `disagree` raise our overlap, flips in `control` lower it. Comparing the
two flip rates separates a real label correction from the selection artefact, and
the per-stratum rates can be reweighted to the full 8093 segments.

Blindness
---------
`webpage/labeling/expert_queue.hallucinate.json` carries only a shuffled list of
segment ids -- no stratum, no model output, no prior scores. The stratum map goes
to `sampling_dataset/data_for_web/expert_review_strata.json`, which is NOT part
of the deployed site, so the labeling UI cannot leak it.

Usage:
    python3 scripts/build_expert_review_queue.py [--control-n 1000] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
VLM_DIR = REPO / "sampling_dataset/data_for_web/hallucinate_bench/300vid/vlm/results"
SITE = REPO / ".main-edit/webpage/labeling"
CATALOG = SITE / "catalog.segment.json"
EXPORT = REPO / "sampling_dataset/data_for_web/hallucinate_final_labels.json"
QUEUE_OUT = SITE / "expert_queue.hallucinate.json"
STRATA_OUT = REPO / "sampling_dataset/data_for_web/expert_review_strata.json"

H2L = {3: "low", 2: "medium", 1: "high"}
EMBODIMENTS = ["bimanual", "humanoid", "single_arm"]


def load() -> pd.DataFrame:
    cat = json.loads(CATALOG.read_text())
    smap = {}
    for it in cat["items"]:
        for s in it["segments"]:
            smap[f"{it['id']}_s{s['index']:03d}"] = (
                f"{it['video_path']}#{s['index']}", it["embodiment"])

    exp = json.loads(EXPORT.read_text())
    rows = []
    for v in exp["videos"]:
        for s in v["segments"]:
            if s["human_score"] is None:
                continue
            sid, emb = smap[s["item_id"]]
            rows.append({
                "item_id": s["item_id"], "segment_id": sid, "video_id": v["id"],
                "embodiment": emb, "human": H2L[s["human_score"]],
                "settled_by": s["consensus"],
            })
    base = pd.DataFrame(rows)

    # our band: the `label` column, i.e. cut(label_ratio, 1.0, 1.5) — identical in
    # every model's csv because it describes the segment, not the model
    frames = [pd.read_csv(VLM_DIR / "gemini-3.6-flash" / f"{e}.csv")[["segment_id", "label"]]
              for e in EMBODIMENTS]
    ours = (pd.concat(frames, ignore_index=True)
            .drop_duplicates("segment_id").rename(columns={"label": "ours"}))
    return base.merge(ours, on="segment_id", how="inner")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control-n", type=int, default=1000,
                    help="control segments to mix in (default 1000)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    w = load()
    disagree = w[w["ours"].isin(["medium", "high"]) & (w["ours"] != w["human"])].copy()
    agree_pool = w[w["ours"] == w["human"]].copy()

    # A control matched CELL-FOR-CELL to the disagree stratum is impossible here:
    # exact proportional matching caps the control at 370 segments, blocked by
    # single_arm/medium/raters (needs 10.2% of the control but only 38 such
    # segments exist). The cells are thin precisely because we are wrong on most
    # medium segments (25% precision), so few correct mediums remain to draw on.
    #
    # So sample the control by simple random draw and carry the cell weights
    # instead: analysis reweights each stratum to the full 8093-segment
    # population, which removes the composition difference without needing the
    # raw samples to look alike.
    keys = ["embodiment", "ours", "settled_by"]
    control = agree_pool.sample(n=min(args.control_n, len(agree_pool)),
                                random_state=args.seed).copy()

    # weight = how many segments of this cell exist per segment sampled from it,
    # so sum(weight) over a stratum reconstructs its true population size
    pop = w.groupby(keys).size().rename("pop")
    for frame, universe in ((disagree, disagree), (control, agree_pool)):
        cell_pop = universe.groupby(keys).size().rename("cell_pop")
        cell_smp = frame.groupby(keys).size().rename("cell_smp")
        j = pd.concat([cell_pop, cell_smp], axis=1).dropna()
        j["weight"] = j["cell_pop"] / j["cell_smp"]
        frame["weight"] = frame.set_index(keys).index.map(j["weight"]).astype(float)

    disagree["stratum"] = "disagree"
    control["stratum"] = "control"
    queue = (pd.concat([disagree, control], ignore_index=True)
             .sample(frac=1.0, random_state=args.seed)
             .reset_index(drop=True))

    # public: ids only, shuffled — nothing that hints at stratum or model output
    QUEUE_OUT.write_text(json.dumps({
        "check": "hallucinate",
        "role": "expert",
        "rater": "doanh_check",
        "version": 1,
        "n": len(queue),
        "note": ("Shuffled expert re-review queue. Stratum labels are deliberately "
                 "absent so the reviewer cannot tell which segments our band got "
                 "wrong; the map lives outside the deployed site."),
        "segments": queue["item_id"].tolist(),
    }, indent=2) + "\n")

    # private: the analysis key
    STRATA_OUT.write_text(json.dumps({
        "note": ("Stratum map for the expert re-review. Kept OUT of webpage/labeling "
                 "so the UI cannot reveal it. `disagree` = our band differs from the "
                 "human label, `control` = our band matches; control is matched to "
                 "disagree on embodiment x our band x settled_by."),
        "control_n_requested": args.control_n,
        "seed": args.seed,
        "counts": {
            "disagree": int((queue["stratum"] == "disagree").sum()),
            "control": int((queue["stratum"] == "control").sum()),
        },
        "strata": {r.item_id: {"stratum": r.stratum, "ours": r.ours,
                               "human_before": r.human, "settled_by": r.settled_by,
                               "embodiment": r.embodiment, "video_id": r.video_id,
                               "weight": round(float(r.weight), 4)}
                   for r in queue.itertuples()},
    }, indent=2) + "\n")

    print(f"disagree : {(queue['stratum'] == 'disagree').sum()}")
    print(f"control  : {(queue['stratum'] == 'control').sum()}")
    print(f"total    : {len(queue)}  over {queue['video_id'].nunique()} videos")
    print("\nraw stratum composition (they need NOT match — analysis reweights):")
    chk = (queue.groupby(["stratum", "embodiment"]).size().unstack(fill_value=0))
    print((100 * chk.div(chk.sum(axis=1), axis=0)).round(1).to_string())
    chk2 = queue.groupby(["stratum", "ours"]).size().unstack(fill_value=0)
    print((100 * chk2.div(chk2.sum(axis=1), axis=0)).round(1).to_string())
    chk3 = queue.groupby(["stratum", "settled_by"]).size().unstack(fill_value=0)
    print((100 * chk3.div(chk3.sum(axis=1), axis=0)).round(1).to_string())
    print(f"\nwrote {QUEUE_OUT.relative_to(REPO)}  (ids only, blind)")
    print(f"wrote {STRATA_OUT.relative_to(REPO)}  (analysis key, off-site)")


if __name__ == "__main__":
    main()
