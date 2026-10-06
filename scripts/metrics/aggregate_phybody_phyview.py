#!/usr/bin/env python3
"""Aggregate PhyBody + PhyView from existing labeled catalogs; sanity vs human scores."""
from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "sampling_dataset/data_for_web/data_for_web.csv"
HALLU = REPO / "sampling_dataset/data_for_web/hallucinate_bench/300-prune"
OUT = REPO / "metrics/phyworld/tables/body_view_sanity.json"


def auc_hi_vs_lo(scores_by_label: dict[int, list[float]], hi=3, lo=1) -> float:
    a = scores_by_label.get(hi, [])
    b = scores_by_label.get(lo, [])
    if not a or not b:
        return float("nan")
    return sum((x > y) + 0.5 * (x == y) for x in a for y in b) / (len(a) * len(b))


def phy_view(g: float, p: float) -> float:
    return 100.0 * (g + p) / 2.0


def main() -> None:
    # --- PhyView ---
    rows = list(csv.DictReader(CSV.open()))
    by_label: dict[int, list[dict]] = defaultdict(list)
    by_robot: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        lab = int(float(r["score"]))
        item = {
            "label": lab,
            "view_geometry": float(r["geometric_consistency"]),
            "view_pose_stability": float(r["pose_stability"]),
            "view_disjoint_rate": float(r["disjoint_frame_ratio"]),
            "phy_view": phy_view(float(r["geometric_consistency"]), float(r["pose_stability"])),
            "robot": r["robot_type"],
            "path": r["video_path"],
        }
        by_label[lab].append(item)
        by_robot[r["robot_type"]].append(item)

    view = {
        "n": len(rows),
        "by_label": {
            str(lab): {
                "n": len(items),
                "phy_view_mean": st.mean(i["phy_view"] for i in items),
                "geometry_mean": st.mean(i["view_geometry"] for i in items),
                "pose_mean": st.mean(i["view_pose_stability"] for i in items),
                "disjoint_mean": st.mean(i["view_disjoint_rate"] for i in items),
            }
            for lab, items in sorted(by_label.items())
        },
        "auc_phy_view_3vs1": auc_hi_vs_lo(
            {lab: [i["phy_view"] for i in items] for lab, items in by_label.items()}
        ),
        "auc_geometry": auc_hi_vs_lo(
            {lab: [i["view_geometry"] for i in items] for lab, items in by_label.items()}
        ),
        "auc_pose": auc_hi_vs_lo(
            {lab: [i["view_pose_stability"] for i in items] for lab, items in by_label.items()}
        ),
        "auc_disjoint_inv": auc_hi_vs_lo(
            {lab: [-i["view_disjoint_rate"] for i in items] for lab, items in by_label.items()}
        ),
        "by_robot": {
            rob: {
                "n": len(items),
                "phy_view_mean": st.mean(i["phy_view"] for i in items),
                "label_mean": st.mean(i["label"] for i in items),
            }
            for rob, items in sorted(by_robot.items())
        },
        # ranking: label1 should have lowest phy_view
        "rank_ok": (
            st.mean(i["phy_view"] for i in by_label[1])
            < st.mean(i["phy_view"] for i in by_label[2])
            < st.mean(i["phy_view"] for i in by_label[3])
        ),
    }

    # --- PhyBody ---
    body = {}
    for robot in ("single_arm", "bimanual", "humanoid"):
        d = json.loads((HALLU / robot / "segments.json").read_text())
        ratios_r, ratios_j, viol_rates = [], [], []
        per_clip = []
        for vid, v in d["videos"].items():
            segs = v["segments"]
            n_viol = sum(
                1
                for s in segs
                if s.get("rigidity", {}).get("violated") or s.get("jerk", {}).get("violated")
            )
            rr = [s["rigidity"]["ratio"] for s in segs if "rigidity" in s]
            jr = [s["jerk"]["ratio"] for s in segs if "jerk" in s]
            viol_rate = n_viol / len(segs) if segs else float("nan")
            # phy_body-ish: invert ratios (2.0 -> 0, 0 -> 100)
            def inv(x):
                return max(0.0, min(100.0, (1.0 - x / 2.0) * 100.0))

            pb = 0.5 * (inv(st.median(rr)) + inv(st.median(jr))) if rr and jr else float("nan")
            ratios_r.extend(rr)
            ratios_j.extend(jr)
            viol_rates.append(viol_rate)
            per_clip.append(
                {
                    "vid": vid,
                    "source": v.get("source"),
                    "n_seg": len(segs),
                    "n_violated": v.get("n_violated", n_viol),
                    "viol_rate": viol_rate,
                    "rigidity_ratio_med": st.median(rr) if rr else None,
                    "jerk_ratio_med": st.median(jr) if jr else None,
                    "phy_body": pb,
                }
            )
        # split clips by whether any segment violated
        clean = [c for c in per_clip if c["n_violated"] == 0]
        dirty = [c for c in per_clip if c["n_violated"] > 0]
        body[robot] = {
            "n_clips": len(per_clip),
            "n_segments": sum(c["n_seg"] for c in per_clip),
            "n_clips_any_violation": len(dirty),
            "seg_violation_rate_mean": st.mean(viol_rates),
            "rigidity_ratio_median": st.median(ratios_r),
            "jerk_ratio_median": st.median(ratios_j),
            "phy_body_clean_mean": st.mean(c["phy_body"] for c in clean) if clean else None,
            "phy_body_dirty_mean": st.mean(c["phy_body"] for c in dirty) if dirty else None,
            "worst5": sorted(per_clip, key=lambda c: c["phy_body"])[:5],
            "best5": sorted(per_clip, key=lambda c: -c["phy_body"])[:5],
            "separation_ok": (
                clean
                and dirty
                and st.mean(c["phy_body"] for c in clean)
                > st.mean(c["phy_body"] for c in dirty)
            ),
        }

    out = {"phy_view": view, "phy_body": body}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2)[:4000])
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
