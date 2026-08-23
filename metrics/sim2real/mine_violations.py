"""Mine real episodes for physics violations and produce the visual evidence.

The negative controls prove each term *can* fire; this proves the terms fire on
REAL generated video and that a human agrees with what they flagged. It reads the
per-episode ``sim2real_metrics.json`` written by ``compute_sim2real_batch.py``,
ranks episodes by each violation term, and re-renders panels + clips for the
worst and cleanest few - so every claim in the ranking can be checked by eye.

Re-rendering is cheap because the masks are cached in ``tracks.npz``; no GPU and
no tracker involved.

Usage:

    python -m metrics.sim2real.mine_violations \
        --work_root /tmp/s2r_batch/dense --out /tmp/s2r_mined --top_k 5
"""

import os
import sys
import json
import glob
import shutil
import argparse

import numpy as np

_REPO = os.environ.get("REPO")
if _REPO and _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from metrics.sim2real.track_backend import load_tracks, rebuild_frames  # noqa: E402
from metrics.sim2real.interaction_probe import analyse_tracks, _json_safe  # noqa: E402
from metrics.sim2real import interaction_viz  # noqa: E402

# term -> (metric path, direction). "high" = a larger value is MORE violating.
RANK_TERMS = {
    "levitation": (("violations", "levitation_rate_excess"), "high"),
    "grasp_follow": (("violations", "grasp_follow_ratio_excess"), "high"),
    "object_presence": (("violations", "object_present_deficit"), "high"),
    "object_shape": (("violations", "object_shape_excess"), "high"),
    "penetration": (("violations", "penetration_excess"), "high"),
    "chatter": (("violations", "contact_chatter_excess"), "high"),
    "contact_timing": (("agreement", "contact_onset_err_frames"), "high"),
    "contact_overlap": (("agreement", "contact_temporal_iou"), "low"),
    "trajectory": (("agreement", "obj_traj_err"), "high"),
    "gap_curve": (("agreement", "gap_curve_err"), "high"),
    "score": (("sim2real_interaction_score",), "low"),
}


def _get(d, path):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def collect(work_root):
    """Load every sim2real_metrics.json under a work root."""
    recs = []
    for f in sorted(glob.glob(os.path.join(work_root, "*", "*", "sim2real_metrics.json"))):
        ep_dir = os.path.dirname(f)
        try:
            m = json.load(open(f))
        except Exception:
            continue
        recs.append({
            "category": os.path.basename(os.path.dirname(ep_dir)),
            "episode": os.path.basename(ep_dir),
            "dir": ep_dir,
            "tracks": os.path.join(ep_dir, "tracks.npz"),
            "metrics": m,
        })
    return recs


def rank(recs, term, top_k, worst=True):
    path, direction = RANK_TERMS[term]
    scored = []
    for r in recs:
        # never rank on an episode the metric itself declared unmeasurable: a failed
        # track leaves most terms at their default and would top the "cleanest" list
        if not r["metrics"].get("valid", True):
            continue
        v = _get(r["metrics"], path)
        if v is None or not np.isfinite(float(v)):
            continue
        scored.append((float(v), r))
    if not scored:
        return []
    # "violating" means large for direction=high, small for direction=low
    reverse = (direction == "high") if worst else (direction == "low")
    scored.sort(key=lambda t: t[0], reverse=reverse)
    return scored[:top_k]


def render(rec, out_dir, tag, fps=5.0, clip=True, conditioning=None):
    """Re-render panel (+ clip) for one episode from its cached masks."""
    if not os.path.exists(rec["tracks"]):
        return None
    tracks = load_tracks(rec["tracks"])
    if conditioning is None:
        conditioning = rec.get("metrics", {}).get("conditioning") or "action"
    m = analyse_tracks(tracks, fps=fps, conditioning=conditioning)
    work = os.path.join(out_dir, "_frames", rec["episode"])
    try:
        rebuild_frames(tracks, tracks["gt_video"], tracks["pred_video"], work)
    except Exception as e:
        print(f"      [mine] cannot rebuild frames for {rec['episode']}: {e}")
        return None
    vw = tracks["views"][m["_main_view"]]
    os.makedirs(out_dir, exist_ok=True)
    base = f"{tag}_{rec['category']}_{rec['episode']}"
    png = interaction_viz.episode_panel(vw, m, os.path.join(out_dir, base + ".png"))
    if clip:
        interaction_viz.violation_clip(
            vw, m, os.path.join(out_dir, base + ".mp4"), fps=int(round(fps)))
    shutil.rmtree(work, ignore_errors=True)
    return png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_root", required=True,
                    help="dir passed as WORK_ROOT to run_sim2real_interaction_batch.sh")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--terms", nargs="*", default=None,
                    help="subset of %s" % ",".join(RANK_TERMS))
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--no_render", action="store_true", help="ranking table only")
    ap.add_argument("--no_clip", action="store_true")
    args = ap.parse_args()

    recs = collect(args.work_root)
    if not recs:
        print(f"[mine] nothing under {args.work_root}")
        return
    n_valid = sum(1 for r in recs if r["metrics"].get("valid", True))
    print(f"[mine] {len(recs)} episodes, {n_valid} measurable "
          f"({len(recs) - n_valid} excluded for tracking failure)")
    terms = args.terms or list(RANK_TERMS)

    report = {"work_root": args.work_root, "num_episodes": len(recs),
              "num_valid": n_valid, "terms": {}}
    for term in terms:
        worst = rank(recs, term, args.top_k, worst=True)
        best = rank(recs, term, args.top_k, worst=False)
        path, direction = RANK_TERMS[term]
        vals = [v for v, _ in rank(recs, term, len(recs), worst=True)]
        print(f"\n=== {term}  ({'.'.join(path)}, violating = "
              f"{'high' if direction == 'high' else 'low'}) ===")
        if vals:
            print(f"    n={len(vals)} min={min(vals):.4f} median={np.median(vals):.4f} "
                  f"max={max(vals):.4f}")
        print("    WORST:")
        for v, r in worst:
            print(f"      {v:>9.4f}  {r['category']}/{r['episode']}")
        print("    CLEANEST:")
        for v, r in best:
            print(f"      {v:>9.4f}  {r['category']}/{r['episode']}")

        entry = {"stats": {"n": len(vals),
                           "min": float(min(vals)) if vals else None,
                           "median": float(np.median(vals)) if vals else None,
                           "max": float(max(vals)) if vals else None},
                 "worst": [{"value": v, "category": r["category"],
                            "episode": r["episode"]} for v, r in worst],
                 "cleanest": [{"value": v, "category": r["category"],
                               "episode": r["episode"]} for v, r in best]}
        if not args.no_render:
            d = os.path.join(args.out, term)
            for v, r in worst:
                p = render(r, d, f"worst{v:.3f}", fps=args.fps, clip=not args.no_clip)
                if p:
                    print(f"      -> {p}")
            for v, r in best:
                render(r, d, f"clean{v:.3f}", fps=args.fps, clip=False)
        report["terms"][term] = entry

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "mining_report.json"), "w") as f:
        json.dump(_json_safe(report), f, indent=2)
    print(f"\nwrote -> {args.out}")
    _ = shutil


if __name__ == "__main__":
    main()
