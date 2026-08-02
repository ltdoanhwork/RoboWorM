"""Batch driver: run the sim2real interaction probe over whole method trees.

Mirrors ``scripts/compute_physics_metrics_ctrlworld.py`` in layout and CLI so the
outputs sit next to the existing ``metrics.json`` / ``physics_metrics.json`` in each
episode dir, and writes a per-category summary.

Because tracking is the expensive part (~100 s/episode) and threshold calibration
plus violation mining both need to re-read the same masks, every episode caches its
``tracks.npz``; a second pass with different thresholds costs seconds.
"""

import os
import sys
import json
import glob
import argparse
import traceback

import numpy as np

_REPO = os.environ.get("REPO")
if _REPO and _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from metrics.sim2real.interaction_probe import (  # noqa: E402
    run_episode, _json_safe,
)
from metrics.sim2real.track_backend import Sim2RealTracker  # noqa: E402
from metrics.sim2real.lift_backend import build_lift  # noqa: E402

CATEGORIES = ("makovian", "non_makovian")


def find_episodes(output_root, input_root, categories=CATEGORIES):
    """Pair each prediction dir with its GT strip + metadata.

    Ctrl-World writes the 3-view strip as ``{pred,gt}_all_views.mp4`` inside the
    episode dir; ``full_pred.mp4`` is only the primary view, which would throw away
    the second exterior camera the contact AND-gate depends on. Metadata comes from
    the shared dense input tree (sampling-dataset-layout rule 3).
    """
    eps = []
    for cat in categories:
        for pred_dir in sorted(glob.glob(os.path.join(output_root, cat, "episode_*"))):
            ep = os.path.basename(pred_dir)
            pred = os.path.join(pred_dir, "pred_all_views.mp4")
            gt = os.path.join(pred_dir, "gt_all_views.mp4")
            meta = os.path.join(input_root, cat, ep, "metadata.json")
            if not os.path.exists(meta):
                meta = os.path.join(pred_dir, "metadata.json")
            missing = [p for p in (pred, gt, meta) if not os.path.exists(p)]
            if missing:
                continue
            eps.append({"category": cat, "episode": ep, "dir": pred_dir,
                        "pred": pred, "gt": gt, "metadata": meta})
    return eps


SUMMARY_KEYS = [
    ("score", lambda m: m["sim2real_interaction_score"]),
    ("valid", lambda m: float(bool(m.get("valid", True)))),
    ("levitation_excess", lambda m: m["violations"].get("levitation_rate_excess")),
    ("levitation_rate", lambda m: m["violations"].get("levitation_rate")),
    ("levitation_rate_gt", lambda m: m["violations"].get("levitation_rate_gt")),
    ("grasp_follow_excess", lambda m: m["violations"].get("grasp_follow_ratio_excess")),
    ("object_present_deficit", lambda m: m["violations"].get("object_present_deficit")),
    ("object_shape_excess", lambda m: m["violations"].get("object_shape_excess")),
    ("penetration_excess", lambda m: m["violations"].get("penetration_excess")),
    ("chatter_excess", lambda m: m["violations"].get("contact_chatter_excess")),
    ("onset_err_frames", lambda m: m["agreement"].get("contact_onset_err_frames")),
    ("contact_tiou", lambda m: m["agreement"].get("contact_temporal_iou")),
    ("obj_traj_err", lambda m: m["agreement"].get("obj_traj_err")),
    ("gap_curve_err", lambda m: m["agreement"].get("gap_curve_err")),
    ("step_pass_rate", lambda m: m["step_level"]["step_pass_rate"]),
    ("task_success", lambda m: float(bool(m["step_level"]["task_success"]))),
]


def row_from_metrics(ep, m):
    row = {"category": ep["category"], "episode": ep["episode"],
           "instruction": m.get("instruction"), "num_frames": m.get("num_frames"),
           "invalid_reasons": m.get("validity", {}).get("invalid_reasons", [])}
    for name, fn in SUMMARY_KEYS:
        try:
            v = fn(m)
        except Exception:
            v = None
        row[name] = None if v is None or (
            isinstance(v, float) and not np.isfinite(v)) else v
    return row


def summarize(rows):
    """Aggregate, excluding episodes the metric declared unmeasurable.

    Averaging over failed tracks would let tracker failures set the benchmark
    number, so ``valid`` is reported as a rate over ALL episodes while every other
    term is averaged over valid ones only.
    """
    valid = [r for r in rows if r.get("valid")]
    out = {"num_episodes": len(rows), "num_valid": len(valid),
           "valid_rate": float(len(valid) / len(rows)) if rows else 0.0}
    for name, _ in SUMMARY_KEYS:
        if name == "valid":
            continue
        vals = [r[name] for r in valid if r.get(name) is not None]
        if vals:
            out[f"mean_{name}"] = float(np.mean(vals))
            out[f"median_{name}"] = float(np.median(vals))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", required=True,
                    help="e.g. sampling_dataset/dense/single_arm/output/multiview/ctrlworld")
    ap.add_argument("--input_root", required=True,
                    help="e.g. sampling_dataset/dense/single_arm/input/multiview/ctrlworld")
    ap.add_argument("--categories", nargs="*", default=list(CATEGORIES))
    ap.add_argument("--sam2_cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    ap.add_argument("--sam2_ckpt", required=True)
    ap.add_argument("--gdino_cfg", required=True)
    ap.add_argument("--gdino_ckpt", required=True)
    ap.add_argument("--depth", default="none", choices=["none", "dav2", "moge"])
    ap.add_argument("--upscale", type=int, default=3)
    ap.add_argument("--box_thr", type=float, default=0.25)
    ap.add_argument("--tau_gap", type=float, default=None)
    ap.add_argument("--eps_move", type=float, default=None)
    ap.add_argument("--max_episodes", type=int, default=None)
    ap.add_argument("--work_root", default=None,
                    help="where tracks.npz + frames live (default: <episode dir>/sim2real)")
    ap.add_argument("--viz", action="store_true", help="render panel.png per episode")
    ap.add_argument("--force", action="store_true", help="recompute even if metrics exist")
    args = ap.parse_args()

    eps = find_episodes(args.output_root, args.input_root, args.categories)
    if args.max_episodes:
        eps = eps[: args.max_episodes]
    if not eps:
        print(f"[batch] no episodes under {args.output_root}")
        return
    print(f"[batch] {len(eps)} episodes")

    tracker = Sim2RealTracker(args.sam2_cfg, args.sam2_ckpt, args.gdino_cfg,
                              args.gdino_ckpt, device="cuda")
    lift = build_lift(args.depth)
    params = {"tau_gap": args.tau_gap, "eps_move": args.eps_move}

    rows = []
    for i, ep in enumerate(eps, 1):
        out_dir = (os.path.join(args.work_root, ep["category"], ep["episode"])
                   if args.work_root else os.path.join(ep["dir"], "sim2real"))
        done = os.path.join(out_dir, "sim2real_metrics.json")
        if os.path.exists(done) and not args.force:
            rows.append(row_from_metrics(ep, json.load(open(done))))
            print(f"[{i}/{len(eps)}] {ep['category']}/{ep['episode']} cached")
            continue
        print(f"[{i}/{len(eps)}] {ep['category']}/{ep['episode']}")
        try:
            m = run_episode(tracker, ep["gt"], ep["pred"], ep["metadata"], out_dir,
                            upscale=args.upscale, params=params, lift=lift,
                            make_viz=args.viz, box_thr=args.box_thr)
            rows.append(row_from_metrics(ep, m))
            sc = m["sim2real_interaction_score"]
            print(f"     score={'INVALID' if sc is None else f'{sc:.3f}'} "
                  f"levit_exc={m['violations'].get('levitation_rate_excess')} "
                  f"steps={m['step_level']['num_passed']}/{m['step_level']['num_nodes']}"
                  + ("" if m.get("valid", True)
                     else "  <- " + ",".join(m["validity"]["invalid_reasons"])))
        except Exception:
            traceback.print_exc()
            print(f"     FAILED {ep['episode']}")

    per_cat = {}
    for cat in args.categories:
        sub = [r for r in rows if r["category"] == cat]
        if sub:
            per_cat[cat] = summarize(sub)
    summary = {"overall": summarize(rows), "per_category": per_cat,
               "episodes": rows, "params": params, "depth": args.depth,
               "output_root": args.output_root}

    dest = os.path.join(args.work_root or args.output_root, "sim2real_summary.json")
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    with open(dest, "w") as f:
        json.dump(_json_safe(summary), f, indent=2)

    o = summary["overall"]
    print(f"\n=== {len(rows)} episodes, {o['num_valid']} valid "
          f"({o['valid_rate'] * 100:.1f}%) ===")
    for k in ("score", "levitation_excess", "grasp_follow_excess", "obj_traj_err",
              "contact_tiou", "onset_err_frames", "step_pass_rate", "task_success"):
        if f"mean_{k}" in o:
            print(f"  mean {k:24s} {o[f'mean_{k}']:.4f}")
    bad = [r for r in rows if not r.get("valid")]
    if bad:
        print(f"  {len(bad)} unmeasurable episodes (tracking failure), e.g.:")
        for r in bad[:5]:
            print(f"    {r['category']}/{r['episode']}  {','.join(r['invalid_reasons'])}")
    print(f"wrote -> {dest}")


if __name__ == "__main__":
    main()
