"""Sim2real interaction probe: one episode, end to end.

Tracks object/gripper/target in the GT and predicted clips, evaluates the physics
predicates, and renders the visual evidence. Called by
``scripts/run_sim2real_interaction.sh`` (which sets up the vendored sam2 /
groundingdino paths) and reused by the batch driver and the mining tool.
"""

import os
import sys
import json
import argparse

import numpy as np

_REPO = os.environ.get("REPO")
if _REPO and _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from metrics.sim2real.track_backend import (  # noqa: E402
    CTRLWORLD_STATIC_VIEWS, Sim2RealTracker, track_episode, cleanup_work,
    save_tracks, load_tracks, rebuild_frames,
)
from metrics.sim2real.interaction_metrics import (  # noqa: E402
    DEFAULTS, episode_metrics, view_metrics,
)
from metrics.sim2real import interaction_viz  # noqa: E402


def _json_safe(obj):
    """numpy -> python, NaN -> None, so json.dump never emits invalid JSON."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if not np.isfinite(f) else f
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    return obj


def analyse_tracks(tracks, fps=5.0, params=None, lift=None,
                   depths_pred=None, depths_gt=None, conditioning="action"):
    """Tracker output -> metric dict (no I/O). Shared by probe / batch / mining."""
    p = dict(DEFAULTS)
    if params:
        p.update({k: v for k, v in params.items() if v is not None})

    views = []
    for i, vw in enumerate(tracks["views"]):
        views.append(view_metrics(
            vw["pred"], vw["gt"], gap_pct=p["gap_pct"], tau_gap=p["tau_gap"],
            eps_move=p["eps_move"], lift=lift,
            depths_pred=None if depths_pred is None else depths_pred[i],
            depths_gt=None if depths_gt is None else depths_gt[i]))

    static = [i for i, vw in enumerate(tracks["views"]) if vw.get("is_static", True)]
    if not static:
        # every camera moves (e.g. GR1 "unlocked waist" ego view): apparent object
        # motion is dominated by ego motion, so levitation is not measurable. Fall
        # back to the ctrlworld static pair only when that layout is actually present.
        static = ([i for i in CTRLWORLD_STATIC_VIEWS if i < len(views)]
                  or list(range(len(views))))
    looks = [vw["referent_pick"]["look"] for vw in tracks["views"]
             if vw.get("referent_pick") and vw["referent_pick"].get("look") is not None]
    m = episode_metrics(views, static, fps=fps, params=p,
                        referent_look=(max(looks) if looks else None),
                        conditioning=conditioning)
    m["num_static_views"] = len(static)
    return m


def run_episode(tracker, gt_video, pred_video, metadata_path, out_dir,
                upscale=3, max_frames=None, params=None, lift=None,
                make_viz=True, keep_work=False, box_thr=0.25, cache_tracks=True,
                num_views=3, gripper_mode="bgsub", view_names=None,
                static_views=None, select_referent=False, conditioning="action"):
    os.makedirs(out_dir, exist_ok=True)
    meta = json.load(open(metadata_path))
    fps = float(meta.get("fps", 5))

    work = os.path.join(out_dir, "_work")
    cache = os.path.join(out_dir, "tracks.npz")
    if cache_tracks and os.path.exists(cache):
        tracks = load_tracks(cache)
        print(f"[probe] reusing cached tracks {cache}")
        if make_viz or (lift is not None and getattr(lift, "has_depth", False)):
            # only the masks are cached; pixels are regenerated on demand
            rebuild_frames(tracks, gt_video, pred_video, work)
    else:
        tracks = track_episode(tracker, gt_video, pred_video, work, meta["instruction"],
                               num_views=num_views, upscale=upscale,
                               max_frames=max_frames, box_thr=box_thr,
                               gripper_mode=gripper_mode, view_names=view_names,
                               static_views=static_views,
                               select_referent=select_referent)
        if cache_tracks:
            save_tracks(tracks, cache)

    depths_pred = depths_gt = None
    if lift is not None and getattr(lift, "has_depth", False):
        import cv2
        depths_pred, depths_gt = [], []
        for vw in tracks["views"]:
            for tag, sink in (("pred", depths_pred), ("gt", depths_gt)):
                d = []
                n = len(vw[tag]["object"])
                for t in range(n):
                    img = cv2.cvtColor(
                        cv2.imread(os.path.join(vw["frames_dir"][tag], f"{t:05d}.jpg")),
                        cv2.COLOR_BGR2RGB)
                    d.append(lift.frame_depth(img))
                sink.append(d)

    metrics = analyse_tracks(tracks, fps=fps, params=params, lift=lift,
                             depths_pred=depths_pred, depths_gt=depths_gt,
                             conditioning=conditioning)
    metrics["episode_id"] = meta.get("episode_id")
    metrics["instruction"] = meta.get("instruction")
    metrics["object_prompt"] = tracks["object_prompt"]
    metrics["target_prompt"] = tracks["target_prompt"]
    metrics["num_frames"] = tracks["num_frames"]
    metrics["view_names"] = tracks["view_names"]
    metrics["depth_backend"] = getattr(lift, "name", "none")

    if make_viz:
        main = tracks["views"][metrics["_main_view"]]
        interaction_viz.episode_panel(
            main, metrics, os.path.join(out_dir, "panel.png"))
        interaction_viz.violation_clip(
            main, metrics, os.path.join(out_dir, "violation_clip.mp4"), fps=int(round(fps)))

    with open(os.path.join(out_dir, "sim2real_metrics.json"), "w") as f:
        json.dump(_json_safe({k: v for k, v in metrics.items()
                              if not k.startswith("_")}), f, indent=2)

    if not keep_work:
        cleanup_work(work)
    return metrics


def print_summary(m):
    v, a, s = m["violations"], m["agreement"], m["step_level"]
    print("\n=== sim2real interaction ===")
    sc = m["sim2real_interaction_score"]
    if not m.get("valid", True):
        print(f"score                     : INVALID "
              f"({', '.join(m['validity']['invalid_reasons'])})")
        print(f"  GT object present frac  : {m['validity']['object_present_frac_gt']:.3f}"
              f"  scored weight {m['validity']['scored_weight_frac']:.2f}")
    else:
        print(f"score                     : {sc:.4f}  "
              f"[charged: {'+'.join(m.get('charged_groups', []))}"
              f", conditioning={m.get('conditioning')}]")
    ps, ags = m.get("physics_score"), m.get("agreement_score")
    print(f"  physics / agreement     : "
          f"{'None' if ps is None else format(ps, '.4f')} / "
          f"{'None' if ags is None else format(ags, '.4f')}"
          + ("   (agreement reported only - model is text-conditioned)"
             if "agreement" not in m.get("charged_groups", []) else ""))
    print(f"levitation_rate           : {v.get('levitation_rate')}  "
          f"(GT {v.get('levitation_rate_gt')}, excess {v.get('levitation_rate_excess')})")
    print(f"grasp_follow ratio / cos  : {v.get('grasp_follow_ratio')} / "
          f"{v.get('grasp_follow_cos')}  (GT {v.get('grasp_follow_ratio_gt')} / "
          f"{v.get('grasp_follow_cos_gt')})")
    print(f"object present frac       : {v.get('object_present_frac')}  "
          f"(deficit {v.get('object_present_deficit')})")
    print(f"area/aspect CV excess     : {v.get('object_area_cv_excess')} / "
          f"{v.get('object_aspect_cv_excess')}")
    print(f"contact transitions       : {v.get('contact_transitions')} "
          f"(GT {v.get('contact_transitions_gt')}, excess {v.get('contact_chatter_excess')})")
    print(f"penetration excess        : {v.get('penetration_excess')}")
    print(f"contact onset pred/GT     : {a.get('contact_onset_pred')} / "
          f"{a.get('contact_onset_gt')}  err {a.get('contact_onset_err_frames')} frames")
    print(f"contact tIoU / F1         : {a.get('contact_temporal_iou')} / {a.get('contact_f1')}")
    print(f"obj traj err (diam)       : {a.get('obj_traj_err')}  "
          f"travel pred/GT {a.get('obj_travel_pred')}/{a.get('obj_travel_gt')}")
    print(f"gap curve err             : {a.get('gap_curve_err')}")
    print(f"step nodes passed         : {s['num_passed']}/{s['num_nodes']} "
          f"(task_success={s['task_success']})")
    for name, r in s["nodes"].items():
        print(f"    {name:10s} pass={str(r.get('pass')):5s} "
              + " ".join(f"{k}={r[k]}" for k in r if k != "pass"))
    print("score terms:")
    for name, d in m["score_terms"].items():
        raw = "None" if d["raw"] is None else f"{d['raw']:.4f}"
        nrm = "None" if d["norm"] is None else f"{d['norm']:.3f}"
        tag = d.get("group", "")
        if not d.get("charged", True):
            tag += " (not charged)"
        print(f"    {name:16s} raw={raw:>9s} norm={nrm:>6s} w={d['weight']}  {tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sam2_cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    ap.add_argument("--sam2_ckpt", required=True)
    ap.add_argument("--gdino_cfg", required=True)
    ap.add_argument("--gdino_ckpt", required=True)
    ap.add_argument("--depth", default="none", choices=["none", "dav2", "moge"])
    ap.add_argument("--upscale", type=int, default=3)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--box_thr", type=float, default=0.25)
    ap.add_argument("--num_views", type=int, default=3,
                    help="views side-by-side in the mp4 strip (1 for singleview)")
    ap.add_argument("--gripper_mode", default="bgsub", choices=["bgsub", "gdino"],
                    help="bgsub: static cam, arm absent at t=0 (ctrlworld exterior); "
                         "gdino: arm already in frame at t=0 (GR1 ego view)")
    ap.add_argument("--view_names", default=None,
                    help="comma-separated view names (default: ctrlworld layout)")
    ap.add_argument("--static_views", default=None,
                    help="comma-separated indices of static cameras (default: 0,1)")
    ap.add_argument("--select_referent", action="store_true",
                    help="resolve which same-category candidate the instruction "
                         "means by tracked motion, not text confidence (needed on "
                         "cluttered tables e.g. GR1 humanoid)")
    ap.add_argument("--conditioning", default="action", choices=["action", "text"],
                    help="what the model was given. 'action': per-frame action/state "
                         "(Ctrl-World) - GT is the expected answer, so timing "
                         "agreement is charged. 'text': one frame + a sentence "
                         "(DreamGen video2world_lvg) - the model cannot know GT's "
                         "timing, so only physics terms are charged and agreement "
                         "is reported for information.")
    ap.add_argument("--tau_gap", type=float, default=None)
    ap.add_argument("--eps_move", type=float, default=None)
    ap.add_argument("--gap_pct", type=float, default=None)
    ap.add_argument("--no_viz", action="store_true")
    ap.add_argument("--keep_work", action="store_true")
    args = ap.parse_args()

    from metrics.sim2real.lift_backend import build_lift

    tracker = Sim2RealTracker(args.sam2_cfg, args.sam2_ckpt, args.gdino_cfg,
                              args.gdino_ckpt, device="cuda")
    lift = build_lift(args.depth)
    params = {"tau_gap": args.tau_gap, "eps_move": args.eps_move, "gap_pct": args.gap_pct}

    m = run_episode(tracker, args.gt, args.pred, args.metadata, args.out,
                    upscale=args.upscale, max_frames=args.max_frames, params=params,
                    lift=lift, make_viz=not args.no_viz, keep_work=args.keep_work,
                    box_thr=args.box_thr, num_views=args.num_views,
                    gripper_mode=args.gripper_mode,
                    view_names=(args.view_names.split(",") if args.view_names else None),
                    static_views=([int(x) for x in args.static_views.split(",")]
                                  if args.static_views else None),
                    select_referent=args.select_referent,
                    conditioning=args.conditioning)
    print_summary(m)
    print(f"\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
