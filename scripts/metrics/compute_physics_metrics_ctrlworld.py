"""Run Tier-1 physics metrics (flow-vs-GT) over ctrlworld multiview episodes.

For each episode it reads ``pred_all_views.mp4`` (predicted views tiled
horizontally) and ``gt_all_views.mp4`` (GT views tiled), splits them into the
individual camera views, and computes optical-flow-vs-GT motion metrics per view
via SEA-RAFT. Results are written to ``physics_metrics.json`` per episode and a
``physics_summary.json`` per (method, split), plus a combined comparison JSON.

Example:
    CUDA_VISIBLE_DEVICES=2 python scripts/metrics/compute_physics_metrics_ctrlworld.py \
        --roots dense worldcache --split makovian --max-episodes 15
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import imageio.v3 as iio

# repo root on path for `metrics` package import
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metrics.dynamics.flow_backend import SeaRaftFlow
from metrics.dynamics.physics_metrics import flow_vs_gt_metrics, aggregate_view_metrics

CTRLWORLD_SUBPATH = "single_arm/output/multiview/ctrlworld"


def load_video(path: Path) -> np.ndarray:
    """Decode an mp4 to (T, H, W, 3) uint8 RGB."""
    frames = iio.imread(path, index=None)
    return np.asarray(frames)


def split_views(strip: np.ndarray, num_views: int) -> list[np.ndarray]:
    """Split a horizontally-tiled (T, H, W*num_views, 3) strip into views."""
    T, H, W, C = strip.shape
    assert W % num_views == 0, f"width {W} not divisible by num_views {num_views}"
    vw = W // num_views
    return [strip[:, :, v * vw : (v + 1) * vw, :] for v in range(num_views)]


def matched_episodes(roots: list[Path], split: str) -> list[str]:
    """Episodes present (with required mp4s) in ALL roots, sorted by index."""
    per_root_sets = []
    for r in roots:
        d = r / CTRLWORLD_SUBPATH / split
        eps = set()
        if d.is_dir():
            for ep in d.iterdir():
                if ep.is_dir() and ep.name.startswith("episode_"):
                    if (ep / "pred_all_views.mp4").exists() and (ep / "gt_all_views.mp4").exists():
                        eps.add(ep.name)
        per_root_sets.append(eps)
    common = set.intersection(*per_root_sets) if per_root_sets else set()
    return sorted(common)


def process_episode(ep_dir: Path, flow_model, num_views: int, overwrite: bool) -> dict:
    out_path = ep_dir / "physics_metrics.json"
    if out_path.exists() and not overwrite:
        with open(out_path) as f:
            return json.load(f)

    pred = load_video(ep_dir / "pred_all_views.mp4")
    gt = load_video(ep_dir / "gt_all_views.mp4")

    # align frame counts (safety; ctrlworld pairs are already equal)
    T = min(pred.shape[0], gt.shape[0])
    pred, gt = pred[:T], gt[:T]

    pred_views = split_views(pred, num_views)
    gt_views = split_views(gt, num_views)

    per_view = {}
    for v, (pv, gv) in enumerate(zip(pred_views, gt_views)):
        per_view[f"view_{v}"] = flow_vs_gt_metrics(pv, gv, flow_model)

    agg = aggregate_view_metrics(per_view)
    result = {**agg, "per_view": per_view, "num_frames": int(T), "num_views": num_views}

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def summarize(results: dict[str, dict]) -> dict:
    keys = ["flow_epe", "motion_mag_pred", "motion_mag_gt", "motion_mag_abs_err", "motion_mag_ratio"]
    summary = {"num_episodes": len(results)}
    for k in keys:
        vals = [r[k] for r in results.values() if k in r]
        if vals:
            summary[f"mean_{k}"] = float(np.mean(vals))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=str(_REPO_ROOT / "sampling_dataset"))
    ap.add_argument("--roots", nargs="+", default=["dense", "worldcache"],
                    help="method dirs under sampling_dataset to evaluate")
    ap.add_argument("--split", default="makovian")
    ap.add_argument("--num-views", type=int, default=3)
    ap.add_argument("--max-episodes", type=int, default=15)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out", default=str(_REPO_ROOT / "sampling_dataset/physics_flow_trial.json"))
    args = ap.parse_args()

    ds_root = Path(args.dataset_root)
    roots = [ds_root / r for r in args.roots]

    episodes = matched_episodes(roots, args.split)
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]
    print(f"[info] {len(episodes)} matched episodes across {args.roots} (split={args.split})")
    print(f"[info] episodes: {episodes}")

    flow_model = SeaRaftFlow(device="cuda")

    all_out = {}
    for method, root in zip(args.roots, roots):
        split_dir = root / CTRLWORLD_SUBPATH / args.split
        results = {}
        for i, ep in enumerate(episodes):
            ep_dir = split_dir / ep
            print(f"[{method}] ({i+1}/{len(episodes)}) {ep} ...", flush=True)
            try:
                r = process_episode(ep_dir, flow_model, args.num_views, args.overwrite)
                results[ep] = r
                print(
                    f"    epe={r.get('flow_epe'):.3f} "
                    f"mag_pred={r.get('motion_mag_pred'):.3f} mag_gt={r.get('motion_mag_gt'):.3f} "
                    f"ratio={r.get('motion_mag_ratio'):.3f} mag_abs_err={r.get('motion_mag_abs_err'):.3f}",
                    flush=True,
                )
            except Exception as e:  # keep going on a bad episode
                print(f"    ERROR: {e!r}", flush=True)

        summ = summarize(results)
        with open(split_dir / "physics_summary.json", "w") as f:
            json.dump(summ, f, indent=2)
        all_out[method] = {"summary": summ, "episodes": results}
        print(f"[{method}] summary: {json.dumps(summ, indent=2)}")

    with open(args.out, "w") as f:
        json.dump(all_out, f, indent=2)
    print(f"[info] wrote combined comparison -> {args.out}")

    # side-by-side print
    print("\n==================== COMPARISON ====================")
    metrics = ["mean_flow_epe", "mean_motion_mag_pred", "mean_motion_mag_gt",
               "mean_motion_mag_abs_err", "mean_motion_mag_ratio"]
    header = f"{'metric':<26}" + "".join(f"{m:>14}" for m in args.roots)
    print(header)
    for mk in metrics:
        row = f"{mk:<26}"
        for method in args.roots:
            v = all_out[method]["summary"].get(mk)
            row += f"{v:>14.4f}" if v is not None else f"{'-':>14}"
        print(row)


if __name__ == "__main__":
    main()
