#!/usr/bin/env python3
"""Rank humanoid multiview episodes by the brightness of the 2x2 GT grid.

The published qualitative ladder came out dark because wrist cameras dominate the
lower half of the grid and those two tasks are dim overall. This scores the SAME
grid the model is trained/evaluated on (the 4 cams of the task's detected layout),
so the ranking reflects what a viewer actually sees -- not one bright overhead view.

Reports mean luma (0-255), the fraction of near-black pixels, and per-view means,
so a grid that is bright on average but has two black wrist tiles is not mistaken
for a good pick.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DS = ROOT / "datasets/humanoid/multiview"

LAYOUT_A = ["observation.images.cam_front_rgb", "observation.images.cam_head_rgb",
            "observation.images.cam_left_wrist_rgb", "observation.images.cam_right_wrist_rgb"]
LAYOUT_B = ["observation.images.cam_high_rgb", "observation.images.cam_third_view",
            "observation.images.cam_left_wrist_rgb", "observation.images.cam_right_wrist_rgb"]


def layout_for(task_dir: Path) -> list[str] | None:
    chunks = sorted(p for p in (task_dir / "videos").iterdir() if p.is_dir())
    if not chunks:
        return None
    cams = {d.name for d in chunks[0].iterdir() if d.is_dir()}
    # Same precedence as scripts/infer/infer_bimanual_multiview_rollout.py:detect_layout
    if "observation.images.cam_front_rgb" in cams:
        return LAYOUT_A
    if "observation.images.cam_high_rgb" in cams or "observation.images.cam_third_view" in cams:
        return LAYOUT_B
    return None


def luma(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])


def score_episode(task_dir: Path, cams: list[str], ep: str, n_probe: int = 5) -> dict | None:
    per_view, dark_fracs = [], []
    for cam in cams:
        hits = list((task_dir / "videos").glob(f"*/{cam}/{ep}.mp4"))
        if not hits:
            return None  # a missing view means no grid can be built
        try:
            frames = np.stack([np.asarray(f) for f in iio.imiter(hits[0])])
        except Exception:
            return None
        idx = np.linspace(0, len(frames) - 1, min(n_probe, len(frames))).astype(int)
        y = luma(frames[idx].astype(np.float32))
        per_view.append(float(y.mean()))
        dark_fracs.append(float((y < 30).mean()))
    return {
        "episode": ep,
        "grid_mean_luma": round(float(np.mean(per_view)), 1),
        "darkest_view_luma": round(float(np.min(per_view)), 1),
        "dark_pixel_frac": round(float(np.mean(dark_fracs)), 3),
        "per_view_luma": [round(v, 1) for v in per_view],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-episodes", type=int, default=12,
                    help="episodes to probe per task (they are long; probing is the cost)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for split in ("makovian", "non_makovian"):
        for task_dir in sorted((DS / split).iterdir()):
            if not (task_dir / "videos").is_dir():
                continue
            cams = layout_for(task_dir)
            if cams is None:
                print(f"  skip {task_dir.name}: no known layout")
                continue
            eps = sorted({p.stem for p in task_dir.rglob("videos/*/*/episode_*.mp4")})
            eps = eps[: args.max_episodes]
            print(f"[{split}/{task_dir.name}] layout={'A' if cams is LAYOUT_A else 'B'} "
                  f"probing {len(eps)} eps")
            for ep in eps:
                r = score_episode(task_dir, cams, ep)
                if r:
                    r.update(split=split, task=task_dir.name)
                    rows.append(r)

    rows.sort(key=lambda r: -r["grid_mean_luma"])
    print(f"\n{'split':<13}{'task':<52}{'episode':<17}{'grid':>6}{'dark_view':>11}{'dark%':>8}")
    for r in rows:
        print(f"{r['split']:<13}{r['task']:<52}{r['episode']:<17}"
              f"{r['grid_mean_luma']:>6}{r['darkest_view_luma']:>11}"
              f"{r['dark_pixel_frac'] * 100:>7.1f}%")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
