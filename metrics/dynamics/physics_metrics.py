"""Tier-1 physics metrics: optical-flow / motion-magnitude error vs GT.

Reference-based motion metrics for the video-world benchmark. Unlike WorldScore's
reference-free ``motion_magnitude`` (median flow of the prediction only), here we
have paired ground-truth video, so we compare the prediction's dense optical-flow
field against the GT's flow field frame-by-frame. This directly captures whether
the predicted motion matches real motion:

- ``flow_epe``            mean end-point error ||flow_pred - flow_gt|| over pixels &
                          frame-pairs. Lower is better (0 = identical motion).
- ``motion_mag_pred``     mean |flow_pred| (predicted motion magnitude, px/frame).
- ``motion_mag_gt``       mean |flow_gt| (GT motion magnitude, px/frame).
- ``motion_mag_abs_err``  | motion_mag_pred - motion_mag_gt |. Lower is better.
- ``motion_mag_ratio``    motion_mag_pred / motion_mag_gt. 1.0 ideal; <1 = frozen /
                          under-moving, >1 = exaggerated / hallucinated motion.

The flow backend is SEA-RAFT (see ``metrics/dynamics/flow_backend.py``).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

_EPS = 1e-6


def _flow_magnitude(flow: np.ndarray) -> np.ndarray:
    return np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)


def flow_vs_gt_metrics(
    pred_frames: np.ndarray,
    gt_frames: np.ndarray,
    flow_model,
    max_pairs: Optional[int] = None,
) -> Dict[str, float]:
    """Compute flow-vs-GT motion metrics for one view.

    Args:
        pred_frames: (T, H, W, 3) uint8 RGB predicted frames.
        gt_frames:   (T, H, W, 3) uint8 RGB ground-truth frames (same T,H,W).
        flow_model:  object with ``.flow(frame1, frame2) -> (H,W,2)`` (SeaRaftFlow).
        max_pairs:   optionally cap the number of consecutive pairs evaluated.

    Returns:
        dict of scalar metrics (see module docstring).
    """
    assert pred_frames.shape == gt_frames.shape, (
        f"pred/gt shape mismatch: {pred_frames.shape} vs {gt_frames.shape}"
    )
    T = pred_frames.shape[0]
    n_pairs = T - 1
    if max_pairs is not None:
        n_pairs = min(n_pairs, max_pairs)

    epe_list: List[float] = []
    mag_pred_list: List[float] = []
    mag_gt_list: List[float] = []

    for i in range(n_pairs):
        fp = flow_model.flow(pred_frames[i], pred_frames[i + 1])
        fg = flow_model.flow(gt_frames[i], gt_frames[i + 1])

        epe = np.sqrt(((fp - fg) ** 2).sum(axis=-1))  # (H, W)
        epe_list.append(float(epe.mean()))
        mag_pred_list.append(float(_flow_magnitude(fp).mean()))
        mag_gt_list.append(float(_flow_magnitude(fg).mean()))

    mag_pred = float(np.mean(mag_pred_list))
    mag_gt = float(np.mean(mag_gt_list))

    return {
        "flow_epe": float(np.mean(epe_list)),
        "motion_mag_pred": mag_pred,
        "motion_mag_gt": mag_gt,
        "motion_mag_abs_err": abs(mag_pred - mag_gt),
        "motion_mag_ratio": mag_pred / (mag_gt + _EPS),
        "num_pairs": n_pairs,
    }


def aggregate_view_metrics(view_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Average per-view metric dicts into a single dict (skips ``num_pairs``)."""
    if not view_metrics:
        return {}
    keys = ["flow_epe", "motion_mag_pred", "motion_mag_gt", "motion_mag_abs_err", "motion_mag_ratio"]
    out: Dict[str, float] = {}
    for k in keys:
        vals = [m[k] for m in view_metrics.values() if k in m]
        if vals:
            out[k] = float(np.mean(vals))
    return out
