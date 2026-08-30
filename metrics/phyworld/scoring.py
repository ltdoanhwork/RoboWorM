"""Map raw PhyWorld numbers onto paper scores (0–100)."""

from __future__ import annotations

import math
from typing import Optional

_EPS = 1e-8


def scene_integrity_score(clip_iqa_plus: float) -> float:
    """CLIP-IQA+ is already in [0, 1]."""
    return float(max(0.0, min(1.0, clip_iqa_plus)) * 100.0)


def task_lock_score(clip_score_0_100: float) -> float:
    """torchmetrics CLIPScore is already 0–100."""
    return float(max(0.0, min(100.0, clip_score_0_100)))


def entity_presence_score(detection_rate: float) -> float:
    return float(max(0.0, min(1.0, detection_rate)) * 100.0)


def phy_qual(scene_integrity: float, task_lock: float, entity_presence: float) -> float:
    return float((scene_integrity + task_lock + entity_presence) / 3.0)


def actor_dynamics_score(med_fg_minus_bg: float, scale: float = 2.0) -> float:
    """Soft-clip median(fg)-median(bg) to [0, 100]. 0 px → 50."""
    x = math.tanh(med_fg_minus_bg / max(scale, _EPS))
    return float(50.0 * (1.0 + x))


def motion_continuity_score(ssim: float) -> float:
    return float(max(0.0, min(1.0, ssim)) * 100.0)


def phy_dyn(actor_mapped: float, continuity_mapped: float) -> float:
    return float((actor_mapped + continuity_mapped) / 2.0)


def kinetic_drift(mag_method: float, mag_dense: float) -> float:
    if mag_method <= 0 or mag_dense <= 0:
        return float("nan")
    return float(abs(math.log(mag_method / mag_dense)))


def continuity_gap(ssim_dense: float, ssim_method: float) -> float:
    return float(ssim_dense - ssim_method)


def look_drift(scene_integrity_method: float, scene_integrity_dense: float) -> float:
    return float(abs(scene_integrity_method - scene_integrity_dense))


def _inv_unit(x: float, lo: float, hi: float) -> Optional[float]:
    """Map [lo, hi] → 100..0 (lower raw = higher score). None if x is nan."""
    if x != x:  # nan
        return None
    if hi <= lo:
        return 100.0 if x <= lo else 0.0
    t = (x - lo) / (hi - lo)
    return float(max(0.0, min(1.0, 1.0 - t)) * 100.0)


def phy_fid(
    motion_fidelity: float,
    kinetic_drift_v: float,
    continuity_gap_v: float,
    look_drift_v: float,
    epe_hi: float = 4.0,
    drift_hi: float = 1.0,
    gap_hi: float = 0.15,
    look_hi: float = 15.0,
) -> float:
    """Wave-0 defaults: clip each error to a plausible range then average."""
    parts = [
        _inv_unit(motion_fidelity, 0.0, epe_hi),
        _inv_unit(kinetic_drift_v, 0.0, drift_hi),
        _inv_unit(continuity_gap_v, -0.02, gap_hi),
        _inv_unit(look_drift_v, 0.0, look_hi),
    ]
    vals = [p for p in parts if p is not None]
    return float(sum(vals) / len(vals)) if vals else float("nan")
