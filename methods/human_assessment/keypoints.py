"""Keypoint trajectory tracking via CoTracker3."""

from __future__ import annotations

import cv2
import numpy as np
import torch

_device = "cuda" if torch.cuda.is_available() else "cpu"
_cotracker = None


def _get_cotracker():
    global _cotracker
    if _cotracker is None:
        _cotracker = torch.hub.load(
            "facebookresearch/co-tracker", "cotracker3_offline"
        ).to(_device)
        _cotracker.eval()
    return _cotracker


def track_keypoints(bgr_frames: list[np.ndarray], pts0: np.ndarray):
    """
    Track pts0 (K, 2) [x, y] across BGR frames.

    Returns:
        traj:  (T, K, 2) float32
        alive: (T, K) bool
    """
    pts0 = np.asarray(pts0, dtype=np.float32)
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in bgr_frames]
    video = torch.from_numpy(np.stack(rgb)).permute(0, 3, 1, 2)[None].float().to(_device)

    k = len(pts0)
    queries = np.zeros((1, k, 3), dtype=np.float32)
    queries[0, :, 1:] = pts0
    q = torch.from_numpy(queries).to(_device)

    with torch.no_grad():
        tracks, vis = _get_cotracker()(video, queries=q)

    traj = tracks[0].cpu().numpy().astype(np.float32)
    alive = vis[0].cpu().numpy().astype(bool)
    return traj, alive
