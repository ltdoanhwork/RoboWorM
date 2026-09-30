#!/usr/bin/env python3
"""Shared pieces for per-camera FVD: episode identity, mosaic geometry, fps matching.

Imported by the camera-matching probe and by the feature extractor so both agree on
exactly how a mosaic is cut and how a clip is retimed. See
.cursor/skills/fvd-shared-real-pool/SKILL.md for the protocol these implement.

Verified mosaic geometry (ffprobe on dense GT of every multiview cell):
    1280x192  CW-4view      1x4 strip of 320x192
     960x192  Ctrl-World    1x3 strip of 320x192
     768x432  DreamGen MV   2x2 grid  of 384x216
GT fps: Ctrl-World / CW-4view = 5 everywhere; DreamGen = 16 (bimanual, humanoid),
10 (single_arm). TARGET_FPS is the minimum, so every clip is decimated, never upsampled.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

I3D_WINDOW = 16
I3D_SIZE = 224
TARGET_FPS = 5.0

# (height, width) -> (rows, cols). Keyed by shape because the layout is a property of the
# model family, and the shape identifies it uniquely across all cells measured.
GRIDS = {
    (192, 1280): (1, 4),
    (192, 960):  (1, 3),
    (432, 768):  (2, 2),
}

SPLIT_DIRS = {'makovian', 'non_makovian', 'val', 'train', 'all'}


def n_views(h: int, w: int) -> int | None:
    g = GRIDS.get((h, w))
    return None if g is None else g[0] * g[1]


def split_mosaic(vid: np.ndarray) -> list[np.ndarray] | None:
    """Cut a mosaic clip (T,H,W,3) into tiles in row-major order.

    Returns tiles by POSITION only. Camera identity is resolved separately, by data
    (see fvd_camera_match.py) rather than by name: humanoid Ctrl-World annotations carry
    no `view_keys`, so the inference script fell back to the bimanual default names and
    the labels stored in metrics.json are not trustworthy for that cell.
    """
    _T, H, W, _C = vid.shape
    g = GRIDS.get((H, W))
    if g is None:
        return None
    r, c = g
    th, tw = H // r, W // c
    return [vid[:, (i // c) * th:(i // c) * th + th, (i % c) * tw:(i % c) * tw + tw]
            for i in range(r * c)]


def resample_fps(vid: np.ndarray, src_fps: float, dst_fps: float = TARGET_FPS
                 ) -> np.ndarray:
    """Nearest-frame decimation to dst_fps. Never interpolates: synthesised frames would
    synthesise motion, and motion is what I3D measures."""
    if not np.isfinite(src_fps) or src_fps <= dst_fps * 1.01:
        return vid
    n_out = max(int(round(len(vid) * dst_fps / src_fps)), I3D_WINDOW)
    idx = np.clip(np.round(np.arange(n_out) * src_fps / dst_fps).astype(int),
                  0, len(vid) - 1)
    return vid[idx]


def ep_key(p: str) -> tuple[str, int] | None:
    """Normalise the three path shapes in this repo to (task, episode_number).

        ctrlworld, task in dir   .../makovian/episode_close_toolbox__000002/gt_all_views.mp4
        dreamgen, task as parent .../iter_000120000/close_toolbox/episode_000002/full_gt.mp4
        single_arm, no task      .../makovian/episode_000255/gt_all_views.mp4
                                 .../makovian/episode_000255_gt.mp4
    """
    parts = Path(p).parts
    for i in range(len(parts) - 1, -1, -1):
        c = parts[i]
        if not c.startswith('episode'):
            continue
        stem = c[:-4] if c.endswith('.mp4') else c
        m = re.match(r'episode_?(?P<task>.*?)__(?P<id>\d+)$', stem)
        if m:
            return (m.group('task').lower(), int(m.group('id')))
        m = re.match(r'episode_?(?P<id>\d+)(?:_gt|_pred|_full)?$', stem)
        if m:
            task = ''
            if i > 0:
                par = parts[i - 1]
                if par not in SPLIT_DIRS and not par.startswith('iter_'):
                    task = par.lower()
            return (task, int(m.group('id')))
    return None


def i3d_features(tile: np.ndarray, model, device, canon_hw=None) -> list[np.ndarray]:
    """Consecutive non-overlapping 16-frame windows -> I3D logits features."""
    import torch
    out = []
    with torch.no_grad():
        for s in range(0, len(tile) - I3D_WINDOW + 1, I3D_WINDOW):
            w = torch.from_numpy(np.ascontiguousarray(tile[s:s + I3D_WINDOW])).to(device)
            w = w.permute(0, 3, 1, 2).float() / 255
            if canon_hw is not None:
                w = torch.nn.functional.interpolate(w, size=canon_hw, mode='bilinear',
                                                    align_corners=False)
            w = torch.nn.functional.interpolate(w, size=(I3D_SIZE, I3D_SIZE),
                                                mode='bilinear', align_corners=False)
            w = (w * 2 - 1).permute(1, 0, 2, 3).unsqueeze(0)
            f = model(w, rescale=False, resize=False, return_features=True)
            out.append(f.squeeze(0).cpu().numpy())
    return out


def moments(x):
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(0)
    xc = x - mu
    return mu, xc.T.dot(xc) / max(len(x) - 1, 1)
