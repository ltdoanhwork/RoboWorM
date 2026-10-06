#!/usr/bin/env python3
"""Shared pieces for per-camera FVD: episode identity, mosaic geometry, fps matching.

THIN SHIM. The implementation moved to ``metrics/fvd/`` (the paper's FVD metric,
registered as ``fvd``) during the 2026-10-05 refactor; this module re-exports the
exact same names so the camera-matching probe, the feature extractor and the pool
scripts keep importing ``from fvd_pool_common import ...`` unchanged. See
metrics/fvd/README.md for the protocol these implement.

Verified mosaic geometry (ffprobe on dense GT of every multiview cell):
     1280x192  CW-4view      1x4 strip of 320x192
      960x192  Ctrl-World    1x3 strip of 320x192
      768x432  DreamGen MV   2x2 grid  of 384x216
GT fps: Ctrl-World / CW-4view = 5 everywhere; DreamGen = 16 (bimanual, humanoid),
10 (single_arm). TARGET_FPS is the minimum, so every clip is decimated, never upsampled.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from metrics.fvd import (GRIDS, I3D_SIZE, I3D_WINDOW, SPLIT_DIRS,  # noqa: E402,F401
                         TARGET_FPS, ep_key, i3d_features, moments, n_views,
                         resample_fps, split_mosaic)

__all__ = ["GRIDS", "I3D_SIZE", "I3D_WINDOW", "SPLIT_DIRS", "TARGET_FPS", "ep_key",
           "i3d_features", "moments", "n_views", "resample_fps", "split_mosaic"]
