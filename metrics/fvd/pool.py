"""Shared real-video reference pool for FVD (paper App. B.3.3).

Protocol in one paragraph: for every (embodiment, camera, regime) there is ONE fixed
pool of real I3D windows, shared by every world model and every acceleration method
of that setting. Real clips reach the pool through many method trees (the same GT is
hard-linked into each), so they are deduplicated by content identity
(:func:`ref_identity`) before pooling. Multi-view predictions are cut into camera tiles
and scored per camera; a cell's FVD is the mean over the cameras common to all its
configurations. Everything is decimated to 5 fps first (``core.resample_fps``).

This module holds the protocol pieces that are independent of the repo's
spreadsheet / figure exporters, so the same pool can be rebuilt for a new model:

    pool = SharedRealPool()
    pool.add_real(("bimanual", "cam0", "makovian"), ident, feats)   # dedup by ident
    pool.add_pred(("bimanual", "Dreamgen MV", "makovian", "dense", "cam0"), feats)
    fvd = pool.fvd(("bimanual", "Dreamgen MV", "makovian", "dense", "cam0"))

The paper numbers were produced by the asset scripts listed in ``README.md`` next to
this file (``assets/scripts/fvd_camera_match.py`` -> ``extract_perview_fvd_feats.py``
-> ``pool_perview_fvd.py`` / ``pool_singleview_fvd.py``), which import from here.
"""
from __future__ import annotations

import collections
import hashlib
from typing import Hashable

import numpy as np

from .core import FVD_MIN_CLIPS, frechet_from_moments, moments

FVD_MIN_WINDOWS = FVD_MIN_CLIPS

#: tile index -> camera id per multiview model family. Decided by data, not by stored
#: view names (assets/scripts/fvd_camera_match.py): CW-4view tile i == DreamGen tile i,
#: Ctrl-World's 3 tiles are DreamGen tiles 0, 2, 3 (it omits cam_low).
TILE_CAMERA = {
    "CW-4view MV": {0: "cam0", 1: "cam1", 2: "cam2", 3: "cam3"},
    "Dreamgen MV": {0: "cam0", 1: "cam1", 2: "cam2", 3: "cam3"},
    "Ctrl-World MV": {0: "cam0", 1: "cam2", 2: "cam3"},
}
#: (embodiment, regime) -> cameras that may be pooled. Cameras whose own GT sits far
#: above the pool's split-half floor (a framing mismatch between models) are dropped
#: (assets/scripts/verify_shared_pool_fairness.py); bimanual is clean everywhere.
#: Measured gt/floor: single_arm cam3 DreamGen 11.4-13.9x, cam2 DreamGen 3.6-4.9x,
#: humanoid cam0 DreamGen 10.3x, cam1 CW-4view 5.6x -> DROP; humanoid cam2/cam3 <= 0.5x,
#: single_arm cam0 0.1-0.9x -> keep. These are the cells fvd_camera_match.py flagged
#: AMBIGUOUS/WEAK (margin 1.3-2.6x), so two independent tests agree.
RESTRICT_CAMERAS = {
    ("humanoid", "makovian"): {"cam2", "cam3"},
    ("humanoid", "non_makovian"): {"cam2", "cam3"},
    ("single_arm", "makovian"): {"cam0"},
    ("single_arm", "non_makovian"): {"cam0"},
}


def ref_identity(ref: str) -> str:
    """A GT file's identity, independent of which method tree it is reached through.

    ``sampling_dataset/dense/.../gt_all_views.mp4`` and
    ``sampling_dataset/pisa/.../gt_all_views.mp4`` are the same bytes; deduplicating on
    the raw path inflated the real pool ~10x (56,668 windows from ~150 real videos on
    bimanual MV makovian). Stripping ``sampling_dataset/<method>/`` gives one key per
    real video.
    """
    parts = ref.split("/sampling_dataset/")
    if len(parts) != 2:
        return ref
    tail = parts[1].split("/", 1)
    return tail[1] if len(tail) == 2 else parts[1]


def split_half(ident: str) -> int:
    """Deterministic 0/1 split of real videos for the noise floor (md5, not ``hash``)."""
    return hashlib.md5(ident.encode()).digest()[0] & 1


def budgeted(x: np.ndarray, k: int | None, seed: str) -> np.ndarray:
    """Deterministically subsample to ``k`` rows so every compared cell uses the same n.

    FVD is biased by sample size (paper Fig. B3: 4.1x median inflation at n=128), so
    cells are only comparable at a matched number of windows.
    """
    if k is None or len(x) <= k:
        return x
    rng = np.random.default_rng(int.from_bytes(hashlib.md5(seed.encode()).digest()[:8], "little"))
    return x[rng.choice(len(x), k, replace=False)]


class SharedRealPool:
    """Accumulate deduplicated real windows per pool key and predicted windows per cell."""

    def __init__(self, min_windows: int = FVD_MIN_WINDOWS) -> None:
        self.min_windows = int(min_windows)
        self.real: dict[Hashable, list[np.ndarray]] = collections.defaultdict(list)
        self.halves: dict[tuple, list[np.ndarray]] = collections.defaultdict(list)
        self.pred: dict[Hashable, list[np.ndarray]] = collections.defaultdict(list)
        self._seen: set = set()

    def add_real(self, key: Hashable, ident: str, feats: np.ndarray) -> bool:
        """Add one real clip's windows; returns False if this video was already pooled."""
        if (key, ident) in self._seen or not len(feats):
            return False
        self._seen.add((key, ident))
        self.real[key].append(np.asarray(feats))
        self.halves[(key, split_half(ident))].append(np.asarray(feats))
        return True

    def add_pred(self, key: Hashable, feats: np.ndarray) -> None:
        if len(feats):
            self.pred[key].append(np.asarray(feats))

    @staticmethod
    def _cat(chunks) -> np.ndarray:
        return np.concatenate(chunks).astype(np.float64)

    def fvd(self, pred_key: Hashable, pool_key: Hashable, budget: int | None = None) -> float | None:
        """FVD of one cell against its shared pool (None below ``min_windows``)."""
        if pred_key not in self.pred or pool_key not in self.real:
            return None
        x = budgeted(self._cat(self.pred[pred_key]), budget, "p|" + str(pred_key))
        r = budgeted(self._cat(self.real[pool_key]), budget, "ref|" + str(pool_key))
        if len(x) < self.min_windows or len(r) < self.min_windows:
            return None
        return frechet_from_moments(*moments(x), *moments(r))

    def floor(self, pool_key: Hashable, budget: int | None = None) -> float | None:
        """Split-half real-vs-real FVD: an episode-diversity upper bound on estimator noise."""
        a, b = self.halves.get((pool_key, 0)), self.halves.get((pool_key, 1))
        if not a or not b:
            return None
        a, b = self._cat(a), self._cat(b)
        n = min(len(a), len(b), budget or 10 ** 9)
        if n < self.min_windows:
            return None
        return frechet_from_moments(*moments(budgeted(a, n, "fa|" + str(pool_key))),
                                    *moments(budgeted(b, n, "fb|" + str(pool_key))))


def mean_over_common_cameras(per_camera: dict[tuple, float]) -> dict[tuple, float]:
    """``{(cell..., config, camera): fvd}`` -> ``{(cell..., config): mean over the cameras
    present for EVERY config of that cell}``. The cell is all key fields but the last two."""
    cams = collections.defaultdict(lambda: collections.defaultdict(set))
    for k in per_camera:
        cams[k[:-2]][k[-2]].add(k[-1])
    common = {c: set.intersection(*by.values()) for c, by in cams.items() if by}
    agg = collections.defaultdict(list)
    for k, v in per_camera.items():
        if k[-1] in common.get(k[:-2], ()):
            agg[k[:-1]].append(v)
    return {k: float(np.mean(v)) for k, v in agg.items()}


__all__ = ["FVD_MIN_WINDOWS", "TILE_CAMERA", "RESTRICT_CAMERAS", "ref_identity", "split_half",
           "budgeted", "SharedRealPool", "mean_over_common_cameras"]
