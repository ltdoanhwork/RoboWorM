"""FVD core: I3D window features, fps decimation, mosaic split, Frechet distance.

Source metric
    Frechet Video Distance -- T. Unterthiner, S. van Steenkiste, K. Kurach,
    R. Marinier, M. Michalski, S. Gelly, "Towards Accurate Generative Models of
    Video: A New Metric & Challenges", arXiv:1812.01717, 2018. Frechet distance
    between Gaussians fitted to I3D (Kinetics-400) features of real and generated
    clips. Lower is better.

This is the single implementation of the FVD pieces the paper uses (App. B.3.3).
``scripts/metrics/compute_dr_metrics.py`` (per-configuration DR pass) and the
shared-pool scripts in ``assets/scripts/`` (via the ``fvd_pool_common.py`` shim)
import from here. The shared real-pool bookkeeping lives in ``metrics/fvd/pool.py``;
setup and protocol are documented in ``metrics/fvd/README.md``.

Protocol constants:
  * I3D = the StyleGAN-V torchscript export (``i3d_torchscript.pt``), 400-d
    pre-logit features, called with ``rescale=False, resize=False``.
  * A clip is cut into consecutive, NON-overlapping 16-frame windows; each window
    is bilinearly resized to 224x224, mapped to [-1, 1] and contributes one
    feature vector. A trailing partial window is dropped.
  * Shared-pool FVD first decimates every clip to ``TARGET_FPS = 5`` by
    nearest-frame selection (never interpolation: synthetic frames would be
    synthetic motion, and motion is what I3D measures).
  * A set with fewer than ``FVD_MIN_CLIPS = 64`` windows gets ``None``: the
    400x400 covariance is too rank-deficient for the number to mean anything.
  * FVD is a SET statistic -- one number per cell, effective n = 1. It cannot
    carry a per-cell CI / Wilcoxon claim (see .cursor/rules/fvd-table-reporting.mdc).

Checkpoint: ``checkpoints/metrics/fvd/i3d_torchscript.pt`` (override with
``$FVD_I3D_CKPT``; the pre-refactor ``checkpoints/fvd/`` path is used as a
fallback while the compat symlinks exist).

Registered as ``fvd`` (:class:`FVD`): per-sample rows hold the window counts,
``summary["fvd"]["mean"]`` holds the set FVD of all preds vs all refs (or vs a
precomputed reference pool, ``ref_pool=<.npy|.npz>``).
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..base import REPO_ROOT, BaseMetric, MetricResult, Sample

# --------------------------------------------------------------------------- #
# protocol constants
# --------------------------------------------------------------------------- #
I3D_WINDOW = 16          # frames per I3D window (the FVD reference implementations)
I3D_SIZE = 224           # I3D input side
TARGET_FPS = 5.0         # shared-pool fps: the minimum GT fps over all cells
FVD_MIN_CLIPS = 64       # min windows per side; below this FVD is reported as None

I3D_CKPT_DEFAULT = REPO_ROOT / "checkpoints/metrics/fvd/i3d_torchscript.pt"
I3D_CKPT_LEGACY = REPO_ROOT / "checkpoints/fvd/i3d_torchscript.pt"


def i3d_ckpt_path() -> Path:
    """``$FVD_I3D_CKPT`` > checkpoints/metrics/fvd/ > legacy checkpoints/fvd/."""
    env = os.environ.get("FVD_I3D_CKPT")
    if env:
        return Path(env)
    if I3D_CKPT_DEFAULT.is_file() or not I3D_CKPT_LEGACY.is_file():
        return I3D_CKPT_DEFAULT
    return I3D_CKPT_LEGACY


#: resolved at import; kept as a module attribute for the legacy callers
I3D_CKPT = i3d_ckpt_path()


def load_i3d(device: str = "cuda", ckpt: str | os.PathLike[str] | None = None):
    import torch

    path = Path(ckpt) if ckpt else i3d_ckpt_path()
    if not path.is_file():
        raise FileNotFoundError(f"I3D torchscript not found: {path} (set FVD_I3D_CKPT)")
    return torch.jit.load(str(path)).to(device).eval()


# --------------------------------------------------------------------------- #
# decoding
# --------------------------------------------------------------------------- #
class CorruptVideo(Exception):
    """Source mp4 that cannot be decoded -- a data defect, not a transient fault."""


def read_video(path: str) -> np.ndarray:
    """(T,H,W,3) uint8 RGB via cv2 (the decoder every published FVD number used).

    Raises CorruptVideo for a file cv2 cannot decode -- 13 mp4s in this dataset are
    truncated (`moov atom not found`, 0-48 bytes or an exact 2 MB buffer multiple).
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise CorruptVideo(f"cannot open {path}")
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not out:
        raise CorruptVideo(f"no frames in {path}")
    return np.stack(out)


def video_fps(path: str) -> float:
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else float("nan")
    cap.release()
    return fps if fps > 0 else float("nan")


# --------------------------------------------------------------------------- #
# mosaic geometry / fps / episode identity (shared per-camera pool)
# --------------------------------------------------------------------------- #
# Verified mosaic geometry (ffprobe on dense GT of every multiview cell):
#     1280x192  CW-4view      1x4 strip of 320x192
#      960x192  Ctrl-World    1x3 strip of 320x192
#      768x432  DreamGen MV   2x2 grid  of 384x216
# GT fps: Ctrl-World / CW-4view = 5 everywhere; DreamGen = 16 (bimanual, humanoid),
# 10 (single_arm). TARGET_FPS is the minimum, so every clip is decimated, never upsampled.
#: (height, width) -> (rows, cols); the shape identifies the model family uniquely.
GRIDS = {
    (192, 1280): (1, 4),
    (192, 960): (1, 3),
    (432, 768): (2, 2),
}

SPLIT_DIRS = {"makovian", "non_makovian", "val", "train", "all"}


def n_views(h: int, w: int) -> int | None:
    g = GRIDS.get((h, w))
    return None if g is None else g[0] * g[1]


def split_mosaic(vid: np.ndarray) -> list[np.ndarray] | None:
    """Cut a mosaic clip (T,H,W,3) into tiles in row-major order.

    Returns tiles by POSITION only. Camera identity is resolved separately, by data
    (see assets/scripts/fvd_camera_match.py) rather than by name: humanoid Ctrl-World
    annotations carry no `view_keys`, so the labels in metrics.json are not trustworthy.
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
    """Nearest-frame decimation to dst_fps. Never interpolates."""
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
        if not c.startswith("episode"):
            continue
        stem = c[:-4] if c.endswith(".mp4") else c
        m = re.match(r"episode_?(?P<task>.*?)__(?P<id>\d+)$", stem)
        if m:
            return (m.group("task").lower(), int(m.group("id")))
        m = re.match(r"episode_?(?P<id>\d+)(?:_gt|_pred|_full)?$", stem)
        if m:
            task = ""
            if i > 0:
                par = parts[i - 1]
                if par not in SPLIT_DIRS and not par.startswith("iter_"):
                    task = par.lower()
            return (task, int(m.group("id")))
    return None


# --------------------------------------------------------------------------- #
# I3D features
# --------------------------------------------------------------------------- #
def window_feature(model, w):
    """One window: float [0,1] (T=16, 3, H, W) tensor -> (400,) feature tensor (on device)."""
    import torch

    w = torch.nn.functional.interpolate(w, size=(I3D_SIZE, I3D_SIZE), mode="bilinear",
                                        align_corners=False)
    w = (w * 2 - 1).permute(1, 0, 2, 3).unsqueeze(0)        # 1,3,T,H,W
    return model(w, rescale=False, resize=False, return_features=True).squeeze(0)


class I3DWindowStream:
    """Feed float [0,1] TCHW chunks in order; emits one feature per full 16-frame window.

    This is exactly the buffering ``compute_dr_metrics.score`` has always done
    (windows never straddle, trailing partial window dropped), factored out so the
    DR pass can interleave it with PSNR/SSIM/LPIPS on the same GPU chunk.
    """

    def __init__(self, model) -> None:
        self.model = model
        self.buf: list = []
        self.feats: list[list[float]] = []

    def push(self, chunk) -> None:
        import torch

        self.buf.append(chunk)
        vid = torch.cat(self.buf) if len(self.buf) > 1 else self.buf[0]
        while len(vid) >= I3D_WINDOW:
            w = vid[:I3D_WINDOW]
            vid = vid[I3D_WINDOW:]
            self.feats.append(window_feature(self.model, w).cpu().numpy().tolist())
        self.buf.clear()
        if len(vid):
            self.buf.append(vid)


def i3d_features(tile: np.ndarray, model, device, canon_hw=None) -> list[np.ndarray]:
    """uint8 (T,H,W,3) clip -> list of (400,) features over non-overlapping 16-f windows.

    ``canon_hw`` first resizes every frame to a canonical (H, W) (the shared pool
    uses this so tiles of different native size go through the same resampling).
    """
    import torch

    out = []
    with torch.no_grad():
        for s in range(0, len(tile) - I3D_WINDOW + 1, I3D_WINDOW):
            w = torch.from_numpy(np.ascontiguousarray(tile[s:s + I3D_WINDOW])).to(device)
            w = w.permute(0, 3, 1, 2).float() / 255
            if canon_hw is not None:
                w = torch.nn.functional.interpolate(w, size=canon_hw, mode="bilinear",
                                                    align_corners=False)
            out.append(window_feature(model, w).cpu().numpy())
    return out


# --------------------------------------------------------------------------- #
# Frechet distance
# --------------------------------------------------------------------------- #
def moments(x):
    """(mean, unbiased covariance) in float64 -- the pooled-FVD sufficient statistics."""
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(0)
    xc = x - mu
    return mu, xc.T.dot(xc) / max(len(x) - 1, 1)


def frechet_from_moments(mx, cx, my, cy) -> float:
    from scipy import linalg

    # `disp=` is deprecated in SciPy >= 1.14 and removed in 1.18; without it older
    # SciPy returns the matrix alone, newer may return a tuple -- accept both.
    covmean = linalg.sqrtm(np.asarray(cx).dot(cy))
    if isinstance(covmean, tuple):
        covmean = covmean[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mx - my) ** 2).sum() + np.trace(cx) + np.trace(cy)
                 - 2 * np.trace(covmean))


def frechet(x: np.ndarray, y: np.ndarray) -> float:
    """FVD between two (N, 400) feature sets (np.cov, as in the dr-v1 aggregation)."""
    mx, my = x.mean(0), y.mean(0)
    cx = np.cov(x, rowvar=False)
    cy = np.cov(y, rowvar=False)
    return frechet_from_moments(mx, cx, my, cy)


def fvd_or_none(pred_feats, ref_feats, min_windows: int = FVD_MIN_CLIPS) -> float | None:
    p, q = np.asarray(pred_feats), np.asarray(ref_feats)
    if len(p) < min_windows or len(q) < min_windows:
        return None
    return frechet(p, q)


# --------------------------------------------------------------------------- #
# BaseMetric
# --------------------------------------------------------------------------- #
class FVD(BaseMetric):
    """Set-level FVD over every sample in one ``evaluate`` call (= one cell).

    Per sample: decode pred (and ref, unless ``ref_pool`` is given), optionally
    decimate to ``target_fps`` (default 5.0 = shared-pool protocol; ``None`` keeps
    native fps = the dr-v1 per-configuration protocol), extract I3D window
    features. ``summary["fvd"]["mean"]`` is the FVD of the pooled pred windows vs
    the pooled ref windows; ``n`` is the number of pred windows. NaN when either
    side has < ``min_windows`` windows.
    """

    name = "fvd"
    higher_is_better = {"fvd": False}

    def __init__(self, device: str = "cuda", target_fps: float | None = TARGET_FPS,
                 min_windows: int = FVD_MIN_CLIPS, ref_pool: str | None = None,
                 ckpt: str | None = None, **config: Any) -> None:
        super().__init__(device=device, target_fps=target_fps, min_windows=min_windows,
                         ref_pool=ref_pool, ckpt=ckpt, **config)
        self.target_fps = None if target_fps in (None, "", "none", 0) else float(target_fps)
        self.min_windows = int(min_windows)
        self.ref_pool = ref_pool
        self._model = None
        self._feats: dict[str, dict[str, list]] = {}
        self._pool: np.ndarray | None = None

    @property
    def requires_ref(self) -> bool:  # type: ignore[override]
        return self.ref_pool is None

    def setup(self) -> None:
        self._model = load_i3d(self.device, self.config.get("ckpt"))
        if self.ref_pool:
            z = np.load(self.ref_pool)
            self._pool = np.asarray(z["feats"] if hasattr(z, "files") else z)

    def _clip_feats(self, src) -> list[np.ndarray]:
        if isinstance(src, np.ndarray):
            vid = src
        else:
            vid = read_video(str(src))
            if self.target_fps is not None:
                vid = resample_fps(vid, video_fps(str(src)), self.target_fps)
        return i3d_features(vid, self._model, self.device)

    def compute(self, sample: Sample) -> dict[str, float]:
        self.ensure_setup()
        fp = self._clip_feats(sample.pred)
        fr = [] if self._pool is not None else self._clip_feats(sample.ref)
        self._feats[sample.id] = {"pred": fp, "ref": fr}
        return {"n_windows_pred": float(len(fp)), "n_windows_ref": float(len(fr))}

    def aggregate(self, per_sample: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        pred = [f for sid in per_sample for f in self._feats.get(sid, {}).get("pred", [])]
        ref = (list(self._pool) if self._pool is not None else
               [f for sid in per_sample for f in self._feats.get(sid, {}).get("ref", [])])
        v = fvd_or_none(pred, ref, self.min_windows)
        return {"fvd": {"mean": float("nan") if v is None else v, "std": float("nan"),
                        "sem": float("nan"), "n": len(pred), "n_ref": len(ref),
                        "n_missing": 0 if v is not None else len(per_sample)}}

    def evaluate(self, samples: Sequence[Sample], **kw) -> MetricResult:
        self._feats = {}
        return super().evaluate(samples, **kw)


__all__ = [
    "I3D_WINDOW", "I3D_SIZE", "TARGET_FPS", "FVD_MIN_CLIPS", "I3D_CKPT", "GRIDS", "SPLIT_DIRS",
    "CorruptVideo", "read_video", "video_fps", "load_i3d", "i3d_ckpt_path",
    "n_views", "split_mosaic", "resample_fps", "ep_key",
    "window_feature", "I3DWindowStream", "i3d_features",
    "moments", "frechet", "frechet_from_moments", "fvd_or_none", "FVD",
]
