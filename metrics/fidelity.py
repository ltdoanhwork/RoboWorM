"""Reference-based fidelity metrics: PSNR, SSIM, LPIPS.

Backend is ``piq`` with its defaults -- the exact calls every
``scripts/infer_*.py`` makes (``piq.psnr / piq.ssim / piq.LPIPS()`` on float
[0,1] NCHW, frames truncated to ``min(T_pred, T_gt)``), so a number produced
here is directly comparable with the ``metrics.json`` the inference runs wrote.

Per-sample value = mean over frames of the per-frame score (piq's
``reduction="mean"`` over the frame batch), chunked on GPU with size-weighted
averaging, which is numerically the same mean.

piq conventions worth knowing before comparing against other papers:
  * PSNR: ``10 log10(1 / (MSE + 1e-8))`` -> bit-identical frames give ~80 dB,
    not inf (``scripts/metrics/_score_vs_dense.py`` uses skimage, which gives inf; the
    two agree on every non-identical frame).
  * SSIM: 11x11 Gaussian (sigma 1.5), and **downsamples** by
    ``max(1, round(min(H, W) / 256))`` first (the original Wang et al. recipe);
    skimage does not, so skimage SSIM differs at >= 384 px.
  * LPIPS: VGG16 backbone, piq's ``lpips_weights.pt`` (torch hub cache), inputs
    in [0,1]. Lower is better.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import FrameMetric


class _PiqFrameMetric(FrameMetric):
    requires_ref = True

    def setup(self) -> None:
        import piq  # noqa: F401  -- fail early with a clear ImportError

    def _weighted_mean(self, fn, pred: np.ndarray, ref: np.ndarray) -> float:
        import torch

        total, n = 0.0, 0
        with torch.no_grad():
            for x, y in self._chunks(pred, ref):
                total += float(fn(x, y).mean().item()) * len(x)
                n += len(x)
        return total / n


class PSNR(_PiqFrameMetric):
    name = "psnr"
    higher_is_better = {"psnr": True}

    def score_frames(self, pred, ref, text) -> dict[str, float]:
        import piq

        return {"psnr": self._weighted_mean(
            lambda x, y: piq.psnr(x, y, data_range=1.0, reduction="none"), pred, ref)}


class SSIM(_PiqFrameMetric):
    name = "ssim"
    higher_is_better = {"ssim": True}

    def score_frames(self, pred, ref, text) -> dict[str, float]:
        import piq

        return {"ssim": self._weighted_mean(
            lambda x, y: piq.ssim(x, y, data_range=1.0, reduction="none"), pred, ref)}


class LPIPS(_PiqFrameMetric):
    name = "lpips"
    higher_is_better = {"lpips": False}

    def __init__(self, device: str = "cuda", **config: Any) -> None:
        super().__init__(device=device, **config)
        self._fn = None

    def setup(self) -> None:
        import piq

        self._fn = piq.LPIPS(reduction="none").to(self.device).eval()

    def score_frames(self, pred, ref, text) -> dict[str, float]:
        return {"lpips": self._weighted_mean(self._fn, pred, ref)}


class Fidelity(_PiqFrameMetric):
    """PSNR + SSIM + LPIPS in one decode pass (what the inference scripts report)."""

    name = "fidelity"
    higher_is_better = {"psnr": True, "ssim": True, "lpips": False}

    def __init__(self, device: str = "cuda", **config: Any) -> None:
        super().__init__(device=device, **config)
        self._lpips = None

    def setup(self) -> None:
        import piq

        self._lpips = piq.LPIPS(reduction="none").to(self.device).eval()

    def score_frames(self, pred, ref, text) -> dict[str, float]:
        import piq
        import torch

        sums = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
        n = 0
        with torch.no_grad():
            for x, y in self._chunks(pred, ref):
                k = len(x)
                sums["psnr"] += float(piq.psnr(x, y, data_range=1.0, reduction="none").mean()) * k
                sums["ssim"] += float(piq.ssim(x, y, data_range=1.0, reduction="none").mean()) * k
                sums["lpips"] += float(self._lpips(x, y).mean()) * k
                n += k
        return {k: v / n for k, v in sums.items()}
