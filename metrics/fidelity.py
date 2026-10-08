"""Reference-based fidelity metrics: PSNR, SSIM, LPIPS.

Backends, all on float [0,1] NCHW with frames truncated to ``min(T_pred, T_ref)``:
  * PSNR, LPIPS: ``piq`` defaults (``piq.psnr`` / ``piq.LPIPS()``), the calls the
    ``scripts/infer/*.py`` runs make. These agree with the stored paper records.
  * SSIM: ``torchmetrics.functional.image.structural_similarity_index_measure``
    with ``data_range=1.0`` and otherwise default arguments, the exact call of
    ``scripts/metrics/compute_dr_metrics.py`` (the DR wave that produced the paper
    numbers), so SSIM here reproduces the stored records.

Per-sample value = mean over frames of the per-frame score, chunked on GPU with
size-weighted averaging, which is numerically the same mean.

Conventions worth knowing before comparing against other papers:
  * PSNR: ``10 log10(1 / (MSE + 1e-8))`` -> bit-identical frames give ~80 dB,
    not inf (``scripts/metrics/_score_vs_dense.py`` uses skimage, which gives inf; the
    two agree on every non-identical frame).
  * SSIM: 11x11 Gaussian (sigma 1.5), K=(0.01, 0.03), no downsampling. ``piq.ssim``
    (which the inference-time ``metrics.json`` files use) downsamples by
    ``max(1, round(min(H, W) / 256))`` first, so it differs at >= 384 px.
  * LPIPS: VGG16 backbone, piq's ``lpips_weights.pt`` (torch hub cache), inputs
    in [0,1]. Lower is better.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import FrameMetric


def _ssim(x, y):
    """Per-frame SSIM, same call as scripts/metrics/compute_dr_metrics.py."""
    from torchmetrics.functional.image import structural_similarity_index_measure

    return structural_similarity_index_measure(x, y, data_range=1.0, reduction="none")


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

    def setup(self) -> None:
        import torchmetrics  # noqa: F401  -- fail early with a clear ImportError

    def score_frames(self, pred, ref, text) -> dict[str, float]:
        return {"ssim": self._weighted_mean(_ssim, pred, ref)}


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
                sums["ssim"] += float(_ssim(x, y).mean()) * k
                sums["lpips"] += float(self._lpips(x, y).mean()) * k
                n += k
        return {k: v / n for k, v in sums.items()}
