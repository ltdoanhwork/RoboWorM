"""Motion Smoothness (MS): VFIMamba frame-interpolation consistency.

Source metric
    WorldScore "motion smoothness" -- H. Duan, H.-X. Yu, S. Chen, L. Fei-Fei, J. Wu,
    "WorldScore: A Unified Evaluation Benchmark for World Generation", ICCV 2025
    (``worldscore/benchmark/metrics/third_party/motion_smoothness_metrics.py``).
    Every odd frame is re-synthesised from its two even neighbours with VFIMamba
    (Zhang et al., NeurIPS 2024) and compared with the real odd frame; a temporally
    smooth video is easy to interpolate. Higher SSIM / lower MSE = smoother.

Use in the paper
    RQ3 baseline (Sec. 4.2): MS is compared against KIVA and found to track only
    surface transitions. It is NOT a headline column. Note the source formulation
    rewards a frozen video with a near-perfect score.

Differences from the WorldScore implementation (both deliberate, both inherited
from the dense-NR wave that produced the stored values):
    * SSIM is a single global SSIM over the uint8 BGR frame (Wang et al. constants),
      not torchmetrics' windowed SSIM; LPIPS is not computed.
    * test-time augmentation is off by default (``tta=False``; WorldScore uses True).
Frames are the paper analysis frames (5 fps, 256x192, every frame, cv2 JPEG on
disk -- VFIMamba reads files).

Backends: VFIMamba code from ``metrics/WorldScore`` third_party,
checkpoint ``checkpoints/metrics/worldscore/VFIMamba.pkl`` (see metrics/third_party.py).

Registered as ``motion_smoothness``; keys ``motion_smoothness`` (= SSIM x100, higher
better) and ``motion_smoothness_mse`` (lower better).
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

import numpy as np

from .base import ReferenceFreeFrameMetric
from .third_party import add_paths, worldscore_cwd, write_frames


def build_vfimamba(device: str = "cuda"):
    """VFIMamba ``Model`` + ``InputPadder``, importing third_party code directly.

    The WorldScore config import is shimmed so the ``worldscore`` package ``__init__``
    (which pulls pyiqa) is never executed.
    """
    add_paths()
    import config as _cfg  # VFIMamba/config.py
    _cfg.MODEL_CONFIG["LOGNAME"] = "VFIMamba"
    _cfg.MODEL_CONFIG["MODEL_ARCH"] = _cfg.init_model_config(F=32, depth=[2, 2, 2, 3, 3])
    chain = "worldscore.benchmark.metrics.third_party.VFIMamba"
    parts = chain.split(".")
    for k in range(1, len(parts) + 1):
        name = ".".join(parts[:k])
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules[chain + ".config"] = _cfg
    from benchmark.utils.padder import InputPadder
    from Trainer_finetune import Model

    m = Model(-1)
    with worldscore_cwd():
        m.load_model()
    m.eval()
    m.device()
    return m, InputPadder


def _ssim(a, b) -> float:
    """Global SSIM on uint8 BGR (channel-mean), constants per Wang et al. 2004."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    return float(((2 * mu_a * mu_b + C1) * (2 * cov + C2)) /
                 ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2)))


def motion_smoothness(frames, model, InputPadder, device="cuda", TTA=True, stride=1,
                      max_triples=None) -> dict:
    """VFIMamba odd-from-even interpolation over frame *paths*.

    Returns mean ``ssim`` / ``mse`` plus the per-triple lists (the dense-NR wave
    re-weights them by actor activity for its continuity column).
    """
    import cv2
    import torch

    even, odd = frames[::2], frames[1::2]
    ssim_l, mse_l = [], []
    pairs = list(zip(even[:-1], even[1:]))
    sel = list(range(0, len(pairs), stride))
    if max_triples is not None:
        sel = sel[:max_triples]
    for i in sel:
        f0, f2 = pairs[i]
        if i >= len(odd):
            break
        I0, I2, mid_gt = cv2.imread(f0), cv2.imread(f2), cv2.imread(odd[i])
        I0_ = (torch.tensor(I0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
        I2_ = (torch.tensor(I2.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
        padder = InputPadder(I0_.shape, divisor=32)
        I0p, I2p = padder.pad(I0_, I2_)
        with torch.no_grad():
            mid = model.inference(I0p, I2p, True, TTA=TTA, fast_TTA=TTA, scale=0.0)
        mid = (padder.unpad(mid)[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
        ssim_l.append(_ssim(mid, mid_gt))
        mse_l.append(float(np.mean((mid.astype(np.float64) - mid_gt.astype(np.float64)) ** 2)))
    return {
        "ssim": float(np.mean(ssim_l)) if ssim_l else float("nan"),
        "mse": float(np.mean(mse_l)) if mse_l else float("nan"),
        "ssim_per_triple": ssim_l,
        "mse_per_triple": mse_l,
        "n": len(ssim_l), "tta": TTA, "stride": stride,
    }


class MotionSmoothness(ReferenceFreeFrameMetric):
    name = "motion_smoothness"
    higher_is_better = {"motion_smoothness": True, "motion_smoothness_mse": False}

    def __init__(self, device: str = "cuda", tta: bool = False, frame_stride: int = 1,
                 **config: Any) -> None:
        super().__init__(device=device, tta=tta, frame_stride=frame_stride, **config)
        self.tta = bool(tta)
        self._model = self._padder = None

    def setup(self) -> None:
        self._model, self._padder = build_vfimamba(self.device)

    def score_frames(self, pred: np.ndarray, ref, text) -> dict[str, float]:
        tmp = Path(tempfile.mkdtemp(prefix="motion_smoothness_"))
        try:
            paths = write_frames(pred, tmp, stride=self.frame_stride)
            r = motion_smoothness(paths, self._model, self._padder, self.device, TTA=self.tta)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return {"motion_smoothness": r["ssim"] * 100.0, "motion_smoothness_mse": r["mse"],
                "n_triples": float(r["n"])}


__all__ = ["build_vfimamba", "motion_smoothness", "MotionSmoothness"]
