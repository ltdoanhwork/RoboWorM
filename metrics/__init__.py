"""RoboWorM evaluation metrics (paper App. B).

Unified interface (``metrics/base.py``)::

    from metrics import Sample, build_metric
    m = build_metric("fidelity", device="cuda")          # psnr + ssim + lpips
    res = m.evaluate([Sample(id="ep0", pred="ep0.mp4", ref="ep0_dense.mp4")])
    res.summary["psnr"]["mean"]; res.save("out/fidelity.json")

Registered metrics (``metrics.registry.available()``):
  kiva               ours, kinescore (external launcher)          metrics/kiva/
  rigis              ours, VGGT-Omega (external launcher)         metrics/rigis/
  visual_integrity   CLIP-IQA+ p10 x100   (Wang et al., 2023)     metrics/visual_integrity.py
  text_alignment     CLIPScore            (Hessel et al., 2021)   metrics/text_alignment.py
  fvd                I3D Frechet dist.    (Unterthiner et al., 2018) metrics/fvd/
  psnr ssim lpips    vs the dense rollout (piq)                   metrics/fidelity.py
  motion_smoothness  RQ3 baseline MS      (Duan et al., 2025)     metrics/motion_smoothness.py
  motion_amplitude   RQ3 baseline MAS     (Deng et al., 2026)     metrics/motion_amplitude.py

CLI: ``python -m metrics.evaluate --help``. Details and citations: ``metrics/README.md``.
Everything is imported lazily.
"""
from __future__ import annotations

import importlib

_LAZY = {
    "BaseMetric": "metrics.base",
    "FrameMetric": "metrics.base",
    "ReferenceFreeFrameMetric": "metrics.base",
    "ExternalMetric": "metrics.base",
    "Sample": "metrics.base",
    "MetricResult": "metrics.base",
    "build_metric": "metrics.registry",
    "get_metric_class": "metrics.registry",
    "register": "metrics.registry",
    "available": "metrics.registry",
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'metrics' has no attribute {name!r}")
