"""Video-world evaluation metrics.

Unified interface (``metrics/base.py``)::

    from metrics import Sample, build_metric
    m = build_metric("fidelity", device="cuda")          # psnr + ssim + lpips
    res = m.evaluate([Sample(id="ep0", pred="ep0.mp4", ref="ep0_gt.mp4")])
    res.summary["psnr"]["mean"]; res.save("out/fidelity.json")

Registered metrics (``metrics.registry.available()``):
  psnr, ssim, lpips, fidelity   piq, reference-based          metrics/fidelity.py
  clip_score                    torchmetrics CLIP ViT-B/16     metrics/clip_score.py
  fvd                           I3D, 16f windows @5fps         metrics/fvd.py
  rigis  (multiview_consistency) VGGT-Omega (external launcher) metrics/rigis/
  kiva   (hallucination)        kinescore  (external launcher) metrics/kiva/

CLI: ``python -m metrics.evaluate --help``. Details: ``metrics/README.md``.

Everything is imported lazily, so ``import metrics.phyworld`` etc. stays cheap.
The legacy helpers (``calculate_psnr`` ...) remain importable from here.
"""
from __future__ import annotations

import importlib

_LAZY = {
    "BaseMetric": "metrics.base",
    "FrameMetric": "metrics.base",
    "ExternalMetric": "metrics.base",
    "Sample": "metrics.base",
    "MetricResult": "metrics.base",
    "build_metric": "metrics.registry",
    "get_metric_class": "metrics.registry",
    "register": "metrics.registry",
    "available": "metrics.registry",
    # legacy function API (kept for scripts/metrics/compute_dr_metrics.py & co.)
    "calculate_psnr": "metrics.video_metrics",
    "get_video_ssim": "metrics.video_metrics",
    "calculate_lpips": "metrics.video_metrics",
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'metrics' has no attribute {name!r}")
