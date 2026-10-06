"""Name -> metric class, resolved lazily.

Each metric pulls a different heavy stack (piq, pyiqa, torchmetrics+CLIP, WorldScore
third_party, or nothing but a subprocess), so classes are imported only when built:
listing or building the kiva metric never imports torch.

Registry names follow the paper (App. B); the source paper of each metric is in
``metrics/README.md``.
"""
from __future__ import annotations

import importlib
from typing import Any

from .base import BaseMetric

_REGISTRY: dict[str, str | type[BaseMetric]] = {
    # ours
    "kiva": "metrics.kiva.metric:Hallucination",
    "rigis": "metrics.rigis.metric:MultiviewConsistency",
    # reference-free (adapted)
    "visual_integrity": "metrics.visual_integrity:VisualIntegrity",
    "text_alignment": "metrics.text_alignment:TextAlignment",
    # distributional
    "fvd": "metrics.fvd:FVD",
    # reference-based, vs the dense rollout
    "psnr": "metrics.fidelity:PSNR",
    "ssim": "metrics.fidelity:SSIM",
    "lpips": "metrics.fidelity:LPIPS",
    "fidelity": "metrics.fidelity:Fidelity",
    # RQ3 baselines
    "motion_smoothness": "metrics.motion_smoothness:MotionSmoothness",
    "motion_amplitude": "metrics.motion_amplitude:MotionAmplitude",
}

#: deprecated / pre-refactor names -> registry name
ALIASES = {
    "clip_score": "text_alignment", "clip": "text_alignment", "clipscore": "text_alignment",
    "clip_iqa": "visual_integrity", "clipiqa+": "visual_integrity",
    "multiview_consistency": "rigis", "mvc": "rigis", "multiview": "rigis",
    "hallucination": "kiva", "hal": "kiva", "kinescore": "kiva",
    "i3d_fvd": "fvd",
    "ms": "motion_smoothness", "mas": "motion_amplitude",
}


def register(name: str, cls: type[BaseMetric] | None = None):
    """Register a BaseMetric subclass; usable as ``@register("my_metric")``."""
    def deco(c: type[BaseMetric]) -> type[BaseMetric]:
        if not (isinstance(c, type) and issubclass(c, BaseMetric)):
            raise TypeError(f"{c!r} is not a BaseMetric subclass")
        _REGISTRY[name] = c
        return c
    return deco(cls) if cls is not None else deco


def available() -> list[str]:
    return sorted(_REGISTRY)


def get_metric_class(name: str) -> type[BaseMetric]:
    key = ALIASES.get(name, name)
    if key not in _REGISTRY:
        raise KeyError(f"unknown metric {name!r}; available: {', '.join(available())}")
    target = _REGISTRY[key]
    if isinstance(target, str):
        mod, attr = target.split(":")
        target = getattr(importlib.import_module(mod), attr)
        _REGISTRY[key] = target
    return target


def build_metric(name: str, **kwargs: Any) -> BaseMetric:
    return get_metric_class(name)(**kwargs)
