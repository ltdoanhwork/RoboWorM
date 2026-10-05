"""Name -> metric class, resolved lazily.

Each metric pulls a different heavy stack (piq, torchmetrics+CLIP, or nothing
but a subprocess), so classes are imported only when built: listing or
building the hallucination metric never imports torch.
"""
from __future__ import annotations

import importlib
from typing import Any

from .base import BaseMetric

_REGISTRY: dict[str, str | type[BaseMetric]] = {
    "psnr": "metrics.fidelity:PSNR",
    "ssim": "metrics.fidelity:SSIM",
    "lpips": "metrics.fidelity:LPIPS",
    "fidelity": "metrics.fidelity:Fidelity",
    "clip_score": "metrics.clip_score:CLIPScore",
    "multiview_consistency": "metrics.multiview_consistency.metric:MultiviewConsistency",
    "hallucination": "metrics.hallucination.metric:Hallucination",
}

ALIASES = {"clip": "clip_score", "mvc": "multiview_consistency", "multiview": "multiview_consistency",
           "hal": "hallucination", "kinescore": "hallucination"}


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
