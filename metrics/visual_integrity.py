"""Visual Integrity: CLIP-IQA+ at the 10th percentile over frames, x100.

Source metric
    CLIP-IQA+ -- J. Wang, K. C. K. Chan, C. C. Loy, "Exploring CLIP for Assessing
    the Look and Feel of Images", AAAI 2023. CLIP (Radford et al., 2021) scores
    the antonym prompt pair "Good photo." / "Bad photo."; CLIP-IQA+ learns the
    prompt context (CoOp). Score in [0, 1], higher = better. Backend: ``pyiqa``
    ``create_metric("clipiqa+")``, the implementation WorldScore (Duan et al.,
    2025) uses for its frame-quality aspect.

Adaptation (paper App. B.3.1)
    The source metric averages over frames. We report the **10th percentile**
    over the scored frames, x100: an autoregressive rollout that collapses for a
    few frames keeps a healthy mean, whereas p10 asks whether nearly all frames
    are still usable. The mean is kept as the diagnostic ``visual_integrity_mean``.
    The ground-truth-is-ceiling audit fails for this column, so it is used for the
    acceleration comparison (vs dense), not to rank models.

Per frame: PIL RGB -> Resize((512, 512)) -> ToTensor -> clipiqa+ (WorldScore's
``IQAPytorchMetric`` preprocessing). Frame selection is the paper analysis
protocol of :class:`metrics.base.ReferenceFreeFrameMetric` (5 fps, 256x192,
every 5th analysis frame, JPEG round trip); ``protocol="raw"`` scores a clip as
decoded.

Registered as ``visual_integrity`` (alias ``clip_iqa``).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import ReferenceFreeFrameMetric

RESIZE = 512
PERCENTILE = 10


class CLIPIQAPlus:
    """Frame-level CLIP-IQA+ scorer (pyiqa ``clipiqa+``, Resize 512 -> ToTensor)."""

    def __init__(self, device: str = "cuda", resize: int = RESIZE) -> None:
        import pyiqa
        from torchvision import transforms

        self.device = device
        self.pre = transforms.Compose([transforms.Resize((resize, resize)), transforms.ToTensor()])
        self.model = pyiqa.create_metric("clipiqa+").to(device)

    def tensor(self, img):
        """PIL image or image path -> (1, 3, 512, 512) float tensor on device."""
        from PIL import Image

        if not hasattr(img, "convert"):
            img = Image.open(img)
        return self.pre(img.convert("RGB")).unsqueeze(0).to(self.device)

    def score_tensor(self, t) -> float:
        import torch

        with torch.no_grad():
            return float(self.model(t).item())

    def __call__(self, img) -> float:
        return self.score_tensor(self.tensor(img))


def visual_integrity(scores, percentile: float = PERCENTILE) -> float:
    """Per-frame CLIP-IQA+ scores in [0, 1] -> Visual Integrity in [0, 100]."""
    p = float(np.percentile(np.asarray(scores, dtype=float), percentile))
    return float(max(0.0, min(1.0, p)) * 100.0)


class VisualIntegrity(ReferenceFreeFrameMetric):
    name = "visual_integrity"
    higher_is_better = {"visual_integrity": True, "visual_integrity_mean": True}

    def __init__(self, device: str = "cuda", percentile: float = PERCENTILE,
                 resize: int = RESIZE, **config: Any) -> None:
        super().__init__(device=device, percentile=percentile, resize=resize, **config)
        self.percentile, self.resize = float(percentile), int(resize)
        self._scorer: CLIPIQAPlus | None = None

    def setup(self) -> None:
        self._scorer = CLIPIQAPlus(self.device, self.resize)

    def score_frames(self, pred: np.ndarray, ref, text) -> dict[str, float]:
        scores = [self._scorer(im) for im in self.pil_frames(pred)]
        if not scores:
            raise ValueError("no frames left to score")
        return {"visual_integrity": visual_integrity(scores, self.percentile),
                "visual_integrity_mean": float(max(0.0, min(1.0, np.mean(scores))) * 100.0),
                "clip_iqa_plus_p10": float(np.percentile(scores, self.percentile)),
                "clip_iqa_plus_mean": float(np.mean(scores)),
                "n_frames_scored": float(len(scores))}


__all__ = ["CLIPIQAPlus", "VisualIntegrity", "visual_integrity", "PERCENTILE", "RESIZE"]
