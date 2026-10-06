"""Text Alignment: CLIPScore between each frame and the task clause, [0, 100].

Source metric
    CLIPScore -- J. Hessel, A. Holtzman, M. Forbes, R. Le Bras, Y. Choi,
    "CLIPScore: A Reference-free Evaluation Metric for Image Captioning",
    EMNLP 2021. ``max(100 * cos(E_I, E_C), 0)`` with CLIP ViT-B/16
    (Radford et al., 2021). Backend: ``torchmetrics`` ``CLIPScore``
    (``openai/clip-vit-base-patch16``), the one WorldScore (Duan et al., 2025)
    uses for its text-control aspect.

Adaptation (paper App. B.3.2)
    * Scored against the **task clause only** (:func:`task_clause`): DreamGen
      prompts append a description of the 2x2 grid layout, which does not match a
      cropped single view and can push the text past CLIP's 77-token window.
    * Averaged over the scored frames.
    * Only defined for text-conditioned models. Image-only models (DreamDojo)
      never receive the instruction and are reported as None, not as a low score;
      real-video anchors count as text-conditioned.
    * CLIPScore has a strong caption-dependent bias, so ranking claims use paired
      deltas under the same prompt.

Per frame: PIL RGB -> Resize((512, 512)) -> ToTensor -> x255 -> uint8 -> CLIPScore,
one frame at a time (torchmetrics clamps the *batch mean*, so batching would
change the result whenever a frame scores negative). Frame selection is the
paper analysis protocol of :class:`metrics.base.ReferenceFreeFrameMetric`;
``protocol="raw"`` scores every decoded frame without the JPEG round trip.

Registered as ``text_alignment`` (deprecated alias ``clip_score``).
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

from .base import ReferenceFreeFrameMetric

DEFAULT_CLIP = "openai/clip-vit-base-patch16"
RESIZE = 512

#: models that receive the instruction (``gt_<model>`` anchors inherit their model)
TEXT_CONDITIONED_MODELS = {"dreamgen", "ctrlworld", "ctrlworld_4view_grid", "gt"}

_TASK_CLAUSE = re.compile(r"(?:performing(?: the task)?:|shows that a robot)\s*(.+?)(?:\s+The video\b|$)",
                          re.S)


def task_clause(instruction: str | None) -> str:
    """Strip a DreamGen / Ctrl-World prompt template down to the task clause.

    ``"... performing the task: put the cup on the coaster. The video shows ..."``
    -> ``"put the cup on the coaster"``. Plain instructions pass through.
    """
    if not instruction:
        return ""
    m = _TASK_CLAUSE.search(instruction)
    return (m.group(1) if m else instruction).strip().rstrip(".").strip()


def text_conditioned(model: str) -> bool:
    """Whether Text Alignment is defined for clips of ``model`` (``gt_*`` anchors included)."""
    return (model[3:] if model.startswith("gt_") else model) in TEXT_CONDITIONED_MODELS


class CLIPScorer:
    """Frame-level CLIPScore (torchmetrics, ViT-B/16, Resize 512 -> uint8)."""

    def __init__(self, device: str = "cuda", model: str = DEFAULT_CLIP, resize: int = RESIZE) -> None:
        from torchmetrics.multimodal.clip_score import CLIPScore as _TMCLIPScore
        from torchvision import transforms

        self.device = device
        self.metric = _TMCLIPScore(model_name_or_path=_resolve_local(model)).to(device)
        self.pre = transforms.Compose([transforms.Resize((resize, resize)), transforms.ToTensor()])

    def tensor(self, img):
        """PIL image or image path -> (3, 512, 512) float tensor on device."""
        from PIL import Image

        if not hasattr(img, "convert"):
            img = Image.open(img)
        return self.pre(img.convert("RGB")).to(self.device)

    def score_tensor(self, t, text: str) -> float:
        import torch

        with torch.no_grad():
            v = float(self.metric((t * 255).to(torch.uint8), text).detach().item())
        self.metric.reset()
        return v

    def __call__(self, img, text: str) -> float:
        return self.score_tensor(self.tensor(img), text)


class TextAlignment(ReferenceFreeFrameMetric):
    name = "text_alignment"
    higher_is_better = {"text_alignment": True}
    requires_text = True

    def __init__(self, device: str = "cuda", model: str = DEFAULT_CLIP, resize: int = RESIZE,
                 use_task_clause: bool = True, **config: Any) -> None:
        super().__init__(device=device, model=model, resize=resize,
                         use_task_clause=use_task_clause, **config)
        self.model_name, self.resize = model, int(resize)
        self.use_task_clause = bool(use_task_clause)
        self._scorer: CLIPScorer | None = None

    def setup(self) -> None:
        self._scorer = CLIPScorer(self.device, self.model_name, self.resize)

    def score_frames(self, pred: np.ndarray, ref, text: str | None) -> dict[str, float]:
        caption = task_clause(text) if self.use_task_clause else (text or "")
        if not caption:
            raise ValueError("empty task clause")
        scores = [self._scorer(im, caption) for im in self.pil_frames(pred)]
        if not scores:
            raise ValueError("no frames left to score")
        return {"text_alignment": float(max(0.0, min(100.0, np.mean(scores)))),
                "n_frames_scored": float(len(scores))}


#: pre-refactor class name
CLIPScore = TextAlignment


def _resolve_local(model: str) -> str:
    """HF id -> local snapshot dir, searching the active cache then ~/.cache.

    Needed because the kinescore launcher exports ``HF_HOME`` into its store;
    once that leaks into a tmux server's global env, every later job looks for
    CLIP in the wrong cache and fails offline. A local path, or an id found in
    neither cache, is returned unchanged (normal HF resolution then applies).
    """
    import os
    from pathlib import Path

    if Path(model).is_dir():
        return model
    from huggingface_hub import snapshot_download

    caches = [None, str(Path.home() / ".cache" / "huggingface" / "hub")]
    for cache_dir in caches:
        try:
            path = snapshot_download(model, cache_dir=cache_dir, local_files_only=True,
                                     allow_patterns=["*.json", "*.txt", "*.bin", "*.safetensors"])
        except Exception:  # noqa: BLE001 -- not in this cache
            continue
        if (Path(path) / "config.json").is_file():
            if cache_dir is not None and os.environ.get("HF_HOME"):
                print(f"[text_alignment] {model} not in HF_HOME={os.environ['HF_HOME']}; using {path}")
            return path
    return model


__all__ = ["CLIPScorer", "TextAlignment", "CLIPScore", "task_clause", "text_conditioned",
           "TEXT_CONDITIONED_MODELS", "DEFAULT_CLIP"]
