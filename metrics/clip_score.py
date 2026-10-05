"""CLIP text-video alignment score.

Mirrors ``metrics/quality/quality_control_probe.py::IQA.frame_scores`` -- the
WorldScore ``content_alignment`` convention the PhyWorld tables use -- so the
numbers are on the same 0-100 scale as ``clip_score`` there:

  frame -> PIL -> Resize((512, 512)) -> ToTensor -> *255 -> uint8
        -> torchmetrics CLIPScore(openai/clip-vit-base-patch16)
  per-frame score = max(100 * cos(E_img, E_txt), 0); sample = mean over frames.

Frames are scored one at a time (the probe does the same): torchmetrics clamps
the *batch mean*, not each frame, so batching would change the result whenever
a frame scores negative.

``frame_stride`` subsamples frames (1 = all, as in the probe's default).
No reference video is needed; text comes from ``Sample.text`` (the CLI fills it
from the ``<episode>.txt`` sidecar's ``[Prompt]`` section).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import FrameMetric

DEFAULT_CLIP = "openai/clip-vit-base-patch16"


class CLIPScore(FrameMetric):
    name = "clip_score"
    higher_is_better = {"clip_score": True}
    requires_text = True

    def __init__(self, device: str = "cuda", model: str = DEFAULT_CLIP,
                 frame_stride: int = 1, resize: int = 512, **config: Any) -> None:
        super().__init__(device=device, model=model, frame_stride=frame_stride,
                         resize=resize, **config)
        self.model_name = model
        self.frame_stride = max(1, int(frame_stride))
        self.resize = int(resize)
        self._metric = None
        self._pre = None

    def setup(self) -> None:
        from torchmetrics.multimodal.clip_score import CLIPScore as _TMCLIPScore
        from torchvision import transforms

        self._metric = _TMCLIPScore(model_name_or_path=_resolve_local(self.model_name)).to(self.device)
        self._pre = transforms.Compose([transforms.Resize((self.resize, self.resize)),
                                        transforms.ToTensor()])

    def score_frames(self, pred: np.ndarray, ref, text: str | None) -> dict[str, float]:
        import torch
        from PIL import Image

        scores = []
        with torch.no_grad():
            for frame in pred[::self.frame_stride]:
                t = self._pre(Image.fromarray(frame)).to(self.device)
                u8 = (t * 255).to(torch.uint8)
                scores.append(float(self._metric(u8, text).detach().item()))
                self._metric.reset()
        return {"clip_score": float(np.mean(scores)), "n_frames_scored": float(len(scores))}


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
                print(f"[clip_score] {model} not in HF_HOME={os.environ['HF_HOME']}; using {path}")
            return path
    return model
