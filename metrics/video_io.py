"""Frame decoding shared by every in-process metric.

All loaders return uint8 RGB (T, H, W, 3). Decoding uses ``imageio.v3`` -- the
same reader ``scripts/_score_vs_dense.py`` uses -- so frames are bit-identical to
what the published numbers were computed on.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".gif")


def load_frames(src) -> np.ndarray:
    if isinstance(src, np.ndarray):
        return _as_rgb_u8(src)
    p = Path(src)
    if p.is_dir():
        return _load_dir(p)
    if p.suffix.lower() in VIDEO_EXTS:
        return _load_video(p)
    if p.suffix.lower() in IMAGE_EXTS:
        import imageio.v3 as iio
        return _as_rgb_u8(np.asarray(iio.imread(p))[None])
    raise ValueError(f"unsupported video source: {src}")


def _load_video(p: Path) -> np.ndarray:
    import imageio.v3 as iio

    if not p.is_file():
        raise FileNotFoundError(p)
    frames = [np.asarray(f) for f in iio.imiter(p)]
    if not frames:
        raise RuntimeError(f"no frames decoded from {p}")
    return _as_rgb_u8(np.stack(frames))


def _natural_key(path: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.name)]


def _load_dir(p: Path) -> np.ndarray:
    import imageio.v3 as iio

    files = sorted((f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS), key=_natural_key)
    if not files:
        raise RuntimeError(f"no image frames in {p}")
    return _as_rgb_u8(np.stack([np.asarray(iio.imread(f)) for f in files]))


def _as_rgb_u8(a: np.ndarray) -> np.ndarray:
    if a.ndim == 3:  # (T, H, W) grayscale
        a = a[..., None]
    if a.ndim != 4:
        raise ValueError(f"expected (T,H,W,C) frames, got shape {a.shape}")
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    elif a.shape[-1] == 4:
        a = a[..., :3]
    if a.dtype != np.uint8:
        if np.issubdtype(a.dtype, np.floating) and a.max() <= 1.0 + 1e-6:
            a = a * 255.0
        a = np.clip(np.rint(a), 0, 255).astype(np.uint8)
    return a


def read_prompt(path: str | Path) -> str | None:
    """Prompt from the ``<episode>.txt`` sidecar the inference scripts write.

    Format::

        [Prompt]
        A robot arm places the cup on the coaster.
        [Negative Prompt]
        ...
    A sidecar without section headers is read whole.
    """
    p = Path(path)
    if not p.is_file():
        return None
    text = p.read_text(errors="replace")
    m = re.search(r"\[Prompt\]\s*\n(.*?)(?:\n\s*\[[^\]]+\]|\Z)", text, flags=re.S)
    out = (m.group(1) if m else text).strip()
    return out or None
