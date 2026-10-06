"""Frame decoding shared by every in-process metric.

All loaders return uint8 RGB (T, H, W, 3). Decoding uses ``imageio.v3`` -- the
same reader ``scripts/metrics/_score_vs_dense.py`` uses -- so frames are bit-identical to
what the published numbers were computed on.

``nr_analysis_frames`` is the separate (cv2-based) preprocessing of the
reference-free metrics: 5 fps, conditioning frame dropped, one camera tile,
256x192.
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


# --------------------------------------------------------------------------- #
# reference-free analysis protocol (Visual Integrity, Text Alignment, MS, MAS)
# --------------------------------------------------------------------------- #
#: fixed analysis grid every reference-free metric is scored on (w, h)
NR_ANALYSIS_SIZE = (256, 192)
#: frames per second the generated clip is decimated to before scoring
NR_ANALYSIS_FPS = 5.0


def nr_analysis_frames(path: str | Path, layout: str = "single", view_index: int = 0,
                       n_views: int = 1, fps: float = NR_ANALYSIS_FPS,
                       size: tuple[int, int] = NR_ANALYSIS_SIZE):
    """Decode one camera of a rollout onto the reference-free analysis grid.

    This is the preprocessing behind every reference-free number in the paper
    (``scripts/metrics/run_dense_nr.py::prepare`` calls it):

    * decimate by the integer stride ``round(src_fps / fps)`` (never interpolate);
    * drop frame 0, the conditioning frame (it is copied from the input, not generated);
    * cut the camera tile out of a mosaic: ``layout="strip"`` (Ctrl-World 1xN) or
      ``"grid"`` (DreamGen 2x2), tile ``view_index`` of ``n_views``;
    * resize to 256x192 with ``cv2.INTER_AREA`` so every model is scored on one grid.

    Returns ``(frames uint8 RGB (T,H,W,3), info dict)``; raises if fewer than 3
    frames survive.
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        raise ValueError("Invalid FPS")
    interval = max(1, round(src_fps / fps))
    frames, ids = [], []
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i > 0 and (i - 1) % interval == 0:
            h, w = fr.shape[:2]
            if layout == "strip":
                fr = fr[:, view_index * w // n_views:(view_index + 1) * w // n_views]
            elif layout == "grid":
                fr = fr[(view_index // 2) * h // 2:(view_index // 2 + 1) * h // 2,
                        (view_index % 2) * w // 2:(view_index % 2 + 1) * w // 2]
            fr = cv2.resize(fr, size, interpolation=cv2.INTER_AREA)
            frames.append(fr[:, :, ::-1].copy())
            ids.append(i)
        i += 1
    cap.release()
    if len(frames) < 3:
        raise ValueError("Fewer than three post-conditioning frames")
    info = dict(source_fps=src_fps, effective_fps=src_fps / interval, source_interval=interval,
                analysis_width=size[0], analysis_height=size[1], sampled_source_frames=ids)
    return np.stack(frames), info


def jpeg_roundtrip(frame: np.ndarray):
    """uint8 RGB frame -> PIL image after a cv2 JPEG encode (quality 95) + PIL decode.

    The published CLIP-IQA+ / CLIPScore numbers were computed on frames that went
    through ``cv2.imwrite(.jpg)`` -> ``PIL.Image.open``; this reproduces that byte-for-byte
    in memory.
    """
    import io

    import cv2
    from PIL import Image

    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(frame[:, :, ::-1]))
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return Image.open(io.BytesIO(buf.tobytes())).convert("RGB")


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
