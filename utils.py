"""Load multiview episode frames (view1/view2 + frame_*.png layout)."""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

__all__ = [
    "discover_episode_dirs",
    "intersect_frames",
    "list_frame_paths",
    "load_frame_bgr",
    "load_frame_gray",
    "frame_path",
    "EpisodeReader",
]


def list_frame_paths(view_dir: str | Path) -> list[Path]:
    p = Path(view_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"Missing view folder: {view_dir}")
    return sorted(p.glob("frame_*.png"))


def frame_index(path: Path) -> int:
    m = re.search(r"frame_(\d+)\.png$", path.name)
    if not m:
        raise ValueError(f"Unexpected frame filename: {path.name}")
    return int(m.group(1))


def intersect_frames(views: list[str], episode_dir: str | Path) -> list[int]:
    per_view = []
    for v in views:
        frames = list_frame_paths(Path(episode_dir) / v)
        per_view.append({frame_index(f) for f in frames})
    common = set.intersection(*per_view) if per_view else set()
    return sorted(common)


def discover_episode_dirs(root_dir: str | Path, views: list[str]) -> list[tuple[str, str]]:
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Missing episodes root: {root_dir}")
    episodes: list[tuple[str, str]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if not all((child / v).is_dir() for v in views):
            continue
        episode_dir = str(child.resolve())
        try:
            if not intersect_frames(list(views), episode_dir):
                continue
        except FileNotFoundError:
            continue
        episodes.append((child.name, episode_dir))
    return episodes


def frame_path(episode_dir: Path | str, view: str, frame_id: int) -> Path:
    return Path(episode_dir) / view / f"frame_{frame_id:04d}.png"


def load_frame_bgr(path: Path | str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def load_frame_gray(path: Path | str) -> np.ndarray:
    img = load_frame_bgr(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


class EpisodeReader:
    """Sequential access to synchronized frames across views."""

    def __init__(
        self,
        episode_dir: Path | str,
        views: tuple[str, str] = ("view1", "view2"),
        *,
        stride: int = 1,
        max_frames: int | None = None,
    ) -> None:
        self.episode_dir = Path(episode_dir)
        self.views = views
        self.stride = max(1, int(stride))
        frame_ids = intersect_frames(list(views), self.episode_dir)
        if max_frames is not None:
            frame_ids = frame_ids[: int(max_frames)]
        self.frame_ids = frame_ids[:: self.stride]
        if not self.frame_ids:
            raise ValueError(f"No common frames under {self.episode_dir}")

    def __len__(self) -> int:
        return len(self.frame_ids)

    def bgr(self, index: int, view: str) -> np.ndarray:
        fid = self.frame_ids[index]
        return load_frame_bgr(frame_path(self.episode_dir, view, fid))

    def gray(self, index: int, view: str) -> np.ndarray:
        return cv2.cvtColor(self.bgr(index, view), cv2.COLOR_BGR2GRAY)

    def frame_id(self, index: int) -> int:
        return self.frame_ids[index]
