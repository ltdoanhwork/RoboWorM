import os
import matplotlib.pyplot as plt
import re
from pathlib import Path
import torch
import numpy as np

def save_depth_maps(depth_map, output_dir="output_depth", prefix="frame", vmin=None, vmax=None):
    """Save depth maps. Expects (B, S, H, W, 1) or (B, S, H, W)."""
    os.makedirs(output_dir, exist_ok=True)
    if isinstance(depth_map, torch.Tensor):
        depth_np = depth_map.detach().cpu().float().numpy()
    else:
        depth_np = np.asarray(depth_map)

    if depth_np.ndim == 5:
        depth_np = depth_np[..., 0]  # (B, S, H, W)

    B, S = depth_np.shape[:2]
    for b in range(B):
        for s in range(S):
            plt.imsave(
                os.path.join(output_dir, f"{prefix}_batch{b}_view{s+1}.png"),
                np.clip(depth_np[b, s], vmin, vmax),
                vmin=vmin,
                vmax=vmax,
                cmap="magma",
            )


def save_rgb_maps(rgb_map, output_dir="output_depth", prefix="rgb"):
    """Save RGB maps. Expects (B, S, H, W, 3) float in [0, 1]."""
    os.makedirs(output_dir, exist_ok=True)
    if isinstance(rgb_map, torch.Tensor):
        rgb_np = rgb_map.detach().cpu().float().numpy()
    else:
        rgb_np = np.asarray(rgb_map, dtype=np.float32)

    rgb_u8 = (np.clip(rgb_np, 0.0, 1.0) * 255.0).astype(np.uint8)
    B, S = rgb_u8.shape[:2]
    for b in range(B):
        for s in range(S):
            plt.imsave(
                os.path.join(output_dir, f"{prefix}_batch{b}_view{s+1}.png"),
                rgb_u8[b, s],
            )


def list_frame_paths(view_dir: str) -> list[Path]:
    """
    Return sorted frame paths: frame_0000.png, frame_0001.png, ...
    """
    p = Path(view_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"Missing view folder: {view_dir}")
    frames = sorted(p.glob("frame_*.png"))
    return frames


def frame_index(path: Path) -> int:
    m = re.search(r"frame_(\d+)\.png$", path.name)
    if not m:
        raise ValueError(f"Unexpected frame filename: {path.name}")
    return int(m.group(1))


def intersect_frames(views: list[str], episode_dir: str) -> list[int]:
    """
    Find common frame indices across all views.
    """
    per_view = []
    for v in views:
        frames = list_frame_paths(os.path.join(episode_dir, v))
        per_view.append({frame_index(f) for f in frames})
    common = set.intersection(*per_view) if per_view else set()
    return sorted(common)


def discover_episode_dirs(root_dir: str, views: list[str]) -> list[tuple[str, str]]:
    """
    List episode folders under ``root_dir`` that contain all ``views`` with common frames.

    Returns:
        Sorted list of ``(folder_name, absolute_episode_dir)``.
    """
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