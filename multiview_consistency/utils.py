import os
import matplotlib.pyplot as plt
import re
from pathlib import Path
import torch
def save_depth_maps(depth_map, output_dir="output_depth", prefix="frame"):
    """
    Lưu depth_map tensor (B, S, H, W, 1) thành các tệp ảnh màu.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Chuyển sang CPU và numpy
    # depth_map shape: (1, 2, 294, 518, 1)
    if isinstance(depth_map, torch.Tensor):
        depth_np = depth_map.detach().cpu().float().numpy()
    else:
        depth_np = depth_map

    
    B, S, H, W, *_ = depth_np.shape
    
    for b in range(B):
        for s in range(S):
            # Lấy map của từng view, loại bỏ chiều batch và channel cuối
            d_map = depth_np[b, s, :, :, 0] if depth_np.ndim == 5 else depth_np[b, s, :, :]
            
            # Chuẩn hóa để hiển thị (Min-Max scaling)
            # Lưu ý: Có thể dùng giá trị cố định (ví dụ 0-10m) nếu muốn so sánh giữa các khung hình
            d_min = d_map.min()
            d_max = d_map.max()
            d_norm = (d_map - d_min) / (d_max - d_min + 1e-8)
            
            # Sử dụng colormap 'magma' hoặc 'viridis' (phổ biến cho depth)
            plt.imsave(
                os.path.join(output_dir, f"{prefix}_batch{b}_view{s+1}.png"),
                d_norm,
                cmap='magma'
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