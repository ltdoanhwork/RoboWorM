"""Draw optical-flow + keypoint overlays and write side-by-side videos."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    from .optical_flow import dequantize_magnitude
    from .utils import EpisodeReader
except ImportError:
    from optical_flow import dequantize_magnitude
    from utils import EpisodeReader

COLORS = [
    (0, 255, 255),
    (255, 0, 255),
    (0, 255, 0),
    (0, 165, 255),
    (255, 255, 0),
    (255, 192, 203),
]


def flow_heatmap(mag: np.ndarray, out_hw: tuple[int, int], alpha: float = 0.45) -> np.ndarray:
    """Magnitude map → BGR heatmap resized to (H, W)."""
    h, w = out_hw
    if mag.max() > 1e-8:
        norm = np.clip(mag / mag.max() * 255.0, 0, 255).astype(np.uint8)
    else:
        norm = np.zeros_like(mag, dtype=np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    return cv2.resize(color, (w, h), interpolation=cv2.INTER_LINEAR), alpha


def blend(base_bgr: np.ndarray, overlay_bgr: np.ndarray, alpha: float) -> np.ndarray:
    return cv2.addWeighted(base_bgr, 1.0 - alpha, overlay_bgr, alpha, 0)


def draw_trajectories(
    frame: np.ndarray,
    traj: np.ndarray,
    alive: np.ndarray,
    t: int,
    trail: int = 12,
) -> np.ndarray:
    """Draw short trails + current keypoints on a BGR frame (in-place copy)."""
    out = frame.copy()
    k = traj.shape[1]
    t0 = max(0, t - trail)
    for ki in range(k):
        c = COLORS[ki % len(COLORS)]
        for i in range(t0, t):
            if alive[i, ki] and alive[i + 1, ki]:
                a = tuple(np.round(traj[i, ki]).astype(int))
                b = tuple(np.round(traj[i + 1, ki]).astype(int))
                cv2.line(out, a, b, c, 2)
        if alive[t, ki]:
            p = tuple(np.round(traj[t, ki]).astype(int))
            cv2.circle(out, p, 5, c, -1)
            cv2.circle(out, p, 6, (255, 255, 255), 1)
    return out


def render_overlay_video(
    episode_dir: str | Path,
    features_path: str | Path,
    out_path: str | Path,
    *,
    fps: float = 15.0,
    show_flow: bool = True,
    show_keypoints: bool = True,
    trail: int = 12,
) -> Path:
    """Write side-by-side (view1|view2) overlay mp4 from saved features."""
    episode_dir = Path(episode_dir)
    out_path = Path(out_path)
    data = np.load(features_path, allow_pickle=True)
    views = [str(v) for v in data["views"].tolist()]
    frame_ids = data["frame_ids"].tolist()

    reader = EpisodeReader(episode_dir)
    # Align to saved frame_ids
    id_to_idx = {fid: i for i, fid in enumerate(reader.frame_ids)}
    indices = [id_to_idx[fid] for fid in frame_ids if fid in id_to_idx]
    if not indices:
        raise ValueError("No overlapping frames between features and episode")

    sample = reader.bgr(indices[0], views[0])
    h, w = sample.shape[:2]
    n_views = len(views)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w * n_views, h))

    for local_t, idx in enumerate(indices):
        panels = []
        for view in views:
            frame = reader.bgr(idx, view)
            if show_flow:
                mag = dequantize_magnitude(
                    data[f"{view}_flow_mag"][local_t],
                    float(data[f"{view}_flow_scale"][local_t]),
                )
                heat, alpha = flow_heatmap(mag, (h, w))
                frame = blend(frame, heat, alpha)
            if show_keypoints:
                frame = draw_trajectories(
                    frame,
                    data[f"{view}_traj"],
                    data[f"{view}_alive"],
                    local_t,
                    trail=trail,
                )
            panels.append(frame)

        combined = np.concatenate(panels, axis=1)
        cv2.putText(
            combined,
            f"frame {local_t}/{len(indices) - 1}",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        writer.write(combined)

    writer.release()
    print(f"Wrote overlay → {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse

    root = Path(__file__).resolve().parent
    default_ep = root / "test_data/frames/good/episode_000255"

    p = argparse.ArgumentParser(description="Overlay flow + keypoints on episode frames")
    p.add_argument("episode_dir", nargs="?", default=str(default_ep))
    p.add_argument("--features", type=str, default=None, help="features.npz (default: episode_dir/features.npz)")
    p.add_argument("-o", "--out", type=str, default=None)
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--no-flow", action="store_true")
    p.add_argument("--no-keypoints", action="store_true")
    args = p.parse_args()

    feat = args.features or str(Path(args.episode_dir) / "features.npz")
    out = args.out or str(Path(args.episode_dir) / "overlay.mp4")
    render_overlay_video(
        args.episode_dir,
        feat,
        out,
        fps=args.fps,
        show_flow=not args.no_flow,
        show_keypoints=not args.no_keypoints,
    )
