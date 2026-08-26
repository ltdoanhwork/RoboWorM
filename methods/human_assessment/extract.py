"""Extract optical-flow magnitudes + keypoint trajectories for an episode."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from .keypoints import track_keypoints
    from .optical_flow import FlowConfig, compute_flow_series
    from .utils import EpisodeReader
except ImportError:
    from keypoints import track_keypoints
    from optical_flow import FlowConfig, compute_flow_series
    from utils import EpisodeReader
import json

JSON_PATH = Path("seed_points.json")

def load_seed_points(episode_name: str) -> dict[str, np.ndarray]:
    if not JSON_PATH.exists():
        raise FileNotFoundError(
            f"Chưa tìm thấy file {JSON_PATH}. Vui lòng chạy annotate_episode.py trước!"
        )

    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    if episode_name not in data:
        raise KeyError(
            f"Chưa có seed points cho '{episode_name}' trong {JSON_PATH}. "
            f"Hãy chạy: python annotate_episode.py <đường_dẫn_episode>"
        )

    pts_dict = data[episode_name]
    return {
        "view1": np.array(pts_dict["view1"], dtype=np.float32),
        "view2": np.array(pts_dict["view2"], dtype=np.float32),
    }

def extract_episode(
    episode_dir: str | Path,
    pts: dict[str, np.ndarray] | None = None,
    *,
    flow_cfg: FlowConfig | None = None,
    out_path: str | Path | None = None,
) -> Path:
    episode_dir = Path(episode_dir)
    name = episode_dir.name

    if pts is None:
        pts = load_seed_points(name)

    reader = EpisodeReader(episode_dir)
    views = reader.views
    t = len(reader)

    payload: dict = {
        "episode": name,
        "frame_ids": np.array(reader.frame_ids, dtype=np.int32),
        "views": np.array(list(views)),
    }

    for view in views:
        bgr = [reader.bgr(i, view) for i in range(t)]
        gray = [reader.gray(i, view) for i in range(t)]

        mag_u8, scales, stride = compute_flow_series(gray, flow_cfg)
        traj, alive = track_keypoints(bgr, pts[view])

        payload[f"{view}_flow_mag"] = mag_u8
        payload[f"{view}_flow_scale"] = scales
        payload[f"{view}_flow_stride"] = np.int32(stride)
        payload[f"{view}_traj"] = traj
        payload[f"{view}_alive"] = alive
        payload[f"{view}_pts0"] = np.asarray(pts[view], dtype=np.float32)

    if out_path is None:
        out_path = episode_dir / "features.npz"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    print(f"Saved features → {out_path}  (T={t})")
    return out_path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Extract flow + keypoints for one episode")
    p.add_argument(
        "episode_dir",
        nargs="?",
        default=(
            "/pfss/mlde/workspaces/mlde_wsp_MGPATH/thanhnv/"
            "WorldModelBenchmark/human_assessment_feature/test_data/frames/real/episode_000855"
        ),
    )
    p.add_argument("-o", "--out", type=str, default=None)
    args = p.parse_args()
    extract_episode(args.episode_dir, out_path=args.out)
