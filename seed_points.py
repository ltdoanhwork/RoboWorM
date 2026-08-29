"""Load seed keypoints from assets/seed_points.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

JSON_PATH = Path(__file__).resolve().parent / "assets" / "seed_points.json"
DEFAULT_DATA_ROOT = Path(
    "/pfss/mlde/workspaces/mlde_wsp_MGPATH/thanhnv/WorldModelBenchmark/data/preprocessing/data_for_web"
)


def load_seed_json() -> dict:
    if not JSON_PATH.exists():
        raise FileNotFoundError(
            f"Chưa tìm thấy file {JSON_PATH}. Vui lòng chạy point_picker_many.py trước!"
        )
    with open(JSON_PATH, "r") as f:
        return json.load(f)


def episode_json_key(episode_dir: Path | str, root: Path | str | None = None) -> str:
    """Relative POSIX path used as the JSON key."""
    episode_dir = Path(episode_dir).resolve()
    if root is not None:
        return episode_dir.relative_to(Path(root).resolve()).as_posix()
    return episode_dir.as_posix()


def resolve_seed_key(
    episode_dir: Path | str,
    data: dict | None = None,
    *,
    root: Path | str | None = None,
) -> str:
    """
    Map an on-disk episode folder to its seed_points.json key.

    Keys look like:
      augment/bimanual/output/multiview/ctrlworld/makovian/episode_...
    """
    if data is None:
        data = load_seed_json()

    episode_dir = Path(episode_dir).resolve()
    root = Path(root or DEFAULT_DATA_ROOT).resolve()

    # 1) Relative path under data root
    try:
        key = episode_dir.relative_to(root).as_posix()
        if key in data:
            return key
    except ValueError:
        pass

    # 2) Absolute path stored as key (legacy)
    key = str(episode_dir)
    if key in data:
        return key

    # 3) Longest suffix match (handles arbitrary parent prefixes)
    posix = episode_dir.as_posix()
    suffix_matches = [k for k in data if posix.endswith(k)]
    if suffix_matches:
        return max(suffix_matches, key=len)

    # 4) Match episode_dir field inside entries
    for k, entry in data.items():
        entry_dir = entry.get("episode_dir")
        if not entry_dir:
            continue
        entry_dir = str(entry_dir).replace("\\", "/")
        if entry_dir == k and posix.endswith(entry_dir):
            return k
        if posix.endswith(entry_dir):
            return k

    # 5) Unique folder-name match
    name = episode_dir.name
    name_matches = [k for k, entry in data.items() if entry.get("name") == name]
    if len(name_matches) == 1:
        return name_matches[0]

    raise KeyError(
        f"Chưa có seed points cho '{episode_dir}' trong {JSON_PATH}. "
        f"Hãy chạy: python point_picker_many.py <data_for_web_root>"
    )


def load_seed_points(
    episode_dir: Path | str,
    *,
    root: Path | str | None = None,
) -> dict[str, np.ndarray]:
    data = load_seed_json()
    key = resolve_seed_key(episode_dir, data, root=root)
    pts_dict = data[key]
    return {
        "view1": np.asarray(pts_dict.get("view1", []), dtype=np.float32).reshape(-1, 2),
        "view2": np.asarray(pts_dict.get("view2", []), dtype=np.float32).reshape(-1, 2),
    }
