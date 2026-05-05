from __future__ import annotations

import argparse
import fnmatch
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from finetuning.config import (
    AIRBOT_GROUP_A,
    AIRBOT_GROUP_B,
    LEJU_GROUP,
    FinetuneConfig,
    add_config_arguments,
    config_from_args,
)


@dataclass
class EpisodeRecord:
    dataset_name: str
    embodiment: str
    embodiment_id: int
    episode_index: int
    split: str
    length: int
    task: str
    parquet_path: str
    camera_keys: list[str | None]
    video_paths: list[str | None]
    fps: int
    state_dim: int
    action_dim: int
    eef_action_dim: int


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_info(dataset_dir: Path) -> dict[str, Any]:
    info_path = dataset_dir / "meta" / "info.json"
    if info_path.exists():
        return json.loads(info_path.read_text(encoding="utf-8"))
    yaml_path = dataset_dir / "info.yaml"
    if yaml_path.exists():
        try:
            import yaml

            return yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}
    return {}


def detect_camera_layout(dataset_dir: Path, info: dict[str, Any]) -> list[str | None]:
    features = info.get("features", {})
    feature_cams = {k for k in features if k.startswith("observation.images.")}
    if set(AIRBOT_GROUP_A).issubset(feature_cams):
        return list(AIRBOT_GROUP_A)
    if set(AIRBOT_GROUP_B).issubset(feature_cams):
        return list(AIRBOT_GROUP_B)
    leju_available = {k for k in LEJU_GROUP if k is not None}
    if leju_available.issubset(feature_cams):
        return list(LEJU_GROUP)

    videos_root = dataset_dir / "videos"
    cam_dirs: set[str] = set()
    for chunk_dir in sorted(videos_root.glob("chunk-*")):
        if chunk_dir.is_dir():
            cam_dirs = {p.name for p in chunk_dir.iterdir() if p.is_dir()}
            break
    if set(AIRBOT_GROUP_A).issubset(cam_dirs):
        return list(AIRBOT_GROUP_A)
    if set(AIRBOT_GROUP_B).issubset(cam_dirs):
        return list(AIRBOT_GROUP_B)
    if leju_available.issubset(cam_dirs):
        return list(LEJU_GROUP)
    raise ValueError(f"Could not detect supported camera layout for {dataset_dir}")


def detect_embodiment(dataset_name: str, info: dict[str, Any]) -> str:
    robot_name = info.get("robot_name")
    if isinstance(robot_name, list) and robot_name:
        return str(robot_name[0])
    if dataset_name.lower().startswith("airbot") or dataset_name.upper().startswith("AIRBOT"):
        return "Airbot_MMK2"
    if dataset_name.lower().startswith("leju"):
        return "Leju"
    return dataset_name.split("_")[0]


def include_dataset(name: str, include_pattern: str, exclude_pattern: str) -> bool:
    includes = [p.strip() for p in include_pattern.split(",") if p.strip()]
    excludes = [p.strip() for p in exclude_pattern.split(",") if p.strip()]
    if includes and not any(fnmatch.fnmatch(name, pat) for pat in includes):
        return False
    if excludes and any(fnmatch.fnmatch(name, pat) for pat in excludes):
        return False
    return True


def episode_task_map(dataset_dir: Path) -> dict[int, str]:
    tasks = {}
    for row in load_jsonl(dataset_dir / "meta" / "tasks.jsonl"):
        idx = row.get("task_index", row.get("index"))
        task = row.get("task", row.get("description"))
        if idx is not None and task:
            tasks[int(idx)] = str(task)

    out = {}
    for row in load_jsonl(dataset_dir / "meta" / "episodes.jsonl"):
        ep = row.get("episode_index")
        if ep is None:
            continue
        if row.get("tasks"):
            out[int(ep)] = str(row["tasks"][0])
        elif row.get("task_index") is not None:
            out[int(ep)] = tasks.get(int(row["task_index"]), "")
        else:
            out[int(ep)] = str(row.get("description", ""))
    return out


def find_episode_parquet(dataset_dir: Path, episode_index: int) -> Path | None:
    for path in sorted((dataset_dir / "data").glob(f"chunk-*/episode_{episode_index:06d}.parquet")):
        return path
    return None


def find_video(dataset_dir: Path, camera_key: str | None, episode_index: int) -> Path | None:
    if camera_key is None:
        return None
    for path in sorted((dataset_dir / "videos").glob(f"chunk-*/{camera_key}/episode_{episode_index:06d}.mp4")):
        return path
    return None


def infer_dims(parquet_path: Path, cfg: FinetuneConfig) -> tuple[int, int, int]:
    df = pd.read_parquet(parquet_path, columns=["observation.state", "action", cfg.action_key])
    state_dim = len(df["observation.state"].iloc[0])
    action_dim = len(df["action"].iloc[0])
    eef_action_dim = len(df[cfg.action_key].iloc[0])
    return state_dim, action_dim, eef_action_dim


def build_manifest(cfg: FinetuneConfig) -> dict[str, Any]:
    root = Path(cfg.dataset_root)
    if not root.exists():
        raise FileNotFoundError(root)

    datasets = [p for p in sorted(root.iterdir()) if p.is_dir() and include_dataset(p.name, cfg.include_pattern, cfg.exclude_pattern)]
    records: list[EpisodeRecord] = []
    embodiment_to_id: dict[str, int] = {}

    for dataset_dir in datasets:
        info = read_info(dataset_dir)
        try:
            camera_keys = detect_camera_layout(dataset_dir, info)
        except ValueError as exc:
            print(f"[manifest] skip {dataset_dir.name}: {exc}")
            continue

        embodiment = detect_embodiment(dataset_dir.name, info)
        if embodiment not in embodiment_to_id:
            embodiment_to_id[embodiment] = len(embodiment_to_id)
        embodiment_id = embodiment_to_id[embodiment]

        tasks = episode_task_map(dataset_dir)
        episode_indices = sorted(tasks.keys())
        if cfg.max_episodes_per_dataset is not None:
            episode_indices = episode_indices[: cfg.max_episodes_per_dataset]
        if not episode_indices:
            print(f"[manifest] skip {dataset_dir.name}: no episodes")
            continue

        val_count = max(1, int(round(len(episode_indices) * cfg.val_ratio)))
        val_set = set(episode_indices[-val_count:])
        fps = int(info.get("fps") or info.get("statistics", {}).get("fps") or cfg.source_fps)
        dims_cache: tuple[int, int, int] | None = None

        for ep in episode_indices:
            parquet_path = find_episode_parquet(dataset_dir, ep)
            if parquet_path is None:
                print(f"[manifest] missing parquet {dataset_dir.name} episode {ep}")
                continue
            video_paths = [find_video(dataset_dir, cam, ep) for cam in camera_keys]
            if any(cam is not None and video is None for cam, video in zip(camera_keys, video_paths, strict=True)):
                print(f"[manifest] missing camera video {dataset_dir.name} episode {ep}")
                continue
            if dims_cache is None:
                dims_cache = infer_dims(parquet_path, cfg)
            try:
                length = int(pd.read_parquet(parquet_path, columns=["frame_index"]).shape[0])
            except Exception:
                length = int(info.get("statistics", {}).get("total_frames", 0))
            records.append(
                EpisodeRecord(
                    dataset_name=dataset_dir.name,
                    embodiment=embodiment,
                    embodiment_id=embodiment_id,
                    episode_index=ep,
                    split="val" if ep in val_set else "train",
                    length=length,
                    task=tasks.get(ep, ""),
                    parquet_path=str(parquet_path),
                    camera_keys=list(camera_keys),
                    video_paths=[str(p) if p is not None else None for p in video_paths],
                    fps=fps,
                    state_dim=dims_cache[0],
                    action_dim=dims_cache[1],
                    eef_action_dim=dims_cache[2],
                )
            )

    return {
        "config": cfg.to_dict(),
        "embodiments": embodiment_to_id,
        "records": [asdict(r) for r in records],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a RoboCOIN multiview training manifest.")
    add_config_arguments(parser)
    parser.add_argument("--output", default=None, help="Defaults to --manifest-path.")
    args = parser.parse_args()
    cfg = config_from_args(args)
    manifest = build_manifest(cfg)
    output = Path(args.output or cfg.manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[manifest] wrote {len(manifest['records'])} episodes to {output}")
    for embodiment, idx in manifest["embodiments"].items():
        print(f"[manifest] embodiment {idx}: {embodiment}")


if __name__ == "__main__":
    main()

