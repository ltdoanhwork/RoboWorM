"""Prepare batch_input.json for DreamGen inference from a local LeRobot DROID dataset."""

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset

CAM_KEYS = [
    "observation.images.exterior_1_left",
    "observation.images.exterior_2_left",
    "observation.images.wrist_left",
]


def create_grid(images):
    if not images:
        return Image.new("RGB", (1280, 720))
    w, h = images[0].size
    grid = Image.new("RGB", (w * 2, h * 2))
    grid.paste(images[0], (0, 0))
    if len(images) > 1:
        grid.paste(images[1], (w, 0))
    if len(images) > 2:
        grid.paste(images[2], (0, h))
    if len(images) > 3:
        grid.paste(images[3], (w, h))
    return grid


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_id", default="lerobot/droid_1.0.1")
    parser.add_argument("--dataset_path", default="./datasets/droid_20p")
    parser.add_argument("--save_dir", default="./sampling_dataset/droid_20p")
    parser.add_argument("--max_episodes", type=int, default=None)
    return parser.parse_args()


def configure_offline_hf(dataset_path: Path):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(dataset_path.resolve())


def load_available_episodes(dataset_path: Path, cam_keys: list[str], max_episodes: int | None) -> list[int]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to inspect local LeRobot metadata") from exc

    episode_files = sorted((dataset_path / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"No episode metadata parquet files found under {dataset_path / 'meta' / 'episodes'}")

    columns = ["episode_index"]
    for key in cam_keys:
        columns.extend([f"videos/{key}/chunk_index", f"videos/{key}/file_index"])

    frames = [pd.read_parquet(path, columns=columns) for path in episode_files]
    episodes_df = pd.concat(frames, ignore_index=True)

    available: list[int] = []
    for row in episodes_df.itertuples(index=False, name=None):
        row_by_col = dict(zip(columns, row))
        episode_index = int(row_by_col["episode_index"])

        has_all_videos = True
        for key in cam_keys:
            chunk_index = int(row_by_col[f"videos/{key}/chunk_index"])
            file_index = int(row_by_col[f"videos/{key}/file_index"])
            video_path = (
                dataset_path
                / "videos"
                / key
                / f"chunk-{chunk_index:03d}"
                / f"file-{file_index:03d}.mp4"
            )
            if not video_path.exists():
                has_all_videos = False
                break

        if has_all_videos:
            available.append(episode_index)
            if max_episodes is not None and len(available) >= max_episodes:
                break

    if not available:
        raise RuntimeError(f"No local episodes with all required camera videos were found in {dataset_path}")
    return available


def make_lerobot_dataset(repo_id: str, dataset_path: Path, episodes: list[int]) -> LeRobotDataset:
    init_kwargs = {
        "root": str(dataset_path),
        "episodes": episodes,
        "download_videos": False,
        "local_files_only": True,
    }
    supported_params = inspect.signature(LeRobotDataset.__init__).parameters
    filtered_kwargs = {key: value for key, value in init_kwargs.items() if key in supported_params}
    return LeRobotDataset(repo_id, **filtered_kwargs)


def get_dataset_frame(ds: LeRobotDataset, absolute_index: int):
    absolute_to_relative = getattr(ds, "_absolute_to_relative_idx", None)
    if absolute_to_relative is None:
        return ds[absolute_index]
    return ds[absolute_to_relative[absolute_index]]


def tensor_to_image(img_tensor) -> Image.Image:
    if hasattr(img_tensor, "detach"):
        img_tensor = img_tensor.detach().cpu()
    img_np = np.asarray(img_tensor)
    if img_np.ndim == 3 and img_np.shape[0] in {1, 3, 4}:
        img_np = np.transpose(img_np, (1, 2, 0))
    if np.issubdtype(img_np.dtype, np.floating) and img_np.size and float(np.nanmax(img_np)) <= 1.0:
        img_np = img_np * 255
    img_np = img_np.clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_np)


def extract_instruction(frame) -> str:
    for key in ("language_instruction", "task", "language_instruction_2", "language_instruction_3"):
        value = frame.get(key)
        if value is None:
            continue
        text = str(value).strip().rstrip(".")
        if text:
            return text
    return "performing a manipulation task"


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    screenshot_dir = save_dir / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {dataset_path} (local files only)...")
    configure_offline_hf(dataset_path)

    try:
        available_episodes = load_available_episodes(dataset_path, CAM_KEYS, args.max_episodes)
        print(f"Found {len(available_episodes)} local episodes with all required camera videos.")
        print("Initializing LeRobotDataset for the selected episodes...", flush=True)
        ds = make_lerobot_dataset(args.repo_id, dataset_path, available_episodes)
    except Exception as e:
        print(f"\n[!] Error loading local dataset: {e}", file=sys.stderr)
        raise SystemExit(1) from e

    ep_table = ds.meta.episodes
    print(f"Processing {len(available_episodes)} episodes...")

    batch_input = []
    skipped = 0
    for episode_index in tqdm(available_episodes, desc="Preparing batch"):
        try:
            start_idx = int(ep_table[episode_index]["dataset_from_index"])
            frame = get_dataset_frame(ds, start_idx)

            images = []
            for k in CAM_KEYS:
                if k in frame:
                    images.append(tensor_to_image(frame[k]))

            grid = create_grid(images)
            screenshot_path = (screenshot_dir / f"episode_{episode_index:06d}.png").resolve()
            grid.save(screenshot_path)

            prompt = extract_instruction(frame)
            full_prompt = (
                f"A multi-view video shows a robot arm performing the task: {prompt}. "
                "The video is split into four views: The top-left view shows the robotic arm "
                "from the left side, the top-right view shows it from the right side, "
                "the bottom-left view shows a first-person perspective from the robot's "
                "end-effector (gripper), and the bottom-right view is a black screen (inactive view). "
                f"The robot arm performs: {prompt}."
            )

            batch_input.append({
                "input_video": str(screenshot_path),
                "prompt": full_prompt,
                "output_video": str(Path(f"./output/droid_20p/episode_{episode_index:06d}_generated.mp4").resolve()),
            })
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                tqdm.write(f"[!] Skipping episode {episode_index}: {e}")
            continue

    if not batch_input:
        print("\n[!] No batch inputs were created.", file=sys.stderr)
        raise SystemExit(1)

    with open(save_dir / "batch_input.json", "w", encoding="utf-8") as f:
        json.dump(batch_input, f, indent=4)

    print(f"\nDone! Created {len(batch_input)} items in {save_dir / 'batch_input.json'}")
    if skipped:
        print(f"Skipped {skipped} episodes that failed during screenshot extraction.")


if __name__ == "__main__":
    main()
