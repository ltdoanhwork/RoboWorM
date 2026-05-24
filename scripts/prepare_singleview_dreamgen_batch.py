"""Prepare batch_input.json for DreamGen singleview inference on all 3 embodiments.

For each episode, extracts the first frame as a conditioning image and pairs it
with the training prompt. Groups episodes by required num_chunks (based on episode
duration at 16fps) and generates separate batch_input_{N}chunks.json files.

Supports:
  - humanoid:   datasets/humanoid/singleview/{makovian,non_makovian}
  - single_arm: datasets/single_arm/singleview/{makovian,non_makovian}
  - bimanual:   datasets/bimanual/singleview/{makovian,non_makovian}/{task_name}

Output layout:
  sampling_dataset/dense/{embodiment}/input/singleview/dreamgen/{split}/
    first_frames/episode_XXXXXX.png
    batch_input_1chunks.json
    batch_input_2chunks.json
    ...

Usage:
    python scripts/prepare_singleview_dreamgen_batch.py
    python scripts/prepare_singleview_dreamgen_batch.py --embodiments humanoid single_arm
    python scripts/prepare_singleview_dreamgen_batch.py --max-episodes 5
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FRAMES_PER_CHUNK = 93
FRAMES_PER_ADDITIONAL_CHUNK = 92  # 93 - 1 conditioning frame overlap
MODEL_FPS = 16

HUMANOID_CAM = "observation.images.ego_view_freq20"
SINGLE_ARM_CAM = "observation.images.wrist_left"
BIMANUAL_CAM = "observation.images.cam_high"

HUMANOID_GENERIC_CAPTION = "A GR-1 humanoid robot performing a manipulation task."
SINGLE_ARM_GENERIC_CAPTION = "A robot arm performing a manipulation task."


def get_video_frame_count(video_path: Path) -> int | None:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "stream=nb_frames",
         "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        try:
            return int(result.stdout.strip().split("\n")[0])
        except ValueError:
            pass
    return None


def compute_num_chunks(source_frames: int, source_fps: float) -> int:
    """Compute num_chunks to match task duration at MODEL_FPS."""
    target_frames = source_frames * (MODEL_FPS / source_fps)
    if target_frames <= FRAMES_PER_CHUNK:
        return 1
    return math.ceil((target_frames - FRAMES_PER_CHUNK) / FRAMES_PER_ADDITIONAL_CHUNK) + 1


def extract_first_frame(video_path: Path, out_png: Path, width: int, height: int) -> bool:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"scale={width}:{height}",
            "-vframes", "1",
            "-q:v", "2",
            str(out_png),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0 and out_png.exists()


def load_humanoid_captions(split_dir: Path) -> dict[int, str]:
    captions_dir = split_dir / "captions"
    ep_caption: dict[int, str] = {}
    if not captions_dir.exists():
        return ep_caption
    for txt_file in captions_dir.glob("*.txt"):
        ep_match = re.search(r"(\d+)$", txt_file.stem)
        if ep_match:
            ep_caption[int(ep_match.group(1))] = txt_file.read_text().strip()
    return ep_caption


def load_single_arm_tasks(split_dir: Path) -> dict[int, str]:
    episodes_file = split_dir / "meta" / "episodes.jsonl"
    tasks_file = split_dir / "meta" / "tasks.jsonl"
    ep_task: dict[int, str] = {}
    if not episodes_file.exists():
        return ep_task

    with episodes_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ep_idx = rec.get("episode_index", -1)
            tasks = rec.get("tasks", [])
            if ep_idx >= 0 and tasks and tasks[0]:
                ep_task[ep_idx] = tasks[0]

    if ep_task:
        return ep_task

    task_map: dict[int, str] = {}
    if tasks_file.exists():
        with tasks_file.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                task_desc = rec.get("task", "")
                if task_desc:
                    task_map[rec["task_index"]] = task_desc

    if task_map and episodes_file.exists():
        with episodes_file.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ep_idx = rec.get("episode_index", -1)
                t_idx = rec.get("task_index", -1)
                if ep_idx >= 0 and t_idx in task_map:
                    ep_task[ep_idx] = task_map[t_idx]

    return ep_task


def find_episodes_humanoid(split_dir: Path) -> list[tuple[int, Path]]:
    videos_root = split_dir / "videos"
    results = []
    for mp4 in sorted(videos_root.glob(f"chunk-*/{HUMANOID_CAM}/*.mp4")):
        ep_match = re.search(r"(\d+)$", mp4.stem)
        if ep_match:
            results.append((int(ep_match.group(1)), mp4))
    return results


def find_episodes_single_arm(split_dir: Path) -> list[tuple[int, Path]]:
    videos_root = split_dir / "videos"
    results = []
    for mp4 in sorted(videos_root.glob(f"chunk-*/{SINGLE_ARM_CAM}/*.mp4")):
        ep_match = re.search(r"(\d+)$", mp4.stem)
        if ep_match:
            results.append((int(ep_match.group(1)), mp4))
    return results


def find_episodes_bimanual(split_dir: Path) -> list[tuple[int, str, Path]]:
    """Returns (episode_id, task_name, video_path) tuples."""
    results = []
    for task_dir in sorted(split_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name
        videos_root = task_dir / "videos"
        if not videos_root.exists():
            continue
        for mp4 in sorted(videos_root.glob(f"chunk-*/{BIMANUAL_CAM}/*.mp4")):
            ep_match = re.search(r"(\d+)$", mp4.stem)
            if ep_match:
                results.append((int(ep_match.group(1)), task_name, mp4))
    return results


def bimanual_task_to_caption(task_name: str) -> str:
    name = task_name.replace("_", " ")
    name = name[0].upper() + name[1:] if name else name
    return f"A bimanual robot performing the task: {name}."


def prepare_humanoid(
    dataset_root: Path, save_root: Path, output_root: Path,
    splits: list[str], max_episodes: int | None, skip_existing: bool,
    iter_name: str,
) -> None:
    source_fps = 20.0
    for split in splits:
        split_dir = dataset_root / split
        if not split_dir.exists():
            print(f"[humanoid/{split}] Directory not found, skipping.")
            continue

        input_dir = save_root / "humanoid" / "input" / "singleview" / "dreamgen" / split
        output_dir = output_root / "humanoid" / "output" / "singleview" / "dreamgen" / split / iter_name
        frames_dir = input_dir / "first_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        episodes = find_episodes_humanoid(split_dir)
        captions = load_humanoid_captions(split_dir)

        if max_episodes:
            episodes = episodes[:max_episodes]

        chunks_groups: dict[int, list] = defaultdict(list)
        for ep_id, video_path in episodes:
            frame_png = frames_dir / f"episode_{ep_id:06d}.png"
            if not (skip_existing and frame_png.exists()):
                if not extract_first_frame(video_path, frame_png, 768, 432):
                    print(f"  [WARN] Failed to extract frame: {video_path.name}")
                    continue

            n_frames = get_video_frame_count(video_path)
            n_chunks = compute_num_chunks(n_frames, source_fps) if n_frames else 2

            caption = captions.get(ep_id, HUMANOID_GENERIC_CAPTION)
            out_video = str(output_dir / f"episode_{ep_id:06d}.mp4")

            chunks_groups[n_chunks].append({
                "input_video": str(frame_png),
                "prompt": caption,
                "output_video": out_video,
            })

        for n_chunks, items in sorted(chunks_groups.items()):
            batch_json = input_dir / f"batch_input_{n_chunks}chunks.json"
            with open(batch_json, "w", encoding="utf-8") as f:
                json.dump(items, f, indent=2, ensure_ascii=False)
            print(f"[humanoid/{split}] {len(items)} episodes (chunks={n_chunks}) → {batch_json}")


def prepare_single_arm(
    dataset_root: Path, save_root: Path, output_root: Path,
    splits: list[str], max_episodes: int | None, skip_existing: bool,
    iter_name: str,
) -> None:
    source_fps = 15.0
    for split in splits:
        split_dir = dataset_root / split
        if not split_dir.exists():
            print(f"[single_arm/{split}] Directory not found, skipping.")
            continue

        input_dir = save_root / "single_arm" / "input" / "singleview" / "dreamgen" / split
        output_dir = output_root / "single_arm" / "output" / "singleview" / "dreamgen" / split / iter_name
        frames_dir = input_dir / "first_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        episodes = find_episodes_single_arm(split_dir)
        tasks = load_single_arm_tasks(split_dir)

        if max_episodes:
            episodes = episodes[:max_episodes]

        chunks_groups: dict[int, list] = defaultdict(list)
        for ep_id, video_path in episodes:
            frame_png = frames_dir / f"episode_{ep_id:06d}.png"
            if not (skip_existing and frame_png.exists()):
                if not extract_first_frame(video_path, frame_png, 768, 432):
                    print(f"  [WARN] Failed to extract frame: {video_path.name}")
                    continue

            n_frames = get_video_frame_count(video_path)
            n_chunks = compute_num_chunks(n_frames, source_fps) if n_frames else 4

            caption = tasks.get(ep_id, SINGLE_ARM_GENERIC_CAPTION)
            out_video = str(output_dir / f"episode_{ep_id:06d}.mp4")

            chunks_groups[n_chunks].append({
                "input_video": str(frame_png),
                "prompt": caption,
                "output_video": out_video,
            })

        for n_chunks, items in sorted(chunks_groups.items()):
            batch_json = input_dir / f"batch_input_{n_chunks}chunks.json"
            with open(batch_json, "w", encoding="utf-8") as f:
                json.dump(items, f, indent=2, ensure_ascii=False)
            print(f"[single_arm/{split}] {len(items)} episodes (chunks={n_chunks}) → {batch_json}")


def prepare_bimanual(
    dataset_root: Path, save_root: Path, output_root: Path,
    splits: list[str], max_episodes: int | None, skip_existing: bool,
    iter_name: str,
) -> None:
    source_fps = 30.0
    for split in splits:
        split_dir = dataset_root / split
        if not split_dir.exists():
            print(f"[bimanual/{split}] Directory not found, skipping.")
            continue

        input_dir = save_root / "bimanual" / "input" / "singleview" / "dreamgen" / split
        output_dir = output_root / "bimanual" / "output" / "singleview" / "dreamgen" / split / iter_name
        frames_dir = input_dir / "first_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        episodes = find_episodes_bimanual(split_dir)

        if max_episodes:
            episodes = episodes[:max_episodes]

        chunks_groups: dict[int, list] = defaultdict(list)
        for ep_id, task_name, video_path in episodes:
            frame_png = frames_dir / f"{task_name}__episode_{ep_id:06d}.png"
            if not (skip_existing and frame_png.exists()):
                if not extract_first_frame(video_path, frame_png, 768, 432):
                    print(f"  [WARN] Failed to extract frame: {video_path.name}")
                    continue

            n_frames = get_video_frame_count(video_path)
            n_chunks = compute_num_chunks(n_frames, source_fps) if n_frames else 5

            caption = bimanual_task_to_caption(task_name)
            out_video = str(output_dir / f"{task_name}__episode_{ep_id:06d}.mp4")

            chunks_groups[n_chunks].append({
                "input_video": str(frame_png),
                "prompt": caption,
                "output_video": out_video,
            })

        for n_chunks, items in sorted(chunks_groups.items()):
            batch_json = input_dir / f"batch_input_{n_chunks}chunks.json"
            with open(batch_json, "w", encoding="utf-8") as f:
                json.dump(items, f, indent=2, ensure_ascii=False)
            print(f"[bimanual/{split}] {len(items)} episodes (chunks={n_chunks}) → {batch_json}")


def main():
    parser = argparse.ArgumentParser(description="Prepare batch_input.json for DreamGen singleview inference")
    parser.add_argument(
        "--datasets-root", type=str,
        default=str(PROJECT_ROOT / "datasets"),
        help="Root of all datasets",
    )
    parser.add_argument(
        "--save-root", type=str,
        default=str(PROJECT_ROOT / "sampling_dataset" / "dense"),
        help="Root for input/output layout",
    )
    parser.add_argument(
        "--output-root", type=str, default="",
        help="Root for output videos. Defaults to same as --save-root.",
    )
    parser.add_argument(
        "--embodiments", nargs="+",
        default=["humanoid", "single_arm", "bimanual"],
        choices=["humanoid", "single_arm", "bimanual"],
    )
    parser.add_argument("--splits", nargs="+", default=["makovian", "non_makovian"])
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    parser.add_argument(
        "--iter-name", type=str, default="iter_000070000",
        help="Iteration name for output subdirectory",
    )
    args = parser.parse_args()

    datasets_root = Path(args.datasets_root)
    save_root = Path(args.save_root)
    output_root = Path(args.output_root) if args.output_root else save_root

    if "humanoid" in args.embodiments:
        prepare_humanoid(
            datasets_root / "humanoid" / "singleview",
            save_root, output_root,
            args.splits, args.max_episodes, args.skip_existing,
            args.iter_name,
        )

    if "single_arm" in args.embodiments:
        prepare_single_arm(
            datasets_root / "single_arm" / "singleview",
            save_root, output_root,
            args.splits, args.max_episodes, args.skip_existing,
            args.iter_name,
        )

    if "bimanual" in args.embodiments:
        prepare_bimanual(
            datasets_root / "bimanual" / "singleview",
            save_root, output_root,
            args.splits, args.max_episodes, args.skip_existing,
            args.iter_name,
        )


if __name__ == "__main__":
    main()
