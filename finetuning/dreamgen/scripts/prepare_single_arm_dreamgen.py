"""Prepare single-arm (DROID) singleview dataset in DreamGen flat format.

Reads from:
    datasets/single_arm/singleview/{makovian,non_makovian}/videos/chunk-*/observation.images.wrist_left/*.mp4

Captions are loaded from meta/tasks.jsonl + meta/episodes.jsonl per split.
Falls back to a generic caption if no task description is available.

Writes to:
    datasets/single_arm_dreamgen/
    ├── videos/{split}__ep{episode_id:06d}.mp4  (transcoded H.264 copy)
    └── metas/{split}__ep{episode_id:06d}.txt   (text caption)

After running this script, pre-compute T5-XXL embeddings:
    cd models/dreamgen
    python -m scripts.get_t5_embeddings \
        --dataset_path ../../datasets/single_arm_dreamgen/

Usage:
    python finetuning/dreamgen/scripts/prepare_single_arm_dreamgen.py
    python finetuning/dreamgen/scripts/prepare_single_arm_dreamgen.py \
        --source datasets/single_arm/singleview \
        --target datasets/single_arm_dreamgen
"""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

GENERIC_CAPTION = "A robot arm performing a manipulation task."
VIDEO_SUBDIR = "observation.images.wrist_left"


def load_task_descriptions(split_dir: Path) -> dict[int, str]:
    """Load per-episode task descriptions from episodes.jsonl.

    Supports two formats:
      - {"episode_index": N, "tasks": ["desc"]}  (LeRoBot v2)
      - {"episode_index": N, "task_index": M} + tasks.jsonl  (LeRoBot v1)
    """
    episodes_file = split_dir / "meta" / "episodes.jsonl"
    tasks_file = split_dir / "meta" / "tasks.jsonl"

    ep_task: dict[int, str] = {}
    if not episodes_file.exists():
        return ep_task

    # Try v2 format first: "tasks" field directly in episodes.jsonl
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

    # Fallback: v1 format with separate tasks.jsonl
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

    if task_map:
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


def prepare(source: Path, target: Path) -> None:
    videos_out = target / "videos"
    metas_out = target / "metas"
    videos_out.mkdir(parents=True, exist_ok=True)
    metas_out.mkdir(parents=True, exist_ok=True)

    total = 0
    skipped = 0

    for split_dir in sorted(source.iterdir()):
        if not split_dir.is_dir():
            continue
        split_name = split_dir.name  # makovian / non_makovian

        ep_task = load_task_descriptions(split_dir)
        if ep_task:
            print(f"  [{split_name}] Loaded {len(ep_task)} task descriptions")
        else:
            print(f"  [{split_name}] No task descriptions — using generic caption")

        videos_root = split_dir / "videos"
        if not videos_root.exists():
            print(f"  WARNING: no videos/ dir in {split_dir}, skipping")
            continue

        mp4_files = sorted(videos_root.glob(f"chunk-*/{VIDEO_SUBDIR}/*.mp4"))
        if not mp4_files:
            mp4_files = sorted(videos_root.glob("chunk-*/*/*.mp4"))

        for mp4 in mp4_files:
            stem = mp4.stem
            ep_match = re.search(r"(\d+)$", stem)
            ep_id = int(ep_match.group(1)) if ep_match else total

            out_stem = f"{split_name}__ep{ep_id:06d}"
            dst_mp4 = videos_out / f"{out_stem}.mp4"
            dst_txt = metas_out / f"{out_stem}.txt"

            if dst_mp4.exists() and dst_txt.exists():
                skipped += 1
                continue

            caption = ep_task.get(ep_id, GENERIC_CAPTION)

            if not dst_mp4.exists():
                # Transcode AV1 -> H.264 (decord doesn't support AV1)
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(mp4), "-c:v", "libx264",
                     "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
                     "-an", str(dst_mp4)],
                    capture_output=True, check=True,
                )
            if not dst_txt.exists():
                dst_txt.write_text(caption)

            total += 1
            if total % 500 == 0:
                print(f"  Processed {total} videos...")

    print(f"\nDone. Created {total} entries, skipped {skipped} existing.")
    print(f"Output: {target}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare DROID single-arm singleview dataset for DreamGen training"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="datasets/single_arm/singleview",
        help="Root of DROID single-arm singleview dataset (contains makovian/ and non_makovian/)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="datasets/single_arm_dreamgen",
        help="Output directory for DreamGen-format dataset",
    )
    args = parser.parse_args()

    source = Path(args.source)
    target = Path(args.target)
    assert source.exists(), f"Source not found: {source}"
    prepare(source, target)


if __name__ == "__main__":
    main()
