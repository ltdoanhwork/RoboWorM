"""Prepare single-arm (DROID) singleview dataset in DreamGen flat format.

Reads from:
    datasets/single_arm/singleview/videos/chunk-*/exterior_1_left/*.mp4

DROID has no text annotations, so a generic caption is used per episode.
Optionally, episode task descriptions can be read from the parquet data files
if tasks.jsonl is present.

Writes to:
    datasets/single_arm_dreamgen/
    ├── videos/ep{episode_id:06d}.mp4  (symlink)
    └── metas/ep{episode_id:06d}.txt   (text caption)

After running this script, pre-compute T5-XXL embeddings:
    cd models/dreamgen
    python -m scripts.get_t5_embeddings \
        --dataset_path ../../datasets/single_arm_dreamgen/

Usage:
    # Run from project root
    python finetuning/dreamgen/scripts/prepare_single_arm_dreamgen.py
    python finetuning/dreamgen/scripts/prepare_single_arm_dreamgen.py \
        --source datasets/single_arm/singleview \
        --target datasets/single_arm_dreamgen
"""

import argparse
import json
import os
import re
from pathlib import Path

GENERIC_CAPTION = "A robot arm performing a manipulation task."
VIDEO_SUBDIR = "exterior_1_left"


def load_task_descriptions(source: Path) -> dict[int, str]:
    """Load per-episode task descriptions from tasks.jsonl if available."""
    tasks_file = source / "meta" / "tasks.jsonl"
    episodes_file = source / "meta" / "episodes.jsonl"

    task_map: dict[int, str] = {}  # task_index -> description
    if tasks_file.exists():
        with tasks_file.open() as f:
            for line in f:
                rec = json.loads(line.strip())
                task_map[rec["task_index"]] = rec.get("task", "")

    ep_task: dict[int, str] = {}  # episode_index -> description
    if episodes_file.exists() and task_map:
        with episodes_file.open() as f:
            for line in f:
                rec = json.loads(line.strip())
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

    ep_task = load_task_descriptions(source)
    if ep_task:
        print(f"  Loaded {len(ep_task)} task descriptions from episodes.jsonl")
    else:
        print(f"  No task descriptions found — using generic caption: '{GENERIC_CAPTION}'")

    videos_root = source / "videos"
    if not videos_root.exists():
        raise FileNotFoundError(f"No videos/ directory found at {source}")

    total = 0
    skipped = 0

    for chunk_dir in sorted(videos_root.iterdir()):
        if not chunk_dir.is_dir():
            continue

        video_dir = chunk_dir / VIDEO_SUBDIR
        if not video_dir.exists():
            print(f"  WARNING: {video_dir} does not exist, skipping")
            continue

        for mp4 in sorted(video_dir.glob("*.mp4")):
            stem = mp4.stem  # e.g. "episode_000042"
            ep_match = re.search(r"(\d+)$", stem)
            ep_id = int(ep_match.group(1)) if ep_match else total

            out_stem = f"ep{ep_id:06d}"
            dst_mp4 = videos_out / f"{out_stem}.mp4"
            dst_txt = metas_out / f"{out_stem}.txt"

            if dst_mp4.exists() and dst_txt.exists():
                skipped += 1
                continue

            caption = ep_task.get(ep_id, GENERIC_CAPTION)

            if not dst_mp4.exists():
                os.symlink(mp4.resolve(), dst_mp4)
            if not dst_txt.exists():
                dst_txt.write_text(caption)

            total += 1
            if total % 5000 == 0:
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
        help="Root of DROID single-arm singleview dataset (LeRobot format)",
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
    assert source.exists(), f"Source not found: {source}. Run extract_droid_singleview.py first."
    prepare(source, target)


if __name__ == "__main__":
    main()
