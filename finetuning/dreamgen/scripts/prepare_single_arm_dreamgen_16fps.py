"""Prepare single-arm (DROID) singleview dataset in DreamGen flat format — 16fps normalized.

Reads from:
    datasets/single_arm/singleview/{makovian,non_makovian}/videos/chunk-*/observation.images.wrist_left/*.mp4

Captions are loaded from meta/tasks.jsonl + meta/episodes.jsonl per split.
Falls back to a generic caption if no task description is available.

Writes to:
    datasets/single_arm_dreamgen_16fps/
    ├── videos/{split}__ep{episode_id:06d}.mp4  (transcoded H.264, 16fps)
    └── metas/{split}__ep{episode_id:06d}.txt   (text caption)

Key difference from the original prepare_single_arm_dreamgen.py:
    - Transcodes to 16fps (source is 15fps) using ffmpeg -r 16
    - Skips videos that end up with < MIN_FRAMES after conversion

After running this script, pre-compute T5-XXL embeddings:
    DATASETS="single_arm_16fps" bash finetuning/dreamgen/scripts/compute_t5_embeddings.sh

Usage:
    python finetuning/dreamgen/scripts/prepare_single_arm_dreamgen_16fps.py
    python finetuning/dreamgen/scripts/prepare_single_arm_dreamgen_16fps.py \
        --source datasets/single_arm/singleview \
        --target datasets/single_arm_dreamgen_16fps
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

GENERIC_CAPTION = "A robot arm performing a manipulation task."
VIDEO_SUBDIR = "observation.images.wrist_left"
TARGET_FPS = 16
MIN_FRAMES = 93


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


def load_task_descriptions(split_dir: Path) -> dict[int, str]:
    """Load per-episode task descriptions from episodes.jsonl."""
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
    skipped_existing = 0
    skipped_short = 0

    for split_dir in sorted(source.iterdir()):
        if not split_dir.is_dir():
            continue
        split_name = split_dir.name

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
                skipped_existing += 1
                continue

            caption = ep_task.get(ep_id, GENERIC_CAPTION)

            if not dst_mp4.exists():
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(mp4),
                     "-r", str(TARGET_FPS),
                     "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                     "-pix_fmt", "yuv420p", "-an", str(dst_mp4)],
                    capture_output=True, check=True,
                )

                n_frames = get_video_frame_count(dst_mp4)
                if n_frames is not None and n_frames < MIN_FRAMES:
                    print(f"  [WARN] {out_stem}: only {n_frames} frames after 16fps conversion, "
                          f"skipping (min={MIN_FRAMES})")
                    dst_mp4.unlink()
                    skipped_short += 1
                    continue

            if not dst_txt.exists():
                dst_txt.write_text(caption)

            total += 1
            if total % 100 == 0:
                print(f"  Processed {total} videos...")

    print(f"\nDone. Created {total} entries, skipped {skipped_existing} existing, "
          f"{skipped_short} too short (<{MIN_FRAMES} frames).")
    print(f"Output: {target}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare DROID single-arm singleview dataset for DreamGen (16fps)"
    )
    parser.add_argument(
        "--source", type=str,
        default="datasets/single_arm/singleview",
        help="Root of DROID single-arm singleview dataset",
    )
    parser.add_argument(
        "--target", type=str,
        default="datasets/single_arm_dreamgen_16fps",
        help="Output directory for DreamGen-format dataset (16fps)",
    )
    args = parser.parse_args()

    source = Path(args.source)
    target = Path(args.target)
    assert source.exists(), f"Source not found: {source}"
    print(f"Preparing single-arm singleview @ {TARGET_FPS}fps")
    print(f"  Source: {source}")
    print(f"  Target: {target}")
    prepare(source, target)


if __name__ == "__main__":
    main()
