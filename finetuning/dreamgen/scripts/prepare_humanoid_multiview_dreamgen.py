"""Prepare humanoid multiview dataset in DreamGen flat format.

Reads from:
    datasets/humanoid/multiview/{makovian,non_makovian}/{scenario}/
        videos/chunk-*/{camera_key}/episode_XXXXXX.mp4
        meta/tasks.jsonl

Camera keys (4 views, common to both makovian and non_makovian):
    observation.images.cam_front_rgb
    observation.images.cam_head_rgb
    observation.images.cam_left_wrist_rgb
    observation.images.cam_right_wrist_rgb

Writes to:
    datasets/humanoid_multiview_dreamgen/
    ├── videos/{split}__{scenario}__{camera_short}__episode_{id:06d}.mp4
    └── metas/{split}__{scenario}__{camera_short}__episode_{id:06d}.txt

Usage:
    python finetuning/dreamgen/scripts/prepare_humanoid_multiview_dreamgen.py
"""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

CAMERA_KEYS = [
    "observation.images.cam_front_rgb",
    "observation.images.cam_head_rgb",
    "observation.images.cam_left_wrist_rgb",
    "observation.images.cam_right_wrist_rgb",
]

CAMERA_SHORT_NAMES = {
    "observation.images.cam_front_rgb": "front",
    "observation.images.cam_head_rgb": "head",
    "observation.images.cam_left_wrist_rgb": "left_wrist",
    "observation.images.cam_right_wrist_rgb": "right_wrist",
}

GENERIC_CAPTION = "An Airbot MMK2 robot performing a manipulation task."


def load_task_caption(scenario_dir: Path) -> str:
    tasks_file = scenario_dir / "meta" / "tasks.jsonl"
    if not tasks_file.exists():
        return GENERIC_CAPTION
    tasks = []
    for line in tasks_file.read_text().strip().splitlines():
        entry = json.loads(line)
        tasks.append(entry.get("task", ""))
    if tasks:
        return tasks[0]
    return GENERIC_CAPTION


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
        split_name = split_dir.name
        if split_name not in ("makovian", "non_makovian"):
            continue

        for scenario_dir in sorted(split_dir.iterdir()):
            if not scenario_dir.is_dir():
                continue
            scenario_name = scenario_dir.name

            task_caption = load_task_caption(scenario_dir)
            videos_root = scenario_dir / "videos"
            if not videos_root.exists():
                print(f"  WARNING: no videos/ in {scenario_dir}, skipping")
                continue

            for camera_key in CAMERA_KEYS:
                camera_short = CAMERA_SHORT_NAMES[camera_key]
                caption = f"An Airbot MMK2 robot performing: {task_caption} Camera view: {camera_short}."

                mp4_files = sorted(videos_root.glob(f"chunk-*/{camera_key}/*.mp4"))
                if not mp4_files:
                    continue

                for mp4 in mp4_files:
                    stem = mp4.stem
                    ep_match = re.search(r"(\d+)$", stem)
                    ep_id = int(ep_match.group(1)) if ep_match else total

                    out_stem = f"{split_name}__{scenario_name}__{camera_short}__episode_{ep_id:06d}"
                    dst_mp4 = videos_out / f"{out_stem}.mp4"
                    dst_txt = metas_out / f"{out_stem}.txt"

                    if dst_mp4.exists() and dst_txt.exists():
                        skipped += 1
                        continue

                    if not dst_mp4.exists():
                        subprocess.run(
                            [
                                "ffmpeg", "-y", "-i", str(mp4),
                                "-c:v", "libx264", "-preset", "fast",
                                "-crf", "18", "-pix_fmt", "yuv420p",
                                "-an", str(dst_mp4),
                            ],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            check=True,
                        )
                    if not dst_txt.exists():
                        dst_txt.write_text(caption)

                    total += 1
                    if total % 200 == 0:
                        print(f"  Processed {total} videos...")

    print(f"\nDone. Created {total} entries, skipped {skipped} existing.")
    print(f"Output: {target}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare humanoid multiview dataset for DreamGen training"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="datasets/humanoid/multiview",
        help="Root of humanoid multiview dataset",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="datasets/humanoid_multiview_dreamgen",
        help="Output directory for DreamGen-format dataset",
    )
    args = parser.parse_args()

    source = Path(args.source)
    target = Path(args.target)
    assert source.exists(), f"Source not found: {source}"
    prepare(source, target)


if __name__ == "__main__":
    main()
