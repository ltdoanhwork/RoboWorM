"""Prepare humanoid GR1 singleview dataset in DreamGen flat format.

Reads from:
    datasets/humanoid/singleview/PhysicalAI-Robotics-GR00T-Teleop-GR1/In-lab_Eval/{task}/videos/chunk-*/

The video key is resolved from meta/modality.json (defaults to ego_view_freq20 for GR1 unified).
Captions are derived from the task folder name (e.g. gr1_unified.mug_robot →
"A GR-1 humanoid robot performing mug robot task."). If a tasks.jsonl or
episodes.jsonl with text annotations is present, those are preferred.

Writes to:
    datasets/humanoid_dreamgen/
    ├── videos/{task}__ep{episode_id:06d}.mp4  (symlink)
    └── metas/{task}__ep{episode_id:06d}.txt   (text caption)

After running this script, pre-compute T5-XXL embeddings:
    cd models/dreamgen
    python -m scripts.get_t5_embeddings \
        --dataset_path ../../datasets/humanoid_dreamgen/

Usage:
    python prepare_humanoid_dreamgen.py
    python prepare_humanoid_dreamgen.py \
        --source datasets/humanoid/singleview \
        --target datasets/humanoid_dreamgen
"""

import argparse
import json
import os
import re
from pathlib import Path

DEFAULT_VIDEO_KEY = "ego_view_freq20"
INLAB_EVAL_SUBDIR = "PhysicalAI-Robotics-GR00T-Teleop-GR1/In-lab_Eval"


def get_video_key(task_dir: Path) -> str:
    """Read video key from meta/modality.json, falling back to default."""
    modality_file = task_dir / "meta" / "modality.json"
    if modality_file.exists():
        try:
            modality = json.loads(modality_file.read_text())
            video_keys = list(modality.get("video", {}).keys())
            if video_keys:
                return video_keys[0]
        except Exception:
            pass
    return DEFAULT_VIDEO_KEY


def load_episode_captions(task_dir: Path) -> dict[int, str]:
    """Load per-episode captions from tasks.jsonl + episodes.jsonl if available."""
    tasks_file = task_dir / "meta" / "tasks.jsonl"
    episodes_file = task_dir / "meta" / "episodes.jsonl"

    task_map: dict[int, str] = {}
    if tasks_file.exists():
        with tasks_file.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                task_map[rec.get("task_index", -1)] = rec.get("task", "")

    ep_caption: dict[int, str] = {}
    if episodes_file.exists() and task_map:
        with episodes_file.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ep_idx = rec.get("episode_index", -1)
                t_idx = rec.get("task_index", -1)
                if ep_idx >= 0 and t_idx in task_map and task_map[t_idx]:
                    ep_caption[ep_idx] = task_map[t_idx]

    return ep_caption


def task_name_to_caption(task_name: str) -> str:
    """Convert task folder name to natural language caption.

    Examples:
        gr1_unified.mug_robot  → "A GR-1 humanoid robot performing mug robot task."
        gr1_unified.fold_cloth_robot → "A GR-1 humanoid robot performing fold cloth task."
    """
    # Strip leading prefix like "gr1_unified."
    name = re.sub(r"^gr1_unified\.", "", task_name)
    # Strip trailing _robot suffix
    name = re.sub(r"_robot$", "", name)
    name = name.replace("_", " ")
    return f"A GR-1 humanoid robot performing {name} task."


def prepare(source: Path, target: Path) -> None:
    videos_out = target / "videos"
    metas_out = target / "metas"
    videos_out.mkdir(parents=True, exist_ok=True)
    metas_out.mkdir(parents=True, exist_ok=True)

    inlab_root = source / INLAB_EVAL_SUBDIR
    if not inlab_root.exists():
        # Try searching directly in source
        inlab_root = source
        print(f"  NOTE: In-lab_Eval dir not found, searching directly in {source}")

    total = 0
    skipped = 0

    task_dirs = sorted(d for d in inlab_root.iterdir() if d.is_dir())
    if not task_dirs:
        raise FileNotFoundError(f"No task directories found in {inlab_root}")

    for task_dir in task_dirs:
        task_name = task_dir.name
        fallback_caption = task_name_to_caption(task_name)

        video_key = get_video_key(task_dir)
        ep_caption = load_episode_captions(task_dir)

        videos_root = task_dir / "videos"
        if not videos_root.exists():
            print(f"  WARNING: no videos/ dir in {task_dir}, skipping")
            continue

        mp4_files = sorted(videos_root.glob(f"chunk-*/{video_key}/*.mp4"))
        if not mp4_files:
            # Try any video subdir
            mp4_files = sorted(videos_root.glob("chunk-*/*/*.mp4"))
            if not mp4_files:
                print(f"  WARNING: no mp4 files found under {videos_root}, skipping")
                continue

        for mp4 in mp4_files:
            stem = mp4.stem  # e.g. "episode_000042"
            ep_match = re.search(r"(\d+)$", stem)
            ep_id = int(ep_match.group(1)) if ep_match else total

            out_stem = f"{task_name}__ep{ep_id:06d}"
            dst_mp4 = videos_out / f"{out_stem}.mp4"
            dst_txt = metas_out / f"{out_stem}.txt"

            if dst_mp4.exists() and dst_txt.exists():
                skipped += 1
                continue

            caption = ep_caption.get(ep_id, fallback_caption)

            if not dst_mp4.exists():
                os.symlink(mp4.resolve(), dst_mp4)
            if not dst_txt.exists():
                dst_txt.write_text(caption)

            total += 1
            if total % 200 == 0:
                print(f"  Processed {total} videos...")

        print(f"  Task {task_name}: {len(mp4_files)} videos")

    print(f"\nDone. Created {total} entries, skipped {skipped} existing.")
    print(f"Output: {target}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare GR1 humanoid singleview dataset for DreamGen training"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="datasets/humanoid/singleview",
        help="Root of humanoid singleview dataset",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="datasets/humanoid_dreamgen",
        help="Output directory for DreamGen-format dataset",
    )
    args = parser.parse_args()

    source = Path(args.source)
    target = Path(args.target)
    assert source.exists(), f"Source not found: {source}"
    prepare(source, target)


if __name__ == "__main__":
    main()
