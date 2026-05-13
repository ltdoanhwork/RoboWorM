"""Prepare bimanual singleview dataset in DreamGen flat format.

Reads from:
    datasets/bimanual/singleview/{makovian,non_makovian}/{task}/videos/chunk-*/cam_high/*.mp4

Writes to:
    datasets/bimanual_dreamgen/
    ├── videos/{task}__ep{episode_id:06d}.mp4  (symlink)
    └── metas/{task}__ep{episode_id:06d}.txt   (text caption)

After running this script, pre-compute T5-XXL embeddings:
    cd models/dreamgen
    python -m scripts.get_t5_embeddings \
        --dataset_path ../../datasets/bimanual_dreamgen/

Usage:
    python prepare_bimanual_dreamgen.py
    python prepare_bimanual_dreamgen.py \
        --source datasets/bimanual/singleview \
        --target datasets/bimanual_dreamgen
"""

import argparse
import os
import re
from pathlib import Path


def task_name_to_caption(task_name: str) -> str:
    """Convert snake_case task folder name to a natural language caption."""
    # Remove common prefixes/suffixes that are already captured contextually
    name = task_name.replace("_", " ")
    # Capitalize first letter
    name = name[0].upper() + name[1:] if name else name
    return f"A bimanual robot performing the task: {name}."


def prepare(source: Path, target: Path) -> None:
    videos_out = target / "videos"
    metas_out = target / "metas"
    videos_out.mkdir(parents=True, exist_ok=True)
    metas_out.mkdir(parents=True, exist_ok=True)

    total = 0
    skipped = 0

    # Walk both makovian and non_makovian splits
    for split_dir in sorted(source.iterdir()):
        if not split_dir.is_dir():
            continue
        split_name = split_dir.name  # makovian / non_makovian

        for task_dir in sorted(split_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            task_name = task_dir.name
            caption = task_name_to_caption(task_name)

            videos_root = task_dir / "videos"
            if not videos_root.exists():
                print(f"  WARNING: no videos/ dir in {task_dir}, skipping")
                continue

            # Collect all cam_high mp4 files across all chunks
            mp4_files = sorted(videos_root.glob("chunk-*/cam_high/*.mp4"))
            if not mp4_files:
                # Fallback: try any video subdir
                mp4_files = sorted(videos_root.glob("chunk-*/*/*.mp4"))

            for mp4 in mp4_files:
                # Build a unique stem: {split}_{task}__ep{episode_id}
                # episode_id is parsed from the filename (e.g. episode_000042.mp4)
                stem = mp4.stem  # e.g. "episode_000042"
                ep_match = re.search(r"(\d+)$", stem)
                ep_id = int(ep_match.group(1)) if ep_match else total

                out_stem = f"{split_name}__{task_name}__ep{ep_id:06d}"
                dst_mp4 = videos_out / f"{out_stem}.mp4"
                dst_txt = metas_out / f"{out_stem}.txt"

                if dst_mp4.exists() and dst_txt.exists():
                    skipped += 1
                    continue

                if not dst_mp4.exists():
                    os.symlink(mp4.resolve(), dst_mp4)
                if not dst_txt.exists():
                    dst_txt.write_text(caption)

                total += 1
                if total % 500 == 0:
                    print(f"  Processed {total} videos...")

    print(f"\nDone. Created {total} entries, skipped {skipped} existing.")
    print(f"Output: {target}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare bimanual singleview dataset for DreamGen training"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="datasets/bimanual/singleview",
        help="Root of bimanual singleview dataset (contains makovian/ and non_makovian/)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="datasets/bimanual_dreamgen",
        help="Output directory for DreamGen-format dataset",
    )
    args = parser.parse_args()

    source = Path(args.source)
    target = Path(args.target)
    assert source.exists(), f"Source not found: {source}"
    prepare(source, target)


if __name__ == "__main__":
    main()
