"""Prepare humanoid GR1 singleview dataset in DreamGen flat format.

Reads from:
    datasets/humanoid/singleview/{makovian,non_makovian}/videos/chunk-*/observation.images.ego_view_freq20/*.mp4

Captions are loaded from:
    datasets/humanoid/singleview/{makovian,non_makovian}/captions/episode_XXXXXX.txt

Writes to:
    datasets/humanoid_dreamgen/
    ├── videos/{split}__episode_{id:06d}.mp4  (copy)
    └── metas/{split}__episode_{id:06d}.txt   (text caption)

After running this script, pre-compute T5-XXL embeddings:
    cd models/dreamgen
    python -m scripts.get_t5_embeddings \
        --dataset_path ../../datasets/humanoid_dreamgen/

Usage:
    python finetuning/dreamgen/scripts/prepare_humanoid_dreamgen.py
    python finetuning/dreamgen/scripts/prepare_humanoid_dreamgen.py \
        --source datasets/humanoid/singleview \
        --target datasets/humanoid_dreamgen
"""

import argparse
import os
import re
import shutil
from pathlib import Path

VIDEO_SUBDIR = "observation.images.ego_view_freq20"
GENERIC_CAPTION = "A GR-1 humanoid robot performing a manipulation task."


def load_captions(split_dir: Path) -> dict[int, str]:
    """Load per-episode captions from captions/ directory."""
    captions_dir = split_dir / "captions"
    ep_caption: dict[int, str] = {}

    if not captions_dir.exists():
        return ep_caption

    for txt_file in captions_dir.glob("*.txt"):
        ep_match = re.search(r"(\d+)$", txt_file.stem)
        if ep_match:
            ep_id = int(ep_match.group(1))
            ep_caption[ep_id] = txt_file.read_text().strip()

    return ep_caption


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
        # Only process makovian / non_makovian splits
        if split_name not in ("makovian", "non_makovian"):
            continue

        ep_caption = load_captions(split_dir)
        if ep_caption:
            print(f"  [{split_name}] Loaded {len(ep_caption)} captions")
        else:
            print(f"  [{split_name}] No captions found — using generic caption")

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

            out_stem = f"{split_name}__{stem}"
            dst_mp4 = videos_out / f"{out_stem}.mp4"
            dst_txt = metas_out / f"{out_stem}.txt"

            if dst_mp4.exists() and dst_txt.exists():
                skipped += 1
                continue

            caption = ep_caption.get(ep_id, GENERIC_CAPTION)

            if not dst_mp4.exists():
                shutil.copy2(mp4, dst_mp4)
            if not dst_txt.exists():
                dst_txt.write_text(caption)

            total += 1
            if total % 200 == 0:
                print(f"  Processed {total} videos...")

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
        help="Root of humanoid singleview dataset (contains makovian/ and non_makovian/)",
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
