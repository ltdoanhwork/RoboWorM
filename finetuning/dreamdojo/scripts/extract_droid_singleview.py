"""Extract single-view (exterior_1_left) subset from DROID dataset.

Creates a LeRobot-compatible dataset at the target path using hardlinks
for parquet data and symlinks for the single selected video stream.

Usage:
    python extract_droid_singleview.py --source datasets/droid_1.0.1_20chunks \
        --target datasets/single_arm/singleview \
        --max_chunks 5   # prototype with first 5 chunks (~5000 episodes)

    # Full extraction (all 95 chunks, ~95k episodes):
    python extract_droid_singleview.py --source datasets/droid_1.0.1_20chunks \
        --target datasets/single_arm/singleview
"""

import argparse
import json
import os
import shutil
from pathlib import Path


VIDEO_KEY = "observation.images.exterior_1_left"


def extract(source: Path, target: Path, max_chunks: int | None = None):
    assert source.exists(), f"Source not found: {source}"
    target.mkdir(parents=True, exist_ok=True)

    info = json.loads((source / "meta" / "info.json").read_text())
    total_chunks = info["total_chunks"]
    if max_chunks is not None:
        total_chunks = min(total_chunks, max_chunks)

    # --- meta ---
    meta_target = target / "meta"
    meta_target.mkdir(parents=True, exist_ok=True)

    for fname in ["episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl"]:
        src_file = source / "meta" / fname
        dst_file = meta_target / fname
        if src_file.exists() and not dst_file.exists():
            shutil.copy2(str(src_file), str(dst_file))
            print(f"  Copied meta/{fname}")

    # Patch info.json: keep only exterior_1_left video
    patched_info = dict(info)
    if max_chunks is not None:
        chunks_size = info["chunks_size"]
        patched_info["total_episodes"] = total_chunks * chunks_size
        patched_info["total_chunks"] = total_chunks
        patched_info["total_videos"] = total_chunks * chunks_size
        patched_info["splits"] = {"train": f"0:{total_chunks * chunks_size}"}

    features = patched_info["features"]
    video_keys_to_remove = []
    for key in features:
        if key.startswith("observation.images.") and key != VIDEO_KEY:
            video_keys_to_remove.append(key)
        if key.startswith("camera_extrinsics.") and key != "camera_extrinsics.exterior_1_left":
            video_keys_to_remove.append(key)
    for key in video_keys_to_remove:
        del features[key]
    patched_info["features"] = features

    (meta_target / "info.json").write_text(json.dumps(patched_info, indent=4))
    print("  Wrote patched meta/info.json")

    # Write modality.json mirroring shared_meta/DROID_modality.json
    modality = {
        "state": {
            "cartesian_position": {
                "original_key": "observation.state.cartesian_position",
                "start": 0,
                "end": 6,
                "rotation_type": None,
                "absolute": True,
                "dtype": "float32",
                "range": None,
            },
            "gripper_position": {
                "original_key": "observation.state.gripper_position",
                "start": 0,
                "end": 1,
                "rotation_type": None,
                "absolute": True,
                "dtype": "float32",
                "range": None,
            },
        },
        "action": {
            "cartesian_position": {
                "original_key": "action.cartesian_position",
                "start": 0,
                "end": 6,
                "rotation_type": None,
                "absolute": True,
                "dtype": "float32",
                "range": None,
            },
            "gripper_position": {
                "original_key": "action.gripper_position",
                "start": 0,
                "end": 1,
                "rotation_type": None,
                "absolute": True,
                "dtype": "float32",
                "range": None,
            },
        },
        "video": {
            "exterior_1_left": {
                "original_key": "observation.images.exterior_1_left",
            }
        },
    }
    (meta_target / "modality.json").write_text(json.dumps(modality, indent=4))
    print("  Wrote meta/modality.json")

    # --- data (hardlink parquet files) ---
    for chunk_idx in range(total_chunks):
        chunk_name = f"chunk-{chunk_idx:03d}"
        src_data_dir = source / "data" / chunk_name
        dst_data_dir = target / "data" / chunk_name
        dst_data_dir.mkdir(parents=True, exist_ok=True)

        if not src_data_dir.exists():
            print(f"  WARNING: {src_data_dir} does not exist, skipping")
            continue

        for parquet in src_data_dir.glob("*.parquet"):
            dst_parquet = dst_data_dir / parquet.name
            if not dst_parquet.exists():
                try:
                    os.link(str(parquet), str(dst_parquet))
                except OSError:
                    os.symlink(str(parquet.resolve()), str(dst_parquet))

        if chunk_idx % 10 == 0:
            print(f"  Linked data/{chunk_name}/")

    # --- videos (symlink only exterior_1_left) ---
    video_subdir = VIDEO_KEY.replace("observation.images.", "")
    for chunk_idx in range(total_chunks):
        chunk_name = f"chunk-{chunk_idx:03d}"
        src_video_dir = source / "videos" / chunk_name / video_subdir
        dst_video_dir = target / "videos" / chunk_name / video_subdir
        dst_video_dir.mkdir(parents=True, exist_ok=True)

        if not src_video_dir.exists():
            print(f"  WARNING: {src_video_dir} does not exist, skipping")
            continue

        for mp4 in src_video_dir.glob("*.mp4"):
            dst_mp4 = dst_video_dir / mp4.name
            if not dst_mp4.exists():
                os.symlink(str(mp4.resolve()), str(dst_mp4))

        if chunk_idx % 10 == 0:
            print(f"  Linked videos/{chunk_name}/{video_subdir}/")

    print(f"\nDone. Extracted {total_chunks} chunks to {target}")


def main():
    parser = argparse.ArgumentParser(description="Extract DROID single-view dataset")
    parser.add_argument("--source", type=str, default="datasets/droid_1.0.1_20chunks",
                        help="Path to source DROID dataset")
    parser.add_argument("--target", type=str, default="datasets/single_arm/singleview",
                        help="Path to target single-view dataset")
    parser.add_argument("--max_chunks", type=int, default=None,
                        help="Max chunks to extract (default: all). Use 5 for prototyping.")
    args = parser.parse_args()

    extract(
        source=Path(args.source),
        target=Path(args.target),
        max_chunks=args.max_chunks,
    )


if __name__ == "__main__":
    main()
