"""
Prepare single_arm/multiview LeRoBot dataset into Ctrl-World annotation format.

For each episode:
  - Reads parquet (states, joints, gripper, instruction)
  - Reads original 15fps videos and downsamples to 5fps (every 3rd frame)
  - Resizes 180 -> 192 height (Ctrl-World expects 192x320)
  - Saves annotation JSON + downsampled videos in Ctrl-World expected layout

Output structure (per subset):
  {output_dir}/{subset}/annotation/val/{episode_id}.json
  {output_dir}/{subset}/videos/val/{episode_id}/0.mp4  (exterior_1_left)
  {output_dir}/{subset}/videos/val/{episode_id}/1.mp4  (exterior_2_left)
  {output_dir}/{subset}/videos/val/{episode_id}/2.mp4  (wrist_left)
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_BASE = "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/datasets/single_arm/multiview"
OUTPUT_BASE = "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/models/Ctrl-World/dataset_example"

VIEW_ORDER = [
    "observation.images.exterior_1_left",
    "observation.images.exterior_2_left",
    "observation.images.wrist_left",
]

DOWNSAMPLE_FACTOR = 3  # 15fps -> 5fps


def find_episodes(dataset_dir):
    """Find all episode parquet files and return sorted list of (chunk_id, episode_id)."""
    data_dir = Path(dataset_dir) / "data"
    episodes = []
    for chunk_dir in sorted(data_dir.glob("chunk-*")):
        for pq in sorted(chunk_dir.glob("episode_*.parquet")):
            ep_id = int(pq.stem.split("_")[1])
            chunk_id = int(chunk_dir.name.split("-")[1])
            episodes.append((chunk_id, ep_id))
    return episodes


def get_video_path(dataset_dir, chunk_id, view_name, episode_id):
    return Path(dataset_dir) / "videos" / f"chunk-{chunk_id:03d}" / view_name / f"episode_{episode_id:06d}.mp4"


def downsample_and_resize_video(input_path, output_path, downsample=3, target_h=192, target_w=320):
    """Use ffmpeg fps filter to downsample 15fps->5fps and resize video."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    target_fps = 15 // downsample  # 5fps
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vf", f"fps={target_fps},scale={target_w}:{target_h}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-an", str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: ffmpeg failed for {input_path}: {result.stderr[:200]}")
        return False
    return True


def get_frame_count(video_path):
    """Get frame count of a video using ffprobe."""
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return 0
    info = json.loads(result.stdout)
    return int(info["streams"][0].get("nb_frames", 0))


def process_episode(dataset_dir, subset_name, chunk_id, episode_id, output_dir, input_save_dir=None):
    """Process one episode: read parquet, downsample videos, write annotation JSON."""
    parquet_path = Path(dataset_dir) / "data" / f"chunk-{chunk_id:03d}" / f"episode_{episode_id:06d}.parquet"
    if not parquet_path.exists():
        return None

    df = pd.read_parquet(parquet_path)
    n_frames_raw = len(df)

    # Downsample parquet data (every 3rd frame)
    df_ds = df.iloc[::DOWNSAMPLE_FACTOR].reset_index(drop=True)
    n_frames = len(df_ds)

    if n_frames < 10:
        return None

    # Extract states: cartesian_position (6D) + gripper (1D) = 7D
    cart_pos = np.array(df_ds["observation.state.cartesian_position"].tolist())  # (N, 6)
    gripper = np.array(df_ds["observation.state.gripper_position"].tolist())  # (N,) scalar
    if gripper.ndim == 1:
        gripper = gripper.reshape(-1, 1)
    states = np.concatenate([cart_pos, gripper], axis=1).tolist()  # (N, 7)

    # Extract joints: joint_position (7D) + gripper (1D) = 8D
    joint_pos = np.array(df_ds["observation.state.joint_position"].tolist())  # (N, 7)
    joints = np.concatenate([joint_pos, gripper], axis=1).tolist()  # (N, 8)

    # Language instruction
    instruction = df_ds["language_instruction"].iloc[0]
    if not instruction or instruction == "":
        instruction = "robot manipulation task"

    # Episode ID string for Ctrl-World
    ep_id_str = f"{episode_id:06d}"

    # Process videos
    video_entries = []
    for view_idx, view_name in enumerate(VIEW_ORDER):
        src_video = get_video_path(dataset_dir, chunk_id, view_name, episode_id)
        if not src_video.exists():
            print(f"  Missing video: {src_video}")
            return None

        dst_video = Path(output_dir) / subset_name / "videos" / "val" / ep_id_str / f"{view_idx}.mp4"
        if not dst_video.exists():
            success = downsample_and_resize_video(src_video, dst_video)
            if not success:
                return None

        rel_path = f"videos/val/{ep_id_str}/{view_idx}.mp4"
        video_entries.append({"video_path": rel_path})

    # Verify video frame count matches downsampled parquet
    first_video = Path(output_dir) / subset_name / "videos" / "val" / ep_id_str / "0.mp4"
    video_frames = get_frame_count(first_video)

    # Use the minimum of parquet-derived and video-derived frame count
    actual_length = min(n_frames, video_frames) if video_frames > 0 else n_frames
    states = states[:actual_length]
    joints = joints[:actual_length]

    # Build annotation
    annotation = {
        "texts": [instruction],
        "episode_id": episode_id,
        "success": True,
        "video_length": actual_length,
        "state_length": actual_length,
        "raw_length": n_frames_raw,
        "videos": video_entries,
        "states": states,
        "joints": joints,
    }

    # Save annotation
    anno_dir = Path(output_dir) / subset_name / "annotation" / "val"
    anno_dir.mkdir(parents=True, exist_ok=True)
    anno_path = anno_dir / f"{ep_id_str}.json"
    with open(anno_path, "w") as f:
        json.dump(annotation, f)

    # Save input for benchmark (GT videos + metadata)
    if input_save_dir is not None:
        import shutil
        ep_input_dir = Path(input_save_dir) / f"episode_{ep_id_str}"
        ep_input_dir.mkdir(parents=True, exist_ok=True)

        # Copy downsampled videos (all 3 views)
        for view_idx in range(len(VIEW_ORDER)):
            src = Path(output_dir) / subset_name / "videos" / "val" / ep_id_str / f"{view_idx}.mp4"
            dst = ep_input_dir / f"view_{view_idx}.mp4"
            if src.exists() and not dst.exists():
                shutil.copy2(str(src), str(dst))

        # Save metadata
        input_meta = {
            "episode_id": episode_id,
            "instruction": instruction,
            "num_frames": actual_length,
            "raw_frames": n_frames_raw,
            "fps": 5,
            "resolution": "320x192",
            "states": states,
            "joints": joints,
        }
        meta_path = ep_input_dir / "metadata.json"
        if not meta_path.exists():
            with open(meta_path, "w") as f:
                json.dump(input_meta, f, indent=2)

    return {"id": ep_id_str, "length": actual_length, "instruction": instruction}


INPUT_SAVE_BASE = "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/sampling_dataset/dense/single_arm/input/multiview/ctrlworld"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=["makovian", "non_makovian", "both"], default="both")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_BASE)
    parser.add_argument("--dataset-base", type=str, default=DATASET_BASE)
    parser.add_argument("--input-save-dir", type=str, default=INPUT_SAVE_BASE)
    args = parser.parse_args()

    subsets = ["makovian", "non_makovian"] if args.subset == "both" else [args.subset]

    for subset in subsets:
        dataset_dir = os.path.join(args.dataset_base, subset)
        subset_output_name = f"single_arm_multiview_{subset}"
        subset_input_save_dir = os.path.join(args.input_save_dir, subset)

        print(f"\n{'='*60}")
        print(f"Processing: {subset} -> {subset_output_name}")
        print(f"Input save: {subset_input_save_dir}")
        print(f"{'='*60}")

        episodes = find_episodes(dataset_dir)
        print(f"Found {len(episodes)} episodes")

        results = []
        for i, (chunk_id, ep_id) in enumerate(episodes):
            print(f"  [{i+1}/{len(episodes)}] chunk={chunk_id:03d} episode={ep_id:06d}", end=" ")
            result = process_episode(
                dataset_dir, subset_output_name, chunk_id, ep_id,
                args.output_dir, input_save_dir=subset_input_save_dir,
            )
            if result:
                results.append(result)
                print(f"-> {result['length']} frames")
            else:
                print("-> SKIPPED")

        # Save summary
        summary_path = os.path.join(args.output_dir, subset_output_name, "preparation_summary.json")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump({
                "total_episodes": len(results),
                "episodes": results,
            }, f, indent=2)

        print(f"\nDone: {len(results)} episodes prepared for {subset}")


if __name__ == "__main__":
    main()
