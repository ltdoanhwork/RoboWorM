"""Prepare single-arm (DROID) STATIC views dataset — 16fps, segmented clips.

Uses the 2 static cameras (exterior_1_left, exterior_2_left).
Each episode video is split into non-overlapping segments of roughly equal length,
ensuring every segment >= 93 frames so the training dataset class can sample from it.

This guarantees the model sees ALL phases of the task (start, middle, completion)
rather than random 5.8s windows from a longer video.

Segment logic for a video of N frames:
  - num_segments = N // SEGMENT_TARGET
  - segment_length = N // num_segments  (between 93 and ~140 frames)
  - Each segment becomes a separate training sample with the same caption

Reads from:
    datasets/single_arm/multiview/{makovian,non_makovian}/videos/chunk-*/
        observation.images.exterior_1_left/*.mp4
        observation.images.exterior_2_left/*.mp4

Writes to:
    datasets/single_arm_static_dreamgen_16fps/
    ├── videos/{split}__ext1__ep{id:06d}__seg{i:02d}.mp4
    ├── metas/{split}__ext1__ep{id:06d}__seg{i:02d}.txt
    └── ...

Usage:
    python finetuning/dreamgen/scripts/prepare_single_arm_static_dreamgen_16fps.py
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

GENERIC_CAPTION = "A robot arm performing a manipulation task."
STATIC_VIEWS = ["observation.images.exterior_1_left", "observation.images.exterior_2_left"]
VIEW_SHORT_NAMES = {"observation.images.exterior_1_left": "ext1",
                    "observation.images.exterior_2_left": "ext2"}
TARGET_FPS = 16
MIN_FRAMES = 94
SEGMENT_TARGET = 93


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


def transcode_16fps(src: Path, dst: Path) -> bool:
    """Transcode source video to 16fps H.264."""
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src),
         "-r", str(TARGET_FPS),
         "-c:v", "libx264", "-preset", "fast", "-crf", "18",
         "-pix_fmt", "yuv420p", "-an", str(dst)],
        capture_output=True,
    )
    return result.returncode == 0


def split_video_segments(src: Path, out_dir: Path, stem_prefix: str,
                         n_frames: int) -> list[Path]:
    """Split a video into roughly equal segments, each >= MIN_FRAMES.

    Returns list of output segment paths.
    """
    if n_frames < MIN_FRAMES:
        return []

    num_segments = max(1, n_frames // SEGMENT_TARGET)
    seg_length = n_frames // num_segments

    if seg_length < MIN_FRAMES:
        num_segments = max(1, n_frames // MIN_FRAMES)
        seg_length = n_frames // num_segments

    outputs = []
    for i in range(num_segments):
        start_frame = i * seg_length
        if i == num_segments - 1:
            end_frame = n_frames
        else:
            end_frame = (i + 1) * seg_length

        seg_frames = end_frame - start_frame
        if seg_frames < MIN_FRAMES:
            continue

        seg_path = out_dir / f"{stem_prefix}__seg{i:02d}.mp4"
        if seg_path.exists():
            outputs.append(seg_path)
            continue

        start_time = start_frame / TARGET_FPS
        duration = seg_frames / TARGET_FPS

        result = subprocess.run(
            ["ffmpeg", "-y",
             "-ss", f"{start_time:.4f}",
             "-i", str(src),
             "-t", f"{duration:.4f}",
             "-c:v", "libx264", "-preset", "fast", "-crf", "18",
             "-pix_fmt", "yuv420p", "-an", str(seg_path)],
            capture_output=True,
        )
        if result.returncode == 0 and seg_path.exists():
            outputs.append(seg_path)

    return outputs


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

    total_clips = 0
    total_episodes = 0
    skipped_existing = 0
    skipped_short = 0

    tmp_dir = Path(tempfile.mkdtemp(prefix="droid_static_"))

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

        for view_name in STATIC_VIEWS:
            view_short = VIEW_SHORT_NAMES[view_name]
            mp4_files = sorted(videos_root.glob(f"chunk-*/{view_name}/*.mp4"))

            if not mp4_files:
                print(f"  [{split_name}/{view_short}] No videos found, skipping")
                continue

            for mp4 in mp4_files:
                stem = mp4.stem
                ep_match = re.search(r"(\d+)$", stem)
                ep_id = int(ep_match.group(1)) if ep_match else total_episodes

                out_stem = f"{split_name}__{view_short}__ep{ep_id:06d}"

                # Check if segments already exist
                existing_segs = list(videos_out.glob(f"{out_stem}__seg*.mp4"))
                if existing_segs:
                    skipped_existing += len(existing_segs)
                    total_clips += len(existing_segs)
                    total_episodes += 1
                    continue

                caption = ep_task.get(ep_id, GENERIC_CAPTION)

                # Transcode to 16fps temp file
                tmp_16fps = tmp_dir / f"{out_stem}.mp4"
                if not transcode_16fps(mp4, tmp_16fps):
                    print(f"  [WARN] Failed to transcode: {mp4.name}")
                    continue

                n_frames = get_video_frame_count(tmp_16fps)
                if n_frames is None or n_frames < MIN_FRAMES:
                    print(f"  [WARN] {out_stem}: {n_frames} frames, too short")
                    tmp_16fps.unlink(missing_ok=True)
                    skipped_short += 1
                    continue

                # Split into segments
                seg_paths = split_video_segments(
                    tmp_16fps, videos_out, out_stem, n_frames
                )

                # Write caption for each segment
                for seg_path in seg_paths:
                    txt_path = metas_out / seg_path.name.replace(".mp4", ".txt")
                    if not txt_path.exists():
                        txt_path.write_text(caption)
                    total_clips += 1

                tmp_16fps.unlink(missing_ok=True)
                total_episodes += 1

                if total_episodes % 50 == 0:
                    print(f"  Processed {total_episodes} episodes → {total_clips} clips...")

    # Cleanup temp dir
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\nDone. {total_episodes} episodes → {total_clips} clips "
          f"(avg {total_clips/max(total_episodes,1):.1f} clips/episode)")
    print(f"  Skipped: {skipped_existing} existing, {skipped_short} too short")
    print(f"Output: {target}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare DROID single-arm STATIC views — 16fps segmented clips"
    )
    parser.add_argument(
        "--source", type=str,
        default="datasets/single_arm/multiview",
    )
    parser.add_argument(
        "--target", type=str,
        default="datasets/single_arm_static_dreamgen_16fps",
    )
    args = parser.parse_args()

    source = Path(args.source)
    target = Path(args.target)
    assert source.exists(), f"Source not found: {source}"
    print(f"Preparing single-arm STATIC views (segmented) @ {TARGET_FPS}fps")
    print(f"  Source: {source}")
    print(f"  Target: {target}")
    print(f"  Views: {', '.join(VIEW_SHORT_NAMES.values())}")
    print(f"  Segment target: {SEGMENT_TARGET} frames ({SEGMENT_TARGET/TARGET_FPS:.1f}s)")
    prepare(source, target)


if __name__ == "__main__":
    main()
