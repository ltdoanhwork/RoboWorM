"""Prepare bimanual singleview dataset — 16fps, segmented clips.

Converts 30fps source → 16fps, then splits each episode into non-overlapping
segments of roughly equal length (each >= 93 frames). This ensures the model
sees ALL phases of the task including completion.

Segment logic for a video of N frames:
  - num_segments = N // SEGMENT_TARGET
  - segment_length = N // num_segments  (between 93 and ~140 frames)
  - Each segment becomes a separate training sample with the same caption

Reads from:
    datasets/bimanual/singleview/{makovian,non_makovian}/{task}/videos/chunk-*/
        observation.images.cam_high/*.mp4

Writes to:
    datasets/bimanual_dreamgen_16fps/
    ├── videos/{split}__{task}__ep{id:06d}__seg{i:02d}.mp4
    ├── metas/{split}__{task}__ep{id:06d}__seg{i:02d}.txt
    └── ...

Usage:
    python finetuning/dreamgen/scripts/prepare_bimanual_dreamgen_16fps.py
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

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
    """Split a video into roughly equal segments, each >= MIN_FRAMES."""
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


def task_name_to_caption(task_name: str) -> str:
    """Convert snake_case task folder name to a natural language caption."""
    name = task_name.replace("_", " ")
    name = name[0].upper() + name[1:] if name else name
    return f"A bimanual robot performing the task: {name}."


def prepare(source: Path, target: Path) -> None:
    videos_out = target / "videos"
    metas_out = target / "metas"
    videos_out.mkdir(parents=True, exist_ok=True)
    metas_out.mkdir(parents=True, exist_ok=True)

    total_clips = 0
    total_episodes = 0
    skipped_existing = 0
    skipped_short = 0

    tmp_dir = Path(tempfile.mkdtemp(prefix="bimanual_16fps_"))

    for split_dir in sorted(source.iterdir()):
        if not split_dir.is_dir():
            continue
        split_name = split_dir.name

        for task_dir in sorted(split_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            task_name = task_dir.name
            caption = task_name_to_caption(task_name)

            videos_root = task_dir / "videos"
            if not videos_root.exists():
                print(f"  WARNING: no videos/ dir in {task_dir}, skipping")
                continue

            mp4_files = sorted(videos_root.glob("chunk-*/observation.images.cam_high/*.mp4"))
            if not mp4_files:
                mp4_files = sorted(videos_root.glob("chunk-*/*/*.mp4"))

            for mp4 in mp4_files:
                stem = mp4.stem
                ep_match = re.search(r"(\d+)$", stem)
                ep_id = int(ep_match.group(1)) if ep_match else total_episodes

                out_stem = f"{split_name}__{task_name}__ep{ep_id:06d}"

                # Check if segments already exist
                existing_segs = list(videos_out.glob(f"{out_stem}__seg*.mp4"))
                if existing_segs:
                    skipped_existing += len(existing_segs)
                    total_clips += len(existing_segs)
                    total_episodes += 1
                    continue

                # Transcode 30fps → 16fps
                tmp_16fps = tmp_dir / f"{out_stem}.mp4"
                if not transcode_16fps(mp4, tmp_16fps):
                    print(f"  [WARN] Failed to transcode: {mp4.name}")
                    continue

                n_frames = get_video_frame_count(tmp_16fps)
                if n_frames is None or n_frames < MIN_FRAMES:
                    print(f"  [WARN] {out_stem}: {n_frames} frames after conversion, too short")
                    tmp_16fps.unlink(missing_ok=True)
                    skipped_short += 1
                    continue

                # Split into segments
                seg_paths = split_video_segments(
                    tmp_16fps, videos_out, out_stem, n_frames
                )

                for seg_path in seg_paths:
                    txt_path = metas_out / seg_path.name.replace(".mp4", ".txt")
                    if not txt_path.exists():
                        txt_path.write_text(caption)
                    total_clips += 1

                tmp_16fps.unlink(missing_ok=True)
                total_episodes += 1

                if total_episodes % 50 == 0:
                    print(f"  Processed {total_episodes} episodes → {total_clips} clips...")

    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\nDone. {total_episodes} episodes → {total_clips} clips "
          f"(avg {total_clips/max(total_episodes,1):.1f} clips/episode)")
    print(f"  Skipped: {skipped_existing} existing, {skipped_short} too short")
    print(f"Output: {target}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare bimanual singleview — 16fps segmented clips"
    )
    parser.add_argument(
        "--source", type=str,
        default="datasets/bimanual/singleview",
    )
    parser.add_argument(
        "--target", type=str,
        default="datasets/bimanual_dreamgen_16fps",
    )
    args = parser.parse_args()

    source = Path(args.source)
    target = Path(args.target)
    assert source.exists(), f"Source not found: {source}"
    print(f"Preparing bimanual singleview (segmented) @ {TARGET_FPS}fps")
    print(f"  Source: {source}")
    print(f"  Target: {target}")
    print(f"  Segment target: {SEGMENT_TARGET} frames ({SEGMENT_TARGET/TARGET_FPS:.1f}s)")
    prepare(source, target)


if __name__ == "__main__":
    main()
