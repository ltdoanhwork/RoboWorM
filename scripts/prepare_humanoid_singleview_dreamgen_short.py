"""Prepare batch_input.json for DreamGen humanoid singleview — SHORT generation (2-3 chunks only).

Instead of rolling out the full episode, this caps generation to 2 or 3 chunks:
  - 2 chunks: 185 frames @ 16fps ≈ 11.6s  (for shorter episodes)
  - 3 chunks: 277 frames @ 16fps ≈ 17.3s  (for longer episodes)

Threshold logic:
  - Source episode ≤ THRESHOLD_FRAMES (at source_fps=20) → 2 chunks
  - Source episode > THRESHOLD_FRAMES → 3 chunks

This covers ALL episodes in the dataset (not just those that naturally fit 2-3 chunks).

Output layout:
  sampling_dataset/dense/humanoid/input/singleview/dreamgen/{split}/
    first_frames/episode_XXXXXX.png   (reused if exists)
    batch_input_short_2chunks.json
    batch_input_short_3chunks.json

Usage:
    python scripts/prepare_humanoid_singleview_dreamgen_short.py
    python scripts/prepare_humanoid_singleview_dreamgen_short.py --threshold-seconds 10
    python scripts/prepare_humanoid_singleview_dreamgen_short.py --max-episodes 20
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FPS = 16
SOURCE_FPS = 20.0
HUMANOID_CAM = "observation.images.ego_view_freq20"
HUMANOID_GENERIC_CAPTION = "A GR-1 humanoid robot performing a manipulation task."


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


def extract_first_frame(video_path: Path, out_png: Path, width: int, height: int) -> bool:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path),
         "-vf", f"scale={width}:{height}", "-vframes", "1", "-q:v", "2",
         str(out_png)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0 and out_png.exists()


def load_captions(split_dir: Path) -> dict[int, str]:
    captions_dir = split_dir / "captions"
    ep_caption: dict[int, str] = {}
    if not captions_dir.exists():
        return ep_caption
    for txt_file in captions_dir.glob("*.txt"):
        ep_match = re.search(r"(\d+)$", txt_file.stem)
        if ep_match:
            ep_caption[int(ep_match.group(1))] = txt_file.read_text().strip()
    return ep_caption


def find_episodes(split_dir: Path) -> list[tuple[int, Path]]:
    videos_root = split_dir / "videos"
    results = []
    for mp4 in sorted(videos_root.glob(f"chunk-*/{HUMANOID_CAM}/*.mp4")):
        ep_match = re.search(r"(\d+)$", mp4.stem)
        if ep_match:
            results.append((int(ep_match.group(1)), mp4))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Prepare short (2-3 chunk) batch_input for DreamGen humanoid singleview"
    )
    parser.add_argument("--datasets-root", type=str,
                        default=str(PROJECT_ROOT / "datasets" / "humanoid" / "singleview"))
    parser.add_argument("--save-root", type=str,
                        default=str(PROJECT_ROOT / "sampling_dataset" / "dense"))
    parser.add_argument("--output-root", type=str, default="")
    parser.add_argument("--splits", nargs="+", default=["makovian", "non_makovian"])
    parser.add_argument("--iter-name", type=str, default="iter_000090000")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--threshold-seconds", type=float, default=10.0,
                        help="Episodes shorter than this (at source fps) get 2 chunks, longer get 3")
    args = parser.parse_args()

    threshold_source_frames = int(args.threshold_seconds * SOURCE_FPS)

    datasets_root = Path(args.datasets_root)
    save_root = Path(args.save_root)
    output_root = Path(args.output_root) if args.output_root else save_root

    for split in args.splits:
        split_dir = datasets_root / split
        if not split_dir.exists():
            print(f"[humanoid/{split}] Directory not found, skipping.")
            continue

        input_dir = save_root / "humanoid" / "input" / "singleview" / "dreamgen" / split
        output_dir = output_root / "humanoid" / "output" / "singleview" / "dreamgen" / split / args.iter_name
        frames_dir = input_dir / "first_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        episodes = find_episodes(split_dir)
        captions = load_captions(split_dir)

        if args.max_episodes:
            episodes = episodes[:args.max_episodes]

        chunks_groups: dict[int, list] = defaultdict(list)
        stats = {"2_chunks": 0, "3_chunks": 0, "skipped": 0}

        for ep_id, video_path in episodes:
            frame_png = frames_dir / f"episode_{ep_id:06d}.png"
            if not (args.skip_existing and frame_png.exists()):
                if not extract_first_frame(video_path, frame_png, 768, 432):
                    print(f"  [WARN] Failed to extract frame: {video_path.name}")
                    stats["skipped"] += 1
                    continue

            if not frame_png.exists():
                stats["skipped"] += 1
                continue

            n_source_frames = get_video_frame_count(video_path)
            if n_source_frames is None:
                n_chunks = 2
            elif n_source_frames <= threshold_source_frames:
                n_chunks = 2
            else:
                n_chunks = 3

            caption = captions.get(ep_id, HUMANOID_GENERIC_CAPTION)
            out_video = str(output_dir / f"episode_{ep_id:06d}.mp4")

            chunks_groups[n_chunks].append({
                "input_video": str(frame_png),
                "prompt": caption,
                "output_video": out_video,
            })
            stats[f"{n_chunks}_chunks"] += 1

        for n_chunks, items in sorted(chunks_groups.items()):
            batch_json = input_dir / f"batch_input_short_{n_chunks}chunks.json"
            with open(batch_json, "w", encoding="utf-8") as f:
                json.dump(items, f, indent=2, ensure_ascii=False)
            print(f"[humanoid/{split}] {len(items)} episodes → {batch_json.name}")

        print(f"[humanoid/{split}] Summary: "
              f"2-chunks={stats['2_chunks']}, 3-chunks={stats['3_chunks']}, "
              f"skipped={stats['skipped']}, threshold={args.threshold_seconds}s "
              f"({threshold_source_frames} frames @ {SOURCE_FPS}fps)")


if __name__ == "__main__":
    main()
