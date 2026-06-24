"""Segment long bimanual multiview GRID videos into overlapping sub-clips.

The grid dataset (datasets/bimanual_multiview_grid_dreamgen) has long episodes
(mean ~31s, up to 100s) but training only sees a random 93-frame window per
step. This script slices each already-stitched 2x2 grid video into overlapping
fixed-length segments so every part of the episode becomes a training sample.

Strategy (sliding window):
    - window length:  --win    seconds (default 10)
    - stride:         --stride seconds (default 5, i.e. 50% overlap)
    - episodes <= --min-keep-whole seconds are kept whole as a single seg00.
    - a final tail window reaching the end is added if the last regular window
      does not cover it.

The prompt is task+layout level (not timestep level), so each segment simply
reuses the parent episode's caption verbatim. T5 embeddings are recomputed
per-file afterwards (filenames match each segment mp4).

Reads from:
    datasets/bimanual_multiview_grid_dreamgen/
        videos/{stem}.mp4
        metas/{stem}.txt

Writes to:
    datasets/bimanual_multiview_grid_seg_dreamgen/
        videos/{stem}__seg{kk:02d}.mp4
        metas/{stem}__seg{kk:02d}.txt

Usage:
    python finetuning/dreamgen/scripts/segment_bimanual_multiview_grid_dreamgen.py
    python finetuning/dreamgen/scripts/segment_bimanual_multiview_grid_dreamgen.py \
        --win 10 --stride 5 --min-keep-whole 10
"""

import argparse
import json
import subprocess
from pathlib import Path


def probe_duration(video_path: Path) -> float:
    """Return video duration in seconds (0.0 on failure)."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=duration:stream=duration",
        "-of", "json",
        str(video_path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        info = json.loads(out)
    except Exception:
        return 0.0

    dur = 0.0
    fmt_dur = info.get("format", {}).get("duration")
    if fmt_dur:
        try:
            dur = float(fmt_dur)
        except (TypeError, ValueError):
            dur = 0.0
    if dur <= 0.0:
        for stream in info.get("streams", []):
            sd = stream.get("duration")
            if sd:
                try:
                    dur = float(sd)
                    break
                except (TypeError, ValueError):
                    continue
    return dur


def compute_windows(duration: float, win: float, stride: float, min_keep_whole: float) -> list[tuple[float, float]]:
    """Return list of (start, length) windows in seconds.

    - duration <= min_keep_whole -> single whole-clip window.
    - otherwise sliding windows of length `win` at stride `stride`, plus a tail
      window ending at `duration` if not already covered.
    """
    eps = 1e-6
    if duration <= min_keep_whole + eps:
        return [(0.0, duration)]

    windows: list[tuple[float, float]] = []
    start = 0.0
    while start + win <= duration + eps:
        windows.append((start, win))
        start += stride

    if not windows:
        # win longer than duration (shouldn't happen since duration > min_keep_whole
        # and min_keep_whole >= win typically); fall back to whole clip.
        return [(0.0, duration)]

    last_start, last_len = windows[-1]
    if last_start + last_len < duration - eps:
        # add a tail window that reaches the end; clamp start so length >= win
        tail_start = max(0.0, duration - win)
        if tail_start > last_start + eps:
            windows.append((tail_start, duration - tail_start))
    return windows


def cut_segment(src: Path, dst: Path, start: float, length: float) -> bool:
    """Cut [start, start+length] from src into dst (re-encode, frame-accurate)."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(src),
        "-t", f"{length:.3f}",
        "-c:v", "libx264", "-preset", "fast",
        "-crf", "18", "-pix_fmt", "yuv420p",
        "-an", str(dst),
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0 and dst.exists()


def segment(source: Path, target: Path, win: float, stride: float, min_keep_whole: float) -> None:
    src_videos = source / "videos"
    src_metas = source / "metas"
    assert src_videos.exists(), f"No videos/ in {source}"

    videos_out = target / "videos"
    metas_out = target / "metas"
    videos_out.mkdir(parents=True, exist_ok=True)
    metas_out.mkdir(parents=True, exist_ok=True)

    mp4_files = sorted(src_videos.glob("*.mp4"))
    print(f"Found {len(mp4_files)} source grid videos in {src_videos}")

    total_segs = 0
    skipped = 0
    failed = 0
    n_whole = 0

    for idx, mp4 in enumerate(mp4_files):
        stem = mp4.stem
        meta_src = src_metas / f"{stem}.txt"
        caption = meta_src.read_text() if meta_src.exists() else ""

        duration = probe_duration(mp4)
        if duration <= 0.0:
            print(f"  WARNING: could not probe duration for {mp4.name}, skipping")
            failed += 1
            continue

        windows = compute_windows(duration, win, stride, min_keep_whole)
        if len(windows) == 1 and abs(windows[0][1] - duration) < 1e-3:
            n_whole += 1

        for kk, (start, length) in enumerate(windows):
            out_stem = f"{stem}__seg{kk:02d}"
            dst_mp4 = videos_out / f"{out_stem}.mp4"
            dst_txt = metas_out / f"{out_stem}.txt"

            if dst_mp4.exists() and dst_txt.exists():
                skipped += 1
                continue

            if not dst_mp4.exists():
                if not cut_segment(mp4, dst_mp4, start, length):
                    print(f"    FAILED: {out_stem} ({start:.1f}s +{length:.1f}s)")
                    failed += 1
                    continue
            if not dst_txt.exists():
                dst_txt.write_text(caption)
            total_segs += 1

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(mp4_files)} episodes, {total_segs} segments so far...")

    print(
        f"\nDone. Created {total_segs} segments "
        f"({n_whole} episodes kept whole), skipped {skipped} existing, {failed} failed."
    )
    print(f"Output: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=str,
        default="datasets/bimanual_multiview_grid_dreamgen",
        help="Root of the (already-stitched) grid DreamGen dataset",
    )
    parser.add_argument(
        "--target", type=str,
        default="datasets/bimanual_multiview_grid_seg_dreamgen",
        help="Output directory for the segmented dataset",
    )
    parser.add_argument("--win", type=float, default=10.0, help="Segment length (seconds)")
    parser.add_argument("--stride", type=float, default=5.0, help="Sliding stride (seconds)")
    parser.add_argument(
        "--min-keep-whole", type=float, default=10.0,
        help="Episodes <= this duration (s) are kept whole, not split",
    )
    args = parser.parse_args()

    source = Path(args.source)
    target = Path(args.target)
    assert source.exists(), f"Source not found: {source}"
    segment(source, target, args.win, args.stride, args.min_keep_whole)


if __name__ == "__main__":
    main()
