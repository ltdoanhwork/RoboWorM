"""Re-encode the pre-segmented humanoid multiview GRID clips from 10fps -> 16fps.

WHY: the humanoid multiview grid dataset (datasets/humanoid_multiview_grid_dreamgen)
was stitched at 10fps from 30fps source, and training conditions clips at their
native mp4 fps while the dataloader grabs 93 CONSECUTIVE frames. At 10fps only
132/248 clips even reach 93 frames, and the base checkpoint is model-480p-16fps.
Re-encoding to a true 16fps (same duration) makes a 93-frame window span 5.8s,
recovers ~204/248 clips, and matches the fps the model is conditioned on -> we
drop override_fps in the experiment. This mirrors the singleview/bimanual
`_16fps`/`_seg16` datasets.

WHAT: re-encode every segment mp4 to a true 16fps (ffmpeg -r 16), keeping the
same duration. Captions (metas/*.txt) and T5 embeddings (t5_xxl/*.pickle) are
semantic-only and do NOT depend on fps, so they are copied verbatim under the
same filename.

Reads from:
    datasets/humanoid_multiview_grid_seg_dreamgen/
        videos/{stem}.mp4
        metas/{stem}.txt
        t5_xxl/{stem}.pickle

Writes to:
    datasets/humanoid_multiview_grid_seg16_dreamgen/
        videos/{stem}.mp4   (16fps)
        metas/{stem}.txt     (copied)
        t5_xxl/{stem}.pickle (copied)

Usage:
    python finetuning/dreamgen/scripts/reencode_humanoid_multiview_seg_16fps.py
    python finetuning/dreamgen/scripts/reencode_humanoid_multiview_seg_16fps.py \
        --source datasets/humanoid_multiview_grid_seg_dreamgen \
        --target datasets/humanoid_multiview_grid_seg16_dreamgen \
        --workers 8
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

TARGET_FPS = 16
MIN_FRAMES = 93


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
    """Transcode source video to 16fps H.264 (same duration, resampled fps)."""
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src),
         "-r", str(TARGET_FPS),
         "-c:v", "libx264", "-preset", "fast", "-crf", "18",
         "-pix_fmt", "yuv420p", "-an", str(dst)],
        capture_output=True,
    )
    return result.returncode == 0


def process_one(args: tuple[str, str]) -> tuple[str, str]:
    """Re-encode a single clip. Returns (stem, status)."""
    src_str, dst_str = args
    src, dst = Path(src_str), Path(dst_str)
    stem = src.stem

    if dst.exists():
        n = get_video_frame_count(dst)
        if n is not None and n >= MIN_FRAMES:
            return stem, "skipped"
        dst.unlink(missing_ok=True)

    if not transcode_16fps(src, dst):
        dst.unlink(missing_ok=True)
        return stem, "failed_transcode"

    n_frames = get_video_frame_count(dst)
    if n_frames is None or n_frames < MIN_FRAMES:
        dst.unlink(missing_ok=True)
        return stem, f"too_short({n_frames})"

    return stem, "ok"


def reencode(source: Path, target: Path, workers: int) -> None:
    src_videos = source / "videos"
    src_metas = source / "metas"
    src_t5 = source / "t5_xxl"
    assert src_videos.exists(), f"No videos/ in {source}"

    videos_out = target / "videos"
    metas_out = target / "metas"
    t5_out = target / "t5_xxl"
    for d in (videos_out, metas_out, t5_out):
        d.mkdir(parents=True, exist_ok=True)

    mp4_files = sorted(src_videos.glob("*.mp4"))
    print(f"Found {len(mp4_files)} source segment videos in {src_videos}")
    print(f"Re-encoding to {TARGET_FPS}fps with {workers} workers -> {videos_out}")

    jobs = [(str(mp4), str(videos_out / mp4.name)) for mp4 in mp4_files]

    n_ok = n_skip = n_fail = 0
    ok_stems: list[str] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_one, job): job for job in jobs}
        done = 0
        for fut in as_completed(futures):
            stem, status = fut.result()
            done += 1
            if status in ("ok", "skipped"):
                ok_stems.append(stem)
                if status == "ok":
                    n_ok += 1
                else:
                    n_skip += 1
            else:
                n_fail += 1
                print(f"  [WARN] {stem}: {status}")
            if done % 100 == 0:
                print(f"  Processed {done}/{len(jobs)} (ok={n_ok} skip={n_skip} fail={n_fail})")

    # Copy captions + T5 embeddings verbatim for every successful video.
    n_meta = n_t5 = 0
    for stem in ok_stems:
        src_txt = src_metas / f"{stem}.txt"
        dst_txt = metas_out / f"{stem}.txt"
        if src_txt.exists() and not dst_txt.exists():
            shutil.copy2(src_txt, dst_txt)
            n_meta += 1
        src_pkl = src_t5 / f"{stem}.pickle"
        dst_pkl = t5_out / f"{stem}.pickle"
        if src_pkl.exists() and not dst_pkl.exists():
            shutil.copy2(src_pkl, dst_pkl)
            n_t5 += 1

    print(
        f"\nDone. videos: {n_ok} re-encoded, {n_skip} skipped, {n_fail} failed. "
        f"Copied {n_meta} metas, {n_t5} T5 embeddings."
    )
    print(f"Output: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=str,
        default="datasets/humanoid_multiview_grid_seg_dreamgen",
        help="Root of the existing 10fps segmented grid dataset",
    )
    parser.add_argument(
        "--target", type=str,
        default="datasets/humanoid_multiview_grid_seg16_dreamgen",
        help="Output directory for the 16fps re-encoded dataset",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of parallel ffmpeg workers",
    )
    args = parser.parse_args()

    source = Path(args.source)
    target = Path(args.target)
    assert source.exists(), f"Source not found: {source}"
    print(f"Re-encoding humanoid multiview grid segments @ {TARGET_FPS}fps")
    print(f"  Source: {source}")
    print(f"  Target: {target}")
    reencode(source, target, args.workers)


if __name__ == "__main__":
    main()
