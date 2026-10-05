#!/usr/bin/env python3
"""Turn generated multiview videos into the frame layout the VGGT-Omega
multiview-consistency metric expects.

The metric (``multiview_consistency.run_video_metrics``) does NOT read mp4s. It
reads::

    <episodes_root>/<episode_id>/view1/frame_0000.png
    <episodes_root>/<episode_id>/view2/frame_0000.png
    ...

Two input shapes are supported:

1. ``--layout {2x2,1x2,1x3,1x4}`` -- one *mosaic* mp4 per episode holding all
   views side by side (what the Ctrl-World 4view-grid / DreamGen multiview runs
   emit as ``pred_all_views.mp4``). view1/view2 are cropped from the mosaic.
2. ``--layout per_view`` -- one mp4 per view per episode, selected with
   ``--view-globs`` (e.g. ``view_0.mp4,view_1.mp4``).

Usage (batch over an output tree)::

    python metrics/multiview_consistency/extract_frames.py \
        --videos-root sampling_dataset/pisa/bimanual/output/multiview/ctrlworld_4view_grid \
        --out-root    sampling_dataset/multiview_consistency/frames/pisa/bimanual \
        --video-name  pred_all_views.mp4 \
        --layout 2x2 --fps 2

Usage (single video)::

    python metrics/multiview_consistency/extract_frames.py \
        --video-path /path/to/pred_all_views.mp4 \
        --out-dir    /path/to/frames/episode_000025 \
        --layout 2x2 --fps 2

Adapted from ``frame_extractor_one.py`` in the VideoWorldModelBenchmark tree
(see docs/multiview_consistency_metric.md).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from glob import glob
from pathlib import Path

MOSAIC_LAYOUTS = ("2x2", "1x2", "1x3", "1x4")
LAYOUTS = MOSAIC_LAYOUTS + ("per_view",)


def _mosaic_filter(layout: str, fps: float) -> str:
    """ffmpeg filter_complex cropping view1 + view2 out of a mosaic frame."""
    if layout == "2x2":  # top-left, top-right
        return (
            f"[0:v]fps={fps},split=2[a][b];"
            "[a]crop=iw/2:ih/2:0:0[v1];"
            "[b]crop=iw/2:ih/2:iw/2:0[v2]"
        )
    if layout == "1x2":
        return (
            f"[0:v]fps={fps},split=2[a][b];"
            "[a]crop=iw/2:ih:0:0[v1];"
            "[b]crop=iw/2:ih:iw/2:0[v2]"
        )
    if layout == "1x3":
        return (
            f"[0:v]fps={fps},split=2[a][b];"
            "[a]crop=iw/3:ih:0:0[v1];"
            "[b]crop=iw/3:ih:iw/3:0[v2]"
        )
    if layout == "1x4":
        return (
            f"[0:v]fps={fps},split=2[a][b];"
            "[a]crop=iw/4:ih:0:0[v1];"
            "[b]crop=iw/4:ih:iw/4:0[v2]"
        )
    raise ValueError(f"Unknown mosaic layout {layout!r}; choose from {MOSAIC_LAYOUTS}")


def _require_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found in PATH")


def _run_ffmpeg(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def _reset_view_dirs(episode_dir: Path, view_names: list[str]) -> list[Path]:
    dirs = []
    for vn in view_names:
        d = episode_dir / vn
        d.mkdir(parents=True, exist_ok=True)
        for old in glob(str(d / "frame_*.png")):
            os.remove(old)
        dirs.append(d)
    return dirs


def extract_mosaic(
    video_path: Path,
    episode_dir: Path,
    layout: str,
    fps: float,
    view_names: list[str],
) -> int:
    _require_ffmpeg()
    view_dirs = _reset_view_dirs(episode_dir, view_names)
    filter_complex = _mosaic_filter(layout, fps)

    def build(decoder: str | None) -> list[str]:
        cmd = ["ffmpeg", "-y", "-hwaccel", "none"]
        if decoder:
            cmd += ["-c:v", decoder]
        cmd += ["-i", str(video_path), "-filter_complex", filter_complex]
        for label, vd in zip(["v1", "v2"], view_dirs):
            cmd += ["-map", f"[{label}]", "-start_number", "0", str(vd / "frame_%04d.png")]
        return cmd

    r = _run_ffmpeg(build("libdav1d"))
    if r.returncode != 0:
        r = _run_ffmpeg(build(None))
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {video_path}\n{r.stderr[-800:]}")

    n = len(glob(str(view_dirs[0] / "frame_*.png")))
    if n == 0:
        raise RuntimeError(f"no frames written under {view_dirs[0]}")
    return n


def extract_per_view(
    video_paths: list[Path],
    episode_dir: Path,
    fps: float,
    view_names: list[str],
) -> int:
    _require_ffmpeg()
    view_dirs = _reset_view_dirs(episode_dir, view_names)
    counts = []
    for src, vd in zip(video_paths, view_dirs):
        cmd = [
            "ffmpeg", "-y", "-hwaccel", "none", "-i", str(src),
            "-vf", f"fps={fps}", "-start_number", "0", str(vd / "frame_%04d.png"),
        ]
        r = _run_ffmpeg(cmd)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {src}\n{r.stderr[-800:]}")
        counts.append(len(glob(str(vd / "frame_*.png"))))
    if not counts or min(counts) == 0:
        raise RuntimeError(f"no frames written under {episode_dir}")
    return min(counts)


def _episode_name(video_path: Path) -> str:
    """Prefer the parent folder name for generic filenames like pred_all_views.mp4."""
    stem = video_path.stem
    if stem in {"pred_all_views", "full_pred", "pred", "output", "video"}:
        return video_path.parent.name
    return stem


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video-path", help="Single input video.")
    src.add_argument("--videos-root", help="Tree to scan for --video-name (batch mode).")

    p.add_argument("--out-dir", help="Episode output dir (single mode).")
    p.add_argument("--out-root", help="Root for <episode>/view1,view2 (batch mode).")
    p.add_argument("--video-name", default="pred_all_views.mp4",
                   help="Filename to look for under --videos-root (batch mode).")
    p.add_argument("--layout", default="2x2", choices=LAYOUTS)
    p.add_argument("--view-globs", default="view_0.mp4,view_1.mp4",
                   help="Comma-separated per-view filenames (--layout per_view).")
    p.add_argument("--view-names", default="view1,view2",
                   help="Output subfolder names; must match --views of the metric.")
    p.add_argument("--fps", type=float, default=2.0, help="Frame sampling rate.")
    p.add_argument("--flat", action="store_true",
                   help="Batch mode: write <out_root>/<episode>/ instead of mirroring "
                        "the relative path under --videos-root.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip episodes that already have frames for every view.")
    p.add_argument("--limit", type=int, default=0, help="Batch mode: stop after N episodes.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    view_names = [v.strip() for v in args.view_names.split(",") if v.strip()]
    if len(view_names) != 2:
        raise SystemExit("--view-names must list exactly 2 views (the metric scores view pairs)")
    view_globs = [v.strip() for v in args.view_globs.split(",") if v.strip()]

    def do_one(sources: list[Path], episode_dir: Path) -> int:
        episode_dir.mkdir(parents=True, exist_ok=True)
        if args.layout == "per_view":
            return extract_per_view(sources, episode_dir, args.fps, view_names)
        return extract_mosaic(sources[0], episode_dir, args.layout, args.fps, view_names)

    # ---------------- single ----------------
    if args.video_path:
        vp = Path(args.video_path).resolve()
        if not vp.is_file():
            raise SystemExit(f"video not found: {vp}")
        out_dir = Path(args.out_dir or (vp.parent / "mvc_frames")).resolve()
        sources = [vp]
        if args.layout == "per_view":
            sources = [vp.parent / g for g in view_globs]
        n = do_one(sources, out_dir)
        print(f"[extract] {n} frames/view -> {out_dir}", file=sys.stderr)
        print(out_dir)
        return

    # ---------------- batch ----------------
    if not args.out_root:
        raise SystemExit("--out-root is required with --videos-root")
    videos_root = Path(args.videos_root).resolve()
    out_root = Path(args.out_root).resolve()

    if args.layout == "per_view":
        found = sorted({p.parent for p in videos_root.rglob(view_globs[0])})
    else:
        found = sorted(videos_root.rglob(args.video_name))
    if not found:
        raise SystemExit(f"no inputs matched under {videos_root}")

    done = failed = skipped = 0
    for item in found:
        if args.layout == "per_view":
            ep_dir_src = item
            sources = [ep_dir_src / g for g in view_globs]
            missing = [s for s in sources if not s.is_file()]
            if missing:
                print(f"[extract] SKIP {ep_dir_src}: missing {[m.name for m in missing]}", file=sys.stderr)
                skipped += 1
                continue
            name = ep_dir_src.name
            rel_parent = ep_dir_src.parent.relative_to(videos_root)
        else:
            sources = [item]
            name = _episode_name(item)
            rel_parent = item.parent.relative_to(videos_root)
            if item.parent.name == name:
                rel_parent = rel_parent.parent

        episode_dir = out_root / name if args.flat else out_root / rel_parent / name

        if args.skip_existing and all(
            (episode_dir / vn).is_dir() and any((episode_dir / vn).glob("frame_*.png"))
            for vn in view_names
        ):
            skipped += 1
            continue

        try:
            n = do_one(sources, episode_dir)
            done += 1
            print(f"[extract] {n:4d} frames/view  {episode_dir}", file=sys.stderr)
        except Exception as exc:  # keep going over a big tree
            failed += 1
            print(f"[extract] FAIL {sources[0]}: {exc}", file=sys.stderr)

        if args.limit and done >= args.limit:
            break

    print(f"[extract] done={done} skipped={skipped} failed={failed} -> {out_root}", file=sys.stderr)


if __name__ == "__main__":
    main()
