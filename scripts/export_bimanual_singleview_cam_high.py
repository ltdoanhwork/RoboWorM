#!/usr/bin/env python3
"""
Export cam_high-only layout from bimanual multiview LeRobot datasets.

Creates:

  <singleview_root>/<makovian|non_makovian>/<task_name>/
    videos/chunk-XXX/observation.images.cam_high/episode_YYYYYY.mp4
      -> symlink (default) or copy from multiview .../videos/...

Optional shared episode parquet (same bytes as multiview):

    data -> symlink to <multiview_root>/.../<task>/data

Optional LeRobot meta aligned to single-view paths:

    meta/info.json    (features + video_path point at videos/..., cam_high only)
    meta/modality.json
    meta/episodes.jsonl -> symlink to source
    meta/tasks.jsonl    -> symlink to source
    meta/stats.json     -> symlink to source

Example:

  python scripts/export_bimanual_singleview_cam_high.py \\
    --multiview-root datasets/bimanual/multiview \\
    --singleview-root datasets/bimanual/singleview
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

CAM_HIGH_KEY = "observation.images.cam_high"

# Output folder under each task (matches standard LeRobot layout name).
SINGLEVIEW_VIDEO_ROOT = "videos"


def _symlink_rel(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    rel = os.path.relpath(src.resolve(), start=dst.parent.resolve())
    dst.symlink_to(rel)


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def iter_task_dirs(multiview_root: Path, category: str) -> list[Path]:
    base = multiview_root / category
    if not base.is_dir():
        return []
    out = []
    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        if p.name.startswith("."):
            continue
        if ".tmp." in p.name:
            continue
        out.append(p)
    return out


def strip_info_for_single_view(info: dict) -> dict:
    feats = info.get("features") or {}
    drop = [
        k
        for k in feats
        if k.startswith("observation.images.") and k != CAM_HIGH_KEY
    ]
    for k in drop:
        del feats[k]
    info["features"] = feats
    info["video_path"] = (
        f"{SINGLEVIEW_VIDEO_ROOT}/chunk-{{episode_chunk:03d}}/{{video_key}}/episode_{{episode_index:06d}}.mp4"
    )
    te = info.get("total_episodes")
    if isinstance(te, int):
        info["total_videos"] = te
    return info


def strip_modality(mod: dict) -> dict:
    vid = mod.get("video") or {}
    if "cam_high" in vid:
        mod["video"] = {"cam_high": vid["cam_high"]}
    return mod


def export_task(
    *,
    src_task: Path,
    dst_task: Path,
    link_data: bool,
    write_meta: bool,
    use_symlink: bool,
    dry_run: bool,
) -> tuple[int, int]:
    """Returns (n_linked_files, n_chunks)."""
    src_vid_root = src_task / "videos"
    cam_rel = Path(CAM_HIGH_KEY)
    n_files = 0
    n_chunks = 0

    if not src_vid_root.is_dir():
        return 0, 0

    for chunk_dir in sorted(src_vid_root.glob("chunk-*")):
        if not chunk_dir.is_dir():
            continue
        cam_dir = chunk_dir / CAM_HIGH_KEY
        if not cam_dir.is_dir():
            continue
        n_chunks += 1
        rel_dst_chunk = dst_task / SINGLEVIEW_VIDEO_ROOT / chunk_dir.name / CAM_HIGH_KEY
        for vid in sorted(cam_dir.glob("episode_*.mp4")):
            dst = rel_dst_chunk / vid.name
            if dry_run:
                n_files += 1
                continue
            if use_symlink:
                _symlink_rel(vid, dst)
            else:
                _copy_file(vid, dst)
            n_files += 1

    if dry_run:
        return n_files, n_chunks

    if link_data and (src_task / "data").is_dir():
        dst_data = dst_task / "data"
        if dst_data.exists() or dst_data.is_symlink():
            if dst_data.is_symlink() or dst_data.is_file():
                dst_data.unlink()
            else:
                shutil.rmtree(dst_data)
        _symlink_rel(src_task / "data", dst_data)

    if write_meta:
        meta_src = src_task / "meta"
        meta_dst = dst_task / "meta"
        meta_dst.mkdir(parents=True, exist_ok=True)
        for name in ("episodes.jsonl", "tasks.jsonl", "stats.json"):
            s = meta_src / name
            if s.is_file():
                _symlink_rel(s, meta_dst / name)
        info_path = meta_src / "info.json"
        if info_path.is_file():
            info = json.loads(info_path.read_text())
            info = strip_info_for_single_view(info)
            (meta_dst / "info.json").write_text(json.dumps(info, indent=2) + "\n")
        mod_path = meta_src / "modality.json"
        if mod_path.is_file():
            mod = json.loads(mod_path.read_text())
            mod = strip_modality(mod)
            (meta_dst / "modality.json").write_text(json.dumps(mod, indent=2) + "\n")

    return n_files, n_chunks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--multiview-root",
        type=Path,
        default=Path("datasets/bimanual/multiview"),
        help="Root containing makovian/ and non_makovian/ task folders.",
    )
    ap.add_argument(
        "--singleview-root",
        type=Path,
        default=Path("datasets/bimanual/singleview"),
        help="Output root (mirrors makovian | non_makovian layout).",
    )
    ap.add_argument(
        "--categories",
        nargs="+",
        choices=("makovian", "non_makovian"),
        default=("makovian", "non_makovian"),
        help="Which splits to export.",
    )
    ap.add_argument(
        "--copy",
        action="store_true",
        help="Copy mp4 files instead of symlinking (heavy). Default is symlink.",
    )
    ap.add_argument(
        "--no-link-data",
        action="store_true",
        help="Do not symlink shared-parquet data/ from multiview task.",
    )
    ap.add_argument(
        "--no-meta",
        action="store_true",
        help="Skip meta/ (only populate videos/).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without writing.",
    )
    args = ap.parse_args()

    mv = args.multiview_root.resolve()
    sv = args.singleview_root.resolve()
    use_symlink = not args.copy
    link_data = not args.no_link_data
    write_meta = not args.no_meta

    total_tasks = 0
    total_files = 0
    skipped = []

    for cat in args.categories:
        for src_task in iter_task_dirs(mv, cat):
            dst_task = sv / cat / src_task.name
            if args.dry_run:
                print(f"[dry-run] {cat}/{src_task.name} -> {dst_task}")
            nf, nc = export_task(
                src_task=src_task,
                dst_task=dst_task,
                link_data=link_data,
                write_meta=write_meta,
                use_symlink=use_symlink,
                dry_run=args.dry_run,
            )
            if nf == 0 and nc == 0:
                skipped.append(f"{cat}/{src_task.name} (no {CAM_HIGH_KEY} under videos/)")
                continue
            total_tasks += 1
            total_files += nf
            if not args.dry_run:
                print(f"{cat}/{src_task.name}: {nf} files, {nc} chunk(s)")

    print(
        f"\nDone: {total_tasks} tasks, {total_files} video files "
        f"({'symlink' if use_symlink else 'copy'})."
    )
    if skipped:
        print(f"Skipped ({len(skipped)}):")
        for s in skipped[:20]:
            print(f"  - {s}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")


if __name__ == "__main__":
    main()
