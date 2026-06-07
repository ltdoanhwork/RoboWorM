#!/usr/bin/env python3
"""Collect sparse attention (SiTo, ITM) generated videos into webpage/videos/.

Scans sampling_dataset/{sito,itm}/ for generated .mp4 files and creates
symlinks under webpage/videos/{sito,itm}/... mirroring the structure.

This ensures the web demo can serve these videos via the existing symlink.

Usage:
    python scripts/collect_sparse_attention_videos.py
    python scripts/collect_sparse_attention_videos.py --methods sito itm --embodiments humanoid single_arm
"""
import argparse
import os
from pathlib import Path


def collect_videos(project_root: Path, methods: list, embodiments: list, dry_run: bool = False):
    sampling_dir = project_root / "sampling_dataset"
    webpage_videos = project_root / "webpage" / "videos"

    if not webpage_videos.exists():
        if webpage_videos.is_symlink():
            print(f"webpage/videos is a symlink -> {os.readlink(webpage_videos)}")
            print("Videos will be served directly from sampling_dataset via the symlink.")
            return
        webpage_videos.mkdir(parents=True, exist_ok=True)

    if webpage_videos.is_symlink():
        target = os.readlink(webpage_videos)
        print(f"webpage/videos is already a symlink to: {target}")
        print("Sparse attention videos are accessible if they exist under sampling_dataset/")
        print("\nChecking available videos:")

        for method in methods:
            method_dir = sampling_dir / method
            if not method_dir.exists():
                print(f"  [{method}] directory not found")
                continue
            for emb in embodiments:
                output_dir = method_dir / emb / "output"
                if not output_dir.exists():
                    continue
                mp4_count = len(list(output_dir.rglob("*.mp4")))
                print(f"  [{method}/{emb}] {mp4_count} videos found")
        return

    total_linked = 0
    for method in methods:
        method_dir = sampling_dir / method
        if not method_dir.exists():
            print(f"[{method}] No sampling_dataset/{method}/ found, skipping.")
            continue

        for emb in embodiments:
            output_dir = method_dir / emb / "output"
            if not output_dir.exists():
                continue

            for mp4 in output_dir.rglob("*.mp4"):
                rel = mp4.relative_to(sampling_dir)
                link_path = webpage_videos / rel
                if link_path.exists() or link_path.is_symlink():
                    continue
                if dry_run:
                    print(f"  [dry-run] {link_path} -> {mp4}")
                else:
                    link_path.parent.mkdir(parents=True, exist_ok=True)
                    link_path.symlink_to(mp4)
                total_linked += 1

    print(f"Created {total_linked} symlinks under webpage/videos/")


def main():
    parser = argparse.ArgumentParser(description="Collect sparse attention videos for webpage demo")
    parser.add_argument("--methods", nargs="+", default=["sito", "itm"],
                        help="Acceleration methods to collect (default: sito itm)")
    parser.add_argument("--embodiments", nargs="+", default=["humanoid", "single_arm", "bimanual"],
                        help="Robot embodiments to scan (default: humanoid single_arm bimanual)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without creating symlinks")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    collect_videos(project_root, args.methods, args.embodiments, args.dry_run)


if __name__ == "__main__":
    main()
