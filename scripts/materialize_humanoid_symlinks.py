#!/usr/bin/env python3
"""
Replace symlinks under datasets/humanoid/singleview with real file copies.

datasets/humanoid/multiview currently has no symlinks; this script only touches
singleview by default.

Run from repo root:
  python scripts/materialize_humanoid_symlinks.py [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def collect_file_symlinks(base: Path) -> list[Path]:
    out: list[Path] = []
    for root, dirs, files in os.walk(base, followlinks=False):
        for name in list(dirs):
            p = Path(root) / name
            if p.is_symlink():
                dirs.remove(name)
        for name in files:
            p = Path(root) / name
            if p.is_symlink():
                out.append(p)
    return out


def iter_dir_symlinks(base: Path) -> list[Path]:
    out: list[Path] = []
    for root, dirs, _files in os.walk(base, followlinks=False):
        for name in list(dirs):
            p = Path(root) / name
            if p.is_symlink():
                out.append(p)
                dirs.remove(name)
    return out


def materialize_files(paths: list[Path], dry_run: bool) -> int:
    err = 0
    for i, p in enumerate(sorted(paths, key=str), start=1):
        if not p.is_symlink():
            continue
        tgt = p.resolve()
        if not tgt.exists():
            print(f"Missing target for {p}: {tgt}", file=sys.stderr)
            err += 1
            continue
        if not tgt.is_file():
            print(f"Expected file target for {p}, got {tgt}", file=sys.stderr)
            err += 1
            continue
        if dry_run:
            if i <= 5 or i == len(paths):
                print(f"[dry-run] copy file {tgt} -> {p}")
            continue
        os.unlink(p)
        shutil.copy2(tgt, p)
        if i % 50 == 0 or i == len(paths):
            print(f"files: {i}/{len(paths)}", flush=True)
    return err


def materialize_dir(p: Path, dry_run: bool) -> int:
    if not p.is_symlink():
        return 0
    tgt = p.resolve()
    if not tgt.is_dir():
        print(f"Expected directory target for {p}, got {tgt}", file=sys.stderr)
        return 1
    if dry_run:
        print(f"[dry-run] copytree {tgt} -> {p}")
        return 0
    os.unlink(p)
    shutil.copytree(
        tgt,
        p,
        symlinks=False,
        copy_function=shutil.copy2,
        dirs_exist_ok=False,
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "datasets"
        / "humanoid"
        / "singleview",
        help="humanoid singleview dataset root",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    base: Path = args.root.resolve()
    if not base.is_dir():
        print(f"Not a directory: {base}", file=sys.stderr)
        return 1

    file_links = [p for p in collect_file_symlinks(base) if p.is_symlink()]
    dir_links = [p for p in iter_dir_symlinks(base) if p.is_symlink()]

    if args.dry_run:
        print(f"[dry-run] file symlinks: {len(file_links)}")
        print(f"[dry-run] directory symlinks: {len(dir_links)}")

    err = materialize_files(file_links, args.dry_run)
    for d in sorted(dir_links, key=str):
        err += materialize_dir(d, args.dry_run)

    if err:
        return 1
    if not args.dry_run:
        leftover = collect_file_symlinks(base) + [
            p for p in iter_dir_symlinks(base) if p.is_symlink()
        ]
        if leftover:
            print(f"Warning: {len(leftover)} symlink(s) remain", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
