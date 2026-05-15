#!/usr/bin/env python3
"""
Replace symlinks under datasets/single_arm with real files.

- Regular files (mp4, parquet): group by resolved target; move target to the first
  link path, then hardlink remaining paths (same inode, no symlink).
- meta/ directory (4 symlinks → one source): hardlink the 5 meta files into each
  layout path so there is no symlink and data is not duplicated.

Run from repo root:
  python scripts/materialize_single_arm_symlinks.py [--dry-run]

After a full run without --dry-run, datasets/droid_1.0.1_20chunks will no longer
contain the moved video/parquet files (they live under single_arm). Re-extract
from upstream if you need droid again.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path


def collect_file_symlinks(base: Path) -> list[Path]:
    out: list[Path] = []
    for root, _dirs, files in os.walk(base, followlinks=False):
        for name in files:
            p = Path(root) / name
            if p.is_symlink():
                out.append(p)
    # directory symlinks appear as "dirs" in walk — handle meta separately
    return out


def iter_dir_symlinks_only(base: Path) -> list[Path]:
    out: list[Path] = []
    for root, dirs, _files in os.walk(base, followlinks=False):
        # do not descend into symlinked dirs
        for name in list(dirs):
            p = Path(root) / name
            if p.is_symlink():
                out.append(p)
                dirs.remove(name)  # do not traverse
    return out


def materialize_file_group(target: Path, links: list[Path], dry_run: bool) -> None:
    links = sorted(links, key=lambda x: str(x))
    primary = links[0]
    rest = links[1:]
    if not target.exists():
        raise FileNotFoundError(f"Missing target {target}")
    if not target.is_file():
        raise IsADirectoryError(f"Expected file: {target}")

    if dry_run:
        print(f"[dry-run] move {target} -> {primary}; hardlink -> {len(rest)} path(s)")
        return

    os.unlink(primary)
    os.replace(target, primary)
    for p in rest:
        if p.exists() or p.is_symlink():
            os.unlink(p)
        os.link(primary, p)


def materialize_meta(meta_links: list[Path], source_meta: Path, dry_run: bool) -> None:
    """All meta symlinks must point to the same source_meta directory."""
    meta_links = sorted(meta_links, key=lambda x: str(x))
    first = meta_links[0]
    sources = sorted(source_meta.iterdir(), key=lambda x: x.name)
    for s in sources:
        if not s.is_file():
            raise ValueError(f"unexpected non-file in meta: {s}")

    if dry_run:
        print(f"[dry-run] meta: hardlink {len(sources)} files into {len(meta_links)} dirs")
        return

    if not first.is_symlink():
        raise RuntimeError(f"expected symlink at {first}")
    os.unlink(first)
    first.mkdir(parents=True)
    for s in sources:
        os.link(s, first / s.name)

    for other in meta_links[1:]:
        if not other.is_symlink():
            raise RuntimeError(f"expected symlink at {other}")
        os.unlink(other)
        other.mkdir(parents=True)
        for s in sources:
            os.link(first / s.name, other / s.name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets" / "single_arm",
        help="single_arm dataset root",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    base: Path = args.root.resolve()
    if not base.is_dir():
        print(f"Not a directory: {base}", file=sys.stderr)
        return 1

    # --- file symlinks ---
    by_target: dict[Path, list[Path]] = defaultdict(list)
    for p in collect_file_symlinks(base):
        if not p.is_symlink():
            continue
        tgt = p.resolve()
        if tgt.is_file():
            by_target[tgt].append(p)

    # Deterministic order: process targets sorted by string
    for tgt in sorted(by_target.keys(), key=str):
        materialize_file_group(tgt, by_target[tgt], args.dry_run)

    # --- meta directory symlinks (only those under single_arm) ---
    dir_links = [p for p in iter_dir_symlinks_only(base) if p.name == "meta" and p.is_symlink()]
    if not dir_links:
        return 0
    resolved = {p.resolve() for p in dir_links}
    if len(resolved) != 1:
        print(
            "Refusing meta handling: multiple distinct targets: "
            + ", ".join(str(x) for x in sorted(resolved, key=str)),
            file=sys.stderr,
        )
        return 1
    source_meta = next(iter(resolved))
    if not source_meta.is_dir():
        print(f"meta target is not a directory: {source_meta}", file=sys.stderr)
        return 1
    materialize_meta(dir_links, source_meta, args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
