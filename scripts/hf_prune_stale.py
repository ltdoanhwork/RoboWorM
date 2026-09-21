#!/usr/bin/env python3
"""
Prune stale files from the Hugging Face dataset repo so it EXACTLY mirrors the
local `sampling_dataset/` tree.

Why this is needed
------------------
`hf upload-large-folder` only ever ADDS or UPDATES files; it never deletes.
So once a path is renamed (e.g. `single_view/` -> `singleview/`), an old
checkpoint dir is dropped (`iter_000010000/`), or non-benchmark trees are
archived out of `sampling_dataset/`, the HF repo keeps carrying them forever.
The repo then becomes a SUPERSET of the local tree and its episode counts no
longer match the benchmark design scope.

Safety
------
- Deletes ONLY paths in (remote - local). Every candidate is re-checked against
  the local tree and the run aborts if any candidate still exists locally.
- Dry-run by default; requires --execute to actually delete.
- Deletions are batched into commits; HF git history still retains the blobs,
  and this repo keeps its own copies under `archive/`.

Usage
-----
    python scripts/hf_prune_stale.py                 # dry run: report only
    python scripts/hf_prune_stale.py --execute       # actually delete
    python scripts/hf_prune_stale.py --report out.txt
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

from huggingface_hub import HfApi
from huggingface_hub import CommitOperationDelete

DEFAULT_REPO = "doanh25032004/video_gen_physics"
DEFAULT_LOCAL = "sampling_dataset"
# Never walk into the uploader's own resume-state cache.
SKIP_DIRS = {".cache"}


def local_files(root: str) -> set[str]:
    """Relative paths of every file in the local tree (excluding SKIP_DIRS)."""
    out: set[str] = set()
    for cur, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            out.add(os.path.relpath(os.path.join(cur, f), root))
    return out


def episode_dir(path: str) -> str | None:
    """The `.../episode_XXXX` prefix of a path, if it has one."""
    parts = path.split("/")
    for i, seg in enumerate(parts):
        if seg.startswith("episode"):
            return "/".join(parts[: i + 1])
    return None


def summarize(paths: list[str], label: str, depth: int = 1, top: int = 15) -> None:
    counter = collections.Counter("/".join(p.split("/")[:depth]) for p in paths)
    print(f"\n=== {label} (grouped by depth {depth}) ===")
    for key, n in counter.most_common(top):
        print(f"  {n:7d}  {key}")
    if len(counter) > top:
        print(f"  ... and {len(counter) - top} more groups")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default=DEFAULT_REPO)
    ap.add_argument("--repo-type", default="dataset")
    ap.add_argument("--local", default=DEFAULT_LOCAL,
                    help="local folder that the repo must mirror")
    ap.add_argument("--execute", action="store_true",
                    help="actually delete (default is a dry run)")
    ap.add_argument("--batch", type=int, default=2000,
                    help="deletions per commit")
    ap.add_argument("--report", default=None,
                    help="write the full candidate list to this file")
    args = ap.parse_args()

    if not os.path.isdir(args.local):
        print(f"ERROR: local folder not found: {args.local}", file=sys.stderr)
        return 1

    api = HfApi()
    print(f"==> repo   : {args.repo_id} ({args.repo_type})")
    print(f"==> local  : {args.local}")

    print("\nListing remote files ...")
    remote = {f for f in api.list_repo_files(args.repo_id, repo_type=args.repo_type)
              if not f.startswith(".")}
    print("Scanning local files ...")
    local = local_files(args.local)

    missing = sorted(local - remote)
    stale = sorted(remote - local)

    print(f"\n  remote files            : {len(remote)}")
    print(f"  local files             : {len(local)}")
    print(f"  missing on remote       : {len(missing)}")
    print(f"  STALE on remote (prune) : {len(stale)}")

    if missing:
        print("\nWARNING: some local files are NOT on the remote. Upload first:")
        for p in missing[:10]:
            print("   MISSING:", p)
        print("   (pruning is still safe, but the repo is incomplete)")

    if not stale:
        print("\nRepo already mirrors the local tree exactly. Nothing to prune.")
        return 0

    # ---- Safety gate: nothing we delete may exist locally ----
    conflicts = [p for p in stale if os.path.exists(os.path.join(args.local, p))]
    if conflicts:
        print(f"\nABORT: {len(conflicts)} prune candidates still exist locally.",
              file=sys.stderr)
        for p in conflicts[:10]:
            print("   CONFLICT:", p, file=sys.stderr)
        return 2
    print("\nSafety check OK: no prune candidate exists in the local tree.")

    summarize(stale, "stale files by top-level dir", depth=1)
    summarize(stale, "stale files by method/embodiment", depth=2)

    stale_eps = {e for e in (episode_dir(p) for p in stale) if e}
    local_eps = {e for e in (episode_dir(p) for p in local) if e}
    print(f"\n  stale episode dirs fully removed : "
          f"{len(stale_eps - local_eps)}")

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w") as fh:
            fh.write("\n".join(stale) + "\n")
        print(f"  full candidate list written to   : {args.report}")

    if not args.execute:
        print("\nDRY RUN — nothing deleted. Re-run with --execute to prune.")
        return 0

    # ---- Delete in batches ----
    total = len(stale)
    batches = [stale[i:i + args.batch] for i in range(0, total, args.batch)]
    print(f"\nDeleting {total} files in {len(batches)} commit(s) ...")
    done = 0
    for bi, batch in enumerate(batches, 1):
        ops = [CommitOperationDelete(path_in_repo=p) for p in batch]
        api.create_commit(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            operations=ops,
            commit_message=(
                f"Prune stale files to mirror local sampling_dataset "
                f"({bi}/{len(batches)})"
            ),
        )
        done += len(batch)
        print(f"  [{bi}/{len(batches)}] deleted {done}/{total}")

    # ---- Re-verify ----
    print("\nRe-listing remote to verify ...")
    remote2 = {f for f in api.list_repo_files(args.repo_id, repo_type=args.repo_type)
               if not f.startswith(".")}
    left = remote2 - local
    gone = local - remote2
    print(f"  remote files now      : {len(remote2)}")
    print(f"  still stale           : {len(left)}")
    print(f"  local missing on remote: {len(gone)}")
    if not left and not gone:
        print("\nSUCCESS: remote now mirrors the local tree exactly.")
        return 0
    for p in sorted(left)[:10]:
        print("   STILL STALE:", p)
    for p in sorted(gone)[:10]:
        print("   MISSING    :", p)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
