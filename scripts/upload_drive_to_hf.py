#!/usr/bin/env python3
"""Upload assets/drive to a HF dataset repo in small, resumable commits.

`hf upload-large-folder` stalls on this tree (5306 small files), so commit in
explicit batches instead: one commit per leaf group, skipping anything already
on the Hub. Safe to re-run — it only uploads what is missing.

    HF_TOKEN=... python3 scripts/upload_drive_to_hf.py [--repo ID] [--batch N]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.utils import HfHubHTTPError

REPO = Path(__file__).resolve().parents[1]
LOCAL = REPO / "assets" / "drive"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="doanh25032004/video-gen-physics-gallery")
    ap.add_argument("--batch", type=int, default=120, help="files per commit")
    ap.add_argument("--retries", type=int, default=4)
    args = ap.parse_args()

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    kw = dict(repo_id=args.repo, repo_type="dataset")

    # size-aware skip: a file already on the Hub is only skipped when its byte
    # size matches, so edited files (e.g. README) are re-pushed
    info = api.repo_info(files_metadata=True, **kw)
    have = {f.rfilename: f.size for f in info.siblings}
    files = sorted(p for p in LOCAL.rglob("*") if p.is_file())
    todo = [p for p in files
            if have.get(str(p.relative_to(LOCAL))) != p.stat().st_size]
    size = sum(p.stat().st_size for p in todo)
    print(f"local {len(files)} files | on hub {len(have)} | to upload {len(todo)} "
          f"({size / 2**30:.2f} GiB)", flush=True)
    if not todo:
        print("nothing to do")
        return

    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    done = 0
    t0 = time.time()
    for bi, batch in enumerate(batches, 1):
        ops = [CommitOperationAdd(path_in_repo=str(p.relative_to(LOCAL)),
                                  path_or_fileobj=str(p)) for p in batch]
        for attempt in range(1, args.retries + 1):
            try:
                api.create_commit(
                    operations=ops,
                    commit_message=f"Upload gallery batch {bi}/{len(batches)}",
                    **kw)
                break
            except (HfHubHTTPError, OSError) as e:
                if attempt == args.retries:
                    print(f"[FAIL] batch {bi}: {e}", file=sys.stderr, flush=True)
                    sys.exit(1)
                wait = 5 * attempt
                print(f"  [retry {attempt}] batch {bi} in {wait}s: "
                      f"{str(e)[:120]}", flush=True)
                time.sleep(wait)
        done += len(batch)
        el = time.time() - t0
        print(f"[{bi:3d}/{len(batches)}] committed {done}/{len(todo)} files "
              f"({el / 60:.1f} min, {done / max(el, 1):.1f} files/s)", flush=True)

    print(f"done in {(time.time() - t0) / 60:.1f} min -> "
          f"https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
