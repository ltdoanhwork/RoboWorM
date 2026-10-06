#!/usr/bin/env python3
"""Upload a directory's CONTENT (children at repo root) to a HuggingFace repo.

Why not `hf upload-large-folder`: it commits in small batches, and the Hub allows
only 128 commits/hour/repo -- a 100k-file tree then stalls in a 429 retry loop.
This uploader groups files into few, large commits (default 3000 files / 20GB per
commit), so a 100k-file tree needs ~35 commits instead of thousands.

Already-present remote files are skipped (matched on path + size), so re-running
resumes and is cheap.

    python3 scripts/tools/hf/upload_content_to_hf.py sampling_dataset doanh25032004/video_gen_physics --repo-type dataset

Env: HF_TOKEN (required), HF_HUB_DISABLE_XET=1 (set automatically).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# The Xet backend fails on this parallel filesystem ("MerkleDB Shard error").
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import CommitOperationAdd, HfApi  # noqa: E402
from huggingface_hub.utils import HfHubHTTPError  # noqa: E402

SKIP_DIRS = {"__pycache__", ".git", ".venv", ".venv_svg", ".cache", ".ipynb_checkpoints", "tmp"}
SKIP_FILES = {".env", ".DS_Store"}


def local_files(src: str) -> list[tuple[str, int]]:
    """(path_in_repo, size) for every real file under src, src itself stripped."""
    out = []
    for dp, dns, fns in os.walk(src):
        dns[:] = [d for d in dns if d not in SKIP_DIRS and not d.endswith(".egg-info")]
        for f in fns:
            if f in SKIP_FILES or f.endswith((".pyc", ".pyo")) or f.startswith(".nfs"):
                continue
            p = os.path.join(dp, f)
            if os.path.islink(p):
                continue
            try:
                out.append((os.path.relpath(p, src), os.path.getsize(p)))
            except OSError:
                continue
    out.sort()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("repo_id")
    ap.add_argument("--repo-type", default="dataset", choices=["dataset", "model"])
    ap.add_argument("--files-per-commit", type=int, default=1000)
    ap.add_argument("--bytes-per-commit", type=int, default=20 * 1024**3)
    ap.add_argument("--max-commits", type=int, default=110,
                    help="stop before the Hub's 128 commits/hour limit")
    ap.add_argument("--sleep-between", type=float, default=45.0,
                    help="pause between commits; the Hub also rate-limits LFS "
                         "object verification, which 429s on sustained bursts")
    ap.add_argument("--prefix", default="",
                    help="Prepend this path to every file in the repo (e.g. data_for_web)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not os.environ.get("HF_TOKEN"):
        print("HF_TOKEN is required", file=sys.stderr)
        return 2
    src = a.src.rstrip("/")
    api = HfApi()

    prefix = a.prefix.strip("/")
    def repo_path(rel: str) -> str:
        return f"{prefix}/{rel}" if prefix else rel

    print(f"=== content upload: {src}/* -> {a.repo_id} ({a.repo_type})"
          f"{' prefix='+prefix if prefix else ''} ===")
    api.create_repo(a.repo_id, repo_type=a.repo_type, exist_ok=True)

    print("[1/3] scanning local files")
    local = local_files(src)
    print(f"      {len(local)} files, {sum(s for _, s in local) / 1024**3:.1f}G")

    print("[2/3] listing remote files")
    remote: dict[str, int] = {}
    try:
        info = api.repo_info(a.repo_id, repo_type=a.repo_type, files_metadata=True)
        remote = {s.rfilename: (s.size or -1) for s in info.siblings}
    except HfHubHTTPError as e:
        print(f"      could not list remote ({e.__class__.__name__}); assuming empty")
    print(f"      {len(remote)} files already on the Hub")

    # Skip only when the remote size matches; -1 means the Hub reported no size.
    todo = [(p, s) for p, s in local if remote.get(repo_path(p), None) not in (s, -1)]
    skipped = len(local) - len(todo)
    print(f"      {skipped} unchanged, {len(todo)} to upload "
          f"({sum(s for _, s in todo) / 1024**3:.1f}G)")
    if not todo:
        print("OK: repo already up to date")
        return 0

    batches, cur, cur_sz = [], [], 0
    for p, s in todo:
        if cur and (len(cur) >= a.files_per_commit or cur_sz + s > a.bytes_per_commit):
            batches.append(cur)
            cur, cur_sz = [], 0
        cur.append(p)
        cur_sz += s
    if cur:
        batches.append(cur)
    print(f"[3/3] {len(batches)} commit(s) planned")
    if a.dry_run:
        return 0

    done = 0
    for i, batch in enumerate(batches, 1):
        if done >= a.max_commits:
            print(f"    stopping at {done} commits to stay under the hourly limit; "
                  f"re-run to continue")
            return 1
        # Build the operations ourselves: passing thousands of allow_patterns to
        # upload_folder makes it match every file against every pattern (quadratic
        # and unusably slow at 100k files x 3000 patterns).
        ops = [CommitOperationAdd(path_in_repo=repo_path(p), path_or_fileobj=os.path.join(src, p))
               for p in batch]
        for attempt in range(1, 8):
            try:
                t0 = time.time()
                api.create_commit(
                    repo_id=a.repo_id,
                    repo_type=a.repo_type,
                    operations=ops,
                    commit_message=f"content batch {i}/{len(batches)} ({len(batch)} files)",
                )
                print(f"    [{i}/{len(batches)}] {len(batch)} files in "
                      f"{time.time() - t0:.0f}s", flush=True)
                done += 1
                break
            except (HfHubHTTPError, RuntimeError) as e:
                # RuntimeError wraps per-file LFS failures, which here are almost
                # always a 429 on .../info/lfs/objects/verify.
                text = f"{e}"
                code = getattr(getattr(e, "response", None), "status_code", None)
                if code == 429 or "429" in text or "Too Many Requests" in text:
                    wait = 600 * attempt
                    print(f"    [{i}/{len(batches)}] rate limited; waiting "
                          f"{wait // 60} min (attempt {attempt}/7)", flush=True)
                else:
                    wait = 60 * attempt
                    print(f"    [{i}/{len(batches)}] {e.__class__.__name__} {code}; "
                          f"retrying in {wait}s (attempt {attempt}/7)", flush=True)
                time.sleep(wait)
        else:
            print(f"    [{i}/{len(batches)}] FAILED after 7 attempts -- re-run to resume")
            return 1
        if i < len(batches) and a.sleep_between:
            time.sleep(a.sleep_between)

    print(f"OK: https://huggingface.co/"
          f"{'datasets/' if a.repo_type == 'dataset' else ''}{a.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
