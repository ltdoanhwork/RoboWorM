"""Download DROID dataset from HuggingFace in chunks with robust error handling.

Usage:
    python scripts/download_droid_dataset.py
    python scripts/download_droid_dataset.py --start_chunk 5 --end_chunk 20
    python scripts/download_droid_dataset.py --end_chunk 95 --max_workers 2
"""

import argparse
import os
import random
import time

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

REPO_ID = "cadene/droid_1.0.1"
DEFAULT_LOCAL_DIR = "datasets/droid_1.0.1_20chunks"
MAX_RETRIES = 30
BASE_BACKOFF = 30
MAX_BACKOFF = 600
INTER_CHUNK_DELAY = 30


def download_chunk(chunk_idx: int, local_dir: str, max_workers: int) -> bool:
    chunk = f"chunk-{chunk_idx:03d}"
    allow_patterns = [
        "meta/*",
        f"data/{chunk}/*",
        f"videos/{chunk}/**",
    ]

    print(f"\n{'='*60}")
    print(f"[Chunk {chunk_idx:03d}] Starting download...")
    print(f"{'='*60}")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                local_dir=local_dir,
                allow_patterns=allow_patterns,
                max_workers=max_workers,
                resume_download=True,
            )
            print(f"[Chunk {chunk_idx:03d}] Done (attempt {attempt})")
            return True

        except HfHubHTTPError as e:
            err_str = str(e)
            if "429" in err_str or "Too Many Requests" in err_str:
                wait = min(MAX_BACKOFF, BASE_BACKOFF * (2 ** (attempt - 1)))
                wait += random.uniform(0, wait * 0.2)
                print(f"[Chunk {chunk_idx:03d}] Rate limited (attempt {attempt}/{MAX_RETRIES}). "
                      f"Waiting {wait:.0f}s...")
                time.sleep(wait)
            elif "500" in err_str or "502" in err_str or "503" in err_str:
                wait = min(MAX_BACKOFF, BASE_BACKOFF * attempt)
                print(f"[Chunk {chunk_idx:03d}] Server error (attempt {attempt}/{MAX_RETRIES}): "
                      f"{err_str[:100]}. Waiting {wait:.0f}s...")
                time.sleep(wait)
            else:
                print(f"[Chunk {chunk_idx:03d}] HTTP error (attempt {attempt}/{MAX_RETRIES}): "
                      f"{err_str[:200]}")
                if attempt >= MAX_RETRIES:
                    return False
                time.sleep(BASE_BACKOFF * attempt)

        except (OSError, IOError, TimeoutError) as e:
            wait = min(MAX_BACKOFF, BASE_BACKOFF * attempt)
            print(f"[Chunk {chunk_idx:03d}] IO/Network error (attempt {attempt}/{MAX_RETRIES}): "
                  f"{type(e).__name__}: {e}. Waiting {wait:.0f}s...")
            time.sleep(wait)

        except Exception as e:
            wait = min(MAX_BACKOFF, BASE_BACKOFF * attempt)
            print(f"[Chunk {chunk_idx:03d}] Unexpected error (attempt {attempt}/{MAX_RETRIES}): "
                  f"{type(e).__name__}: {e}. Waiting {wait:.0f}s...")
            if attempt >= MAX_RETRIES:
                return False
            time.sleep(wait)

    print(f"[Chunk {chunk_idx:03d}] FAILED after {MAX_RETRIES} retries")
    return False


def main():
    parser = argparse.ArgumentParser(description="Download DROID dataset chunks")
    parser.add_argument("--start_chunk", type=int, default=0)
    parser.add_argument("--end_chunk", type=int, default=20,
                        help="Exclusive end chunk index (default 20)")
    parser.add_argument("--local_dir", type=str, default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--max_workers", type=int, default=2,
                        help="Parallel download threads per chunk (default 2)")
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)

    failed_chunks = []
    for i in range(args.start_chunk, args.end_chunk):
        success = download_chunk(i, args.local_dir, args.max_workers)
        if not success:
            failed_chunks.append(i)
            print(f"[WARNING] Chunk {i:03d} failed, continuing to next chunk...")

        if i < args.end_chunk - 1:
            time.sleep(INTER_CHUNK_DELAY)

    print(f"\n{'='*60}")
    print(f"Download complete. {args.end_chunk - args.start_chunk - len(failed_chunks)}"
          f"/{args.end_chunk - args.start_chunk} chunks succeeded.")
    if failed_chunks:
        chunks_str = " ".join(str(c) for c in failed_chunks)
        print(f"FAILED chunks: {chunks_str}")
        print(f"Re-run with: python {__file__} --start_chunk <N> --end_chunk <N+1>")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()