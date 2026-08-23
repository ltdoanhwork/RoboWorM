#!/usr/bin/env python3
"""Build a BLIND public bundle for the deployed labeling site from data_for_web/.

Reads sampling_dataset/data_for_web/manifest.json (the full, internal record
with source_method / aug_tag / role / path — built by build_data_for_web.py)
and produces:

  sampling_dataset/data_for_web/clips/{id}.mp4      flat, opaque filenames
  sampling_dataset/data_for_web/catalog.public.json  blind catalog (no method/
                                                      role/path — safe to ship
                                                      to the deployed frontend)

The public catalog's "video" field points at the HuggingFace dataset CDN, so
the deployed static site (Vercel) never needs to host or symlink video files.

Usage:
    python scripts/build_public_labeling_bundle.py
    python scripts/build_public_labeling_bundle.py --hf-repo doanh25032004/video_gen_physics
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEST = REPO / "sampling_dataset" / "data_for_web"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=DEST)
    ap.add_argument("--hf-repo", default="doanh25032004/video_gen_physics")
    ap.add_argument("--hf-prefix", default="data_for_web/clips")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest_path = args.dest / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    items = manifest["items"]
    print(f"{len(items)} items in manifest")

    clips_dir = args.dest / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    base_url = f"https://huggingface.co/datasets/{args.hf_repo}/resolve/main/{args.hf_prefix}"

    public_items = []
    n_copied = n_missing = 0
    for it in items:
        item_id = it["id"]
        src = args.dest / it["web_path"]
        dest_clip = clips_dir / f"{item_id}.mp4"
        if not src.is_file():
            n_missing += 1
            continue
        if not args.dry_run and (not dest_clip.exists() or dest_clip.stat().st_size != src.stat().st_size):
            shutil.copy2(src, dest_clip)
        n_copied += 1
        public_items.append(
            {
                "id": item_id,
                "video": f"{base_url}/{item_id}.mp4",
                "embodiment": it["embodiment"],
                "view": it["view"],
                "model": it["model"],
                "split": it["split"],
                "task": it["task"],
                "prompt": it.get("prompt", ""),
            }
        )

    public_items.sort(key=lambda x: x["id"])
    catalog = {"n": len(public_items), "hf_repo": args.hf_repo, "items": public_items}
    if not args.dry_run:
        (args.dest / "catalog.public.json").write_text(json.dumps(catalog, indent=2) + "\n")

    print(f"clips copied: {n_copied}  missing: {n_missing}")
    print(f"catalog.public.json: {len(public_items)} items, video base = {base_url}")
    if args.dry_run:
        print("[--dry-run] no files written")
    else:
        print(f"\nNext: upload clips/ to the Hub, e.g.\n"
              f"  HF_TOKEN=... python scripts/upload_content_to_hf.py "
              f"{clips_dir} {args.hf_repo} --repo-type dataset --prefix {args.hf_prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
