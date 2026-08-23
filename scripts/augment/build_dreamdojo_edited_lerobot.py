"""Build a mini LeRobot dataset for DreamDojo AUGMENT with the object swapped.

DreamDojo (single_arm singleview) is action-conditioned with an EMPTY prompt: it
uses only the first GT frame as visual conditioning plus the action stream, and
reconstructs GT from the source video for reference. To make a "harder" augmented
input we keep the parquet (states/actions) IDENTICAL and only swap the object in
every frame of observation.images.global, then package a mini LeRobot dataset
containing just the manifest-selected episodes.

Original episode indices are PRESERVED (parquet content is copied byte-for-byte;
only episodes.jsonl / info.json are filtered) so no parquet index remapping is
needed. The DreamDojo loader iterates whatever episodes are present.

IMPORTANT: the DreamDojo loader (groot_dreams/dataloader.py::is_lerobot_dataset)
decides a path is LeRobot only if it contains a known token such as "/single_arm/"
(or gr1/aloha/bimanual/humanoid). Keep "single_arm" in the --out path or the
loader silently falls back to a generic VideoDataset (which lacks .lerobot_dataset
and crashes the episode planner). The default --out already satisfies this.

Output:
  sampling_dataset/augment/_lerobot/single_arm/singleview/{subset}/
    meta/{info.json, episodes.jsonl, tasks.jsonl, modality.json}
    data/chunk-000/episode_XXXXXX.parquet         (copied)
    videos/chunk-000/observation.images.global/episode_XXXXXX.mp4  (edited)

Runs in models/dreamgen/.venv (needs diffusers/torch for editing).

Usage:
  python scripts/augment/build_dreamdojo_edited_lerobot.py \
    --manifest augment_manifest.json --subset makovian \
    --src datasets/single_arm/singleview \
    --out sampling_dataset/augment/_lerobot/single_arm/singleview \
    --seed 0 --use-mask
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edit_object import ObjectEditor, _try_load_sam, _diff_box, _feather, _composite  # noqa: E402
from edit_ctrlworld_views import read_video_frames, write_video_frames, edit_clip  # noqa: E402

CAM = "observation.images.global"


def find_episode_paths(src_split: Path, ep_id: int):
    """Return (parquet_path, video_path, chunk_id) for an episode id."""
    for chunk_dir in sorted((src_split / "data").glob("chunk-*")):
        pq = chunk_dir / f"episode_{ep_id:06d}.parquet"
        if pq.exists():
            chunk_id = int(chunk_dir.name.split("-")[1])
            vid = src_split / "videos" / f"chunk-{chunk_id:03d}" / CAM / f"episode_{ep_id:06d}.mp4"
            return pq, vid, chunk_id
    return None, None, None


def filter_jsonl(src: Path, dst: Path, keep_ids: set, key: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    kept = []
    if src.exists():
        with src.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get(key) in keep_ids:
                    kept.append(rec)
    with dst.open("w") as f:
        for rec in kept:
            f.write(json.dumps(rec) + "\n")
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="augment_manifest.json")
    ap.add_argument("--subset", choices=["makovian", "non_makovian"], required=True)
    ap.add_argument("--src", default="datasets/single_arm/singleview")
    ap.add_argument("--out", default="sampling_dataset/augment/_lerobot/single_arm/singleview")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--image-guidance", type=float, default=1.8)
    ap.add_argument("--use-mask", action="store_true")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--all-frames", action="store_true",
                    help="Edit every frame (default: only frame 0, the only "
                         "visual condition DreamDojo's autoregressive rollout uses).")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    entries = manifest.get("dreamdojo_single_arm", {}).get(args.subset, [])
    if not entries:
        print(f"[build_dreamdojo] no manifest entries for dreamdojo_single_arm/{args.subset}")
        return

    src_split = Path(args.src) / args.subset
    out_split = Path(args.out) / args.subset
    keep_ids = {int(e["episode"]) for e in entries}

    # --- meta ---
    (out_split / "meta").mkdir(parents=True, exist_ok=True)
    for name in ["tasks.jsonl", "modality.json"]:
        s = src_split / "meta" / name
        if s.exists():
            shutil.copy2(s, out_split / "meta" / name)
    kept_eps = filter_jsonl(src_split / "meta" / "episodes.jsonl",
                            out_split / "meta" / "episodes.jsonl",
                            keep_ids, "episode_index")

    # info.json with updated totals
    info = json.loads((src_split / "meta" / "info.json").read_text())
    total_frames = sum(int(e.get("length", 0)) for e in kept_eps)
    info["total_episodes"] = len(kept_eps)
    info["total_frames"] = total_frames
    if "splits" in info:
        info["splits"] = {"train": f"0:{len(kept_eps)}"}
    (out_split / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    # --- editor ---
    editor = ObjectEditor(args.model, args.steps, args.guidance, args.image_guidance)
    sam = _try_load_sam() if args.use_mask else None

    for e in entries:
        ep = int(e["episode"])
        instr = e["edit_instruction"]
        pq, vid, chunk_id = find_episode_paths(src_split, ep)
        if pq is None or vid is None or not vid.exists():
            print(f"  [WARN] episode {ep:06d}: parquet/video not found, skipping")
            continue

        # copy parquet unchanged
        dst_pq = out_split / "data" / f"chunk-{chunk_id:03d}" / f"episode_{ep:06d}.parquet"
        dst_pq.parent.mkdir(parents=True, exist_ok=True)
        if not dst_pq.exists():
            shutil.copy2(pq, dst_pq)

        # edit video (frame 0 only by default; DreamDojo conditions on frame 0)
        dst_vid = out_split / "videos" / f"chunk-{chunk_id:03d}" / CAM / f"episode_{ep:06d}.mp4"
        if dst_vid.exists():
            print(f"  episode {ep:06d}: edited video exists, skip")
            continue
        frames = read_video_frames(vid)
        edited = edit_clip(editor, frames, instr, args.seed, args.use_mask, sam,
                           first_frame_only=not args.all_frames)
        write_video_frames(edited, dst_vid, fps=args.fps)
        print(f"  episode {ep:06d}: '{instr}' -> {frames.shape[0]} frames -> {dst_vid}")

    print(f"[build_dreamdojo] done -> {out_split} ({len(kept_eps)} episodes)")


if __name__ == "__main__":
    main()
