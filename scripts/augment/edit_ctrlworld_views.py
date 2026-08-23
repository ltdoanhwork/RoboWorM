"""Build an AUGMENT Ctrl-World annotation tree with the object RECOLORED.

Ctrl-World is action-conditioned: `states`/`joints` drive the rollout and only the
FIRST frame of each view is used as visual conditioning (the history buffer then
grows from the model's own generated latents — later GT frames are never fed back).
So an augmented input keeps states/joints/actions byte-identical and recolors the
object in frame 0 of every view. That is ~clip_len times cheaper than editing all
frames and produces exactly the same conditioning signal.

Recolor (not "replace with a different object") is deliberate: replacement makes
the image editor invent a different object per view, so the 3D scene disagrees with
itself from frame 0. See .cursor/rules/augment-track-layout.mdc.

Handles all three embodiments (the prepared trees differ):
  single_arm : dataset_example/single_arm_multiview_{split}/, numeric stems ("000255")
  bimanual   : dataset_example/bimanual_multiview/,  stems carry split+task
  humanoid   : dataset_example/humanoid_multiview/,  ditto

Variants: bimanual/humanoid have fewer than 10 real episodes per split, so the
manifest reuses an episode with a different color under a distinct `variant` tag.
Each (episode, variant) becomes its OWN annotation `{episode}__{variant}` in the
augment tree, so the inference script emits one output per variant.

Runs in the editor venv (.venv_edit, diffusers>=0.35 for Qwen-Image-Edit).

Usage:
  python scripts/augment/edit_ctrlworld_views.py \
    --config ctrlworld_bimanual_multiview --subset makovian \
    --dense-root models/Ctrl-World/dataset_example \
    --augment-root models/Ctrl-World/dataset_example_augment
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

# Which prepared dataset_example subdir each config reads. single_arm keeps one
# dir per category; bimanual/humanoid keep both categories in one dir (the
# category is embedded in the annotation stem instead).
#
# bimanual must be the *_5fps tree: that is what the dense outputs were produced
# from (dense num_frames_gt matches its annotation video_length) and what the
# finetuned checkpoint-280000 expects.
SUBSET_DIR = {
    "ctrlworld_single_arm_multiview": lambda split: f"single_arm_multiview_{split}",
    "ctrlworld_bimanual_multiview": lambda split: "bimanual_multiview_5fps",
    "ctrlworld_humanoid_multiview": lambda split: "humanoid_multiview",
    "ctrlworld_bimanual_multiview_4view_grid": lambda split: "bimanual_multiview_4view_grid",
    "ctrlworld_humanoid_multiview_4view_grid": lambda split: "humanoid_multiview_4view_grid",
}


def find_annotation(dense_dir: Path, stem: str) -> tuple[Path, str] | tuple[None, None]:
    """Locate `{stem}.json` under annotation/{val,train} and report which split.

    single_arm keeps everything in `val/`; bimanual/humanoid put most episodes in
    `train/`. The augment annotation must land in the SAME split dir, because the
    inference scripts scan both and read the video paths relative to the tree.
    """
    for split in ("val", "train"):
        p = dense_dir / "annotation" / split / f"{stem}.json"
        if p.exists():
            return p, split
    return None, None


def read_video_frames(path: Path) -> np.ndarray:
    from decord import VideoReader, cpu
    vr = VideoReader(str(path), ctx=cpu(0), num_threads=2)
    try:
        return vr.get_batch(range(len(vr))).asnumpy()
    except AttributeError:
        return vr.get_batch(range(len(vr))).numpy()


def write_video_frames(frames: np.ndarray, path: Path, fps: int = 5) -> bool:
    """Encode frames to mp4 with the same codec settings as the prepare script."""
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1:3]
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
         "-an", str(path)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    proc.stdin.write(frames.astype(np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    return path.exists()


def edit_clip(editor, frames: np.ndarray, instruction: str, seed: int,
              use_mask: bool, sam, first_frame_only: bool = True) -> np.ndarray:
    """Recolor the object in the clip (frame 0 only by default).

    Frame 0 is Ctrl-World's ONLY visual condition, so editing more frames costs
    time without changing the conditioning. `--all-frames` exists for a rollout
    variant that would re-anchor on later GT frames.

    With masking on, the mask is computed PER FRAME (the object and arm move), so
    a reused frame-0 mask can't smear color into vacated regions; the fixed seed
    keeps per-frame edits stylistically stable.
    """
    n = frames.shape[0]
    edit_indices = [0] if first_frame_only else range(n)
    out = frames.copy()
    for i in edit_indices:
        orig = Image.fromarray(frames[i])
        edited = editor.edit(orig, instruction, seed)
        if use_mask:
            box = _diff_box(orig, edited)
            mask = None
            if box is not None:
                if sam is not None:
                    try:
                        mask = sam(frames[i], box)
                    except (RuntimeError, ValueError):
                        mask = None
                if mask is None:
                    mask = np.zeros(frames.shape[1:3], dtype=bool)
                    mask[box[1]:box[3], box[0]:box[2]] = True
            if mask is not None:
                edited = _composite(orig, edited, _feather(mask))
        out[i] = np.asarray(edited.convert("RGB"), dtype=np.uint8)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, choices=sorted(SUBSET_DIR),
                    help="Manifest section / augment config to build.")
    ap.add_argument("--subset", choices=["makovian", "non_makovian"], required=True)
    ap.add_argument("--manifest", default="augment_manifest.json")
    ap.add_argument("--dense-root", default="models/Ctrl-World/dataset_example")
    ap.add_argument("--augment-root", default="models/Ctrl-World/dataset_example_augment")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="qwen",
                    help="Editor backend (qwen recommended; ip2p smears color).")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--image-guidance", type=float, default=1.8)
    ap.add_argument("--use-mask", action="store_true")
    ap.add_argument("--edit-views", default="",
                    help="Comma list of view indices to edit. Default: all views.")
    ap.add_argument("--all-frames", action="store_true",
                    help="Edit every frame (default: only frame 0, the sole visual "
                         "condition Ctrl-World's rollout uses).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only build the first N manifest entries (quick verification).")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    entries = manifest.get(args.config, {}).get(args.subset, [])
    if not entries:
        print(f"[edit_ctrlworld] no manifest entries for {args.config}/{args.subset}")
        return
    if args.limit:
        entries = entries[:args.limit]

    subset_name = SUBSET_DIR[args.config](args.subset)
    dense_dir = Path(args.dense_root) / subset_name
    aug_dir = Path(args.augment_root) / subset_name
    if not (dense_dir / "annotation").exists():
        print(f"[edit_ctrlworld] dense annotation tree missing at {dense_dir}; "
              f"run the matching prepare_ctrlworld_*.py first.")
        sys.exit(1)

    edit_views = ([int(v) for v in args.edit_views.split(",") if v.strip()]
                  if args.edit_views else None)

    editor = None  # loaded lazily so a fully-cached run needs no GPU
    sam = None

    print(f"[edit_ctrlworld] {args.config}/{args.subset}: {len(entries)} entries "
          f"-> {aug_dir}  (model={args.model}, "
          f"{'all frames' if args.all_frames else 'frame 0 only'})")

    for e in entries:
        ep = str(e["episode"])
        variant = e.get("variant")
        # One annotation per (episode, variant) so reused episodes yield distinct
        # outputs; the inference script names its output dir after this stem.
        out_stem = f"{ep}__{variant}" if variant else ep
        instr = e["edit_instruction"]

        anno_src, split_dir = find_annotation(dense_dir, ep)
        if anno_src is None:
            print(f"  [WARN] {ep}: annotation not found under {dense_dir}/annotation/"
                  f"{{val,train}}, skipping")
            continue
        anno = json.loads(anno_src.read_text())
        anno_dst = aug_dir / "annotation" / split_dir / f"{out_stem}.json"
        fps = anno.get("fps") or 5

        print(f"  {out_stem}: {instr}")
        # Videos are stored per-variant so different colors can't overwrite one another.
        new_video_infos = []
        ok = True
        for vid_idx, vid_info in enumerate(anno["videos"]):
            src_video = dense_dir / vid_info["video_path"]
            rel = Path(vid_info["video_path"])
            dst_rel = rel.parent.parent / out_stem / rel.name if variant else rel
            dst_video = aug_dir / dst_rel
            new_video_infos.append({**vid_info, "video_path": str(dst_rel)})

            if dst_video.exists():
                print(f"    view {vid_idx}: exists, skip")
                continue
            if not src_video.exists():
                print(f"    [WARN] view {vid_idx}: source video missing ({src_video})")
                ok = False
                break
            if edit_views is not None and vid_idx not in edit_views:
                dst_video.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src_video), str(dst_video))
                print(f"    view {vid_idx}: copied unedited")
                continue

            if editor is None:
                editor = ObjectEditor(args.model, args.steps, args.guidance, args.image_guidance)
                sam = _try_load_sam() if args.use_mask else None
            frames = read_video_frames(src_video)
            edited = edit_clip(editor, frames, instr, args.seed, args.use_mask, sam,
                               first_frame_only=not args.all_frames)
            write_video_frames(edited, dst_video, fps=fps)
            print(f"    view {vid_idx}: {frames.shape[0]} frames -> {dst_video}")

        if not ok:
            continue

        anno_out = {**anno, "videos": new_video_infos}
        if e.get("new_instruction"):
            anno_out["texts"] = [e["new_instruction"]]
        anno_dst.parent.mkdir(parents=True, exist_ok=True)
        anno_dst.write_text(json.dumps(anno_out))

        # Provenance so an output can always be traced back. NOTE: this must NOT
        # live in annotation/val — the infer scripts glob `annotation/val/*.json`
        # to discover episodes and would treat it as a (broken) extra episode.
        prov = aug_dir / "augment_info" / f"{out_stem}.json"
        prov.parent.mkdir(parents=True, exist_ok=True)
        prov.write_text(json.dumps({
            "config": args.config,
            "subset": args.subset,
            "source_episode": ep,
            "variant": variant,
            "object": e.get("object"),
            "edit_instruction": instr,
            "source_instruction": e.get("source_instruction"),
            "new_instruction": e.get("new_instruction"),
            "edit_model": args.model,
            "seed": args.seed,
            "first_frame_only": not args.all_frames,
        }, indent=2, ensure_ascii=False))

    print(f"[edit_ctrlworld] done -> {aug_dir}")


if __name__ == "__main__":
    main()
