"""Recolor the conditioning FIRST FRAME for the DreamGen augment branches.

DreamGen is text-conditioned and only sees one image: the first frame (a single
view for singleview configs, a stitched 2x2 grid for multiview). So an augmented
DreamGen input is exactly "one edited PNG + maybe a different prompt" — this
script produces the PNGs, the run scripts wire prompts and inference.

Source frames are REUSED from the dense input tree (layout rule: inputs are
prepared once under `dense/`); if one is missing it is rebuilt from the dataset
videos with the same geometry the dense prep used. Edited PNGs land in the
mirrored augment input tree:

  sampling_dataset/augment/{emb}/input/{view}/dreamgen/{split}/.../first_frames_edited/
      {leaf}.png          # leaf = dense stem + "__{branch}" (augment_paths.leaf_name)

Only the `edit_only` and `combo` branches need this; `prompt_only` keeps the
original frame. Recolor (not object replacement) keeps shape/position stable —
see .cursor/rules/augment-track-layout.mdc.

Runs in the editor venv (.venv_edit, diffusers>=0.35 for Qwen-Image-Edit).

Usage:
  python scripts/augment/edit_dreamgen_first_frames.py \
    --config dreamgen_single_arm_singleview --split makovian --branch edit_only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import augment_paths as AP  # noqa: E402
from edit_object import ObjectEditor  # noqa: E402

REPO = AP.REPO_ROOT
DATASETS = REPO / "datasets"

# 2x2 grid geometry, matching each dense prep exactly (grid = 2*quad).
#   single_arm multiview : 320x180 quads -> 640x360  (prepare_single_arm_multiview_dreamgen_batch.py)
#   bimanual multiview   : 384x216 quads -> 768x432  (prepare_bimanual_multiview_dreamgen_batch.py)
QUAD_SIZE = {
    "dreamgen_single_arm_multiview": (320, 180),
    "dreamgen_bimanual_multiview": (384, 216),
}

# Camera slots per multiview config, row-major [TL, TR, BL, BR]. `None` = black
# filler (single_arm only has 3 cameras), same as the dense prep.
GRID_CAMS = {
    "dreamgen_single_arm_multiview": [
        "observation.images.exterior_1_left",
        "observation.images.exterior_2_left",
        "observation.images.wrist_left",
        None,
    ],
    "dreamgen_bimanual_multiview": [
        "observation.images.cam_high",
        "observation.images.cam_low",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ],
}

# How each config's dense first frame is located / rebuilt.
SINGLEVIEW_SCALE = "640:480"


# Which subdir of the (dense) input dir holds the prepared first frames, and how
# the file is named there. Both differ per config because the dense prep scripts
# were written independently.
FRAME_SUBDIR = {
    "dreamgen_single_arm_singleview": "first_frames",
    "dreamgen_bimanual_singleview": "first_frames",
    "dreamgen_single_arm_multiview": "",     # grid PNGs sit directly in the split dir
    "dreamgen_bimanual_multiview": "",       # ditto, under the task dir
}


def _dense_frame(config: str, split: str, entry: dict) -> Path:
    """Path of the DENSE prepared first frame this augment edit starts from."""
    base = AP.dreamgen_input_dir(config, split, entry.get("task"), root=AP.DENSE_ROOT)
    sub = FRAME_SUBDIR[config]
    if sub:
        base = base / sub
    # The multiview grid PNGs are named by the bare episode id, not the output
    # stem (which for bimanual also carries the task).
    name = (f"episode_{int(entry['episode']):06d}"
            if config == "dreamgen_bimanual_multiview"
            else AP.episode_stem(config, entry))
    return base / f"{name}.png"


def _find_video(task_root: Path, cam: str, ep: int) -> Path | None:
    for chunk in sorted(task_root.glob("videos/chunk-*")):
        cand = chunk / cam / f"episode_{ep:06d}.mp4"
        if cand.exists():
            return cand
    return None


def _extract_first_frame(video: Path, out_png: Path, scale: str = SINGLEVIEW_SCALE) -> bool:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-i", str(video), "-vf", f"scale={scale}",
         "-vframes", "1", "-q:v", "2", str(out_png)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return res.returncode == 0 and out_png.exists()


def _build_grid_frame(config: str, task_root: Path, ep: int, out_png: Path) -> bool:
    """Stitch each camera's first frame into the 2x2 grid the dense prep produces.

    A `None` camera slot becomes a black quadrant via lavfi (single_arm has only
    3 cameras, so its bottom-right is intentionally black — the model was trained
    on exactly that grid).
    """
    quad_w, quad_h = QUAD_SIZE[config]
    inputs: list[str] = []
    filters: list[str] = []
    for idx, cam in enumerate(GRID_CAMS[config]):
        if cam is None:
            inputs += ["-f", "lavfi", "-i", f"color=c=black:s={quad_w}x{quad_h}:r=1"]
        else:
            vp = _find_video(task_root, cam, ep)
            if vp is None:
                return False
            inputs += ["-i", str(vp)]
        filters.append(f"[{idx}:v]scale={quad_w}:{quad_h}[v{idx}]")
    graph = (";".join(filters)
             + ";[v0][v1]hstack[top];[v2][v3]hstack[bottom];[top][bottom]vstack[out]")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(
        ["ffmpeg", "-nostdin", "-y"] + inputs
        + ["-filter_complex", graph, "-map", "[out]", "-vframes", "1", str(out_png)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return res.returncode == 0 and out_png.exists()


def _rebuild_dense_frame(config: str, split: str, entry: dict, out_png: Path) -> bool:
    ep = int(entry["episode"])
    if config == "dreamgen_single_arm_singleview":
        v = _find_video(DATASETS / "single_arm/singleview" / split,
                        "observation.images.global", ep)
        return bool(v) and _extract_first_frame(v, out_png)
    if config == "dreamgen_bimanual_singleview":
        v = _find_video(DATASETS / "bimanual/singleview" / split / entry["task"],
                        "observation.images.cam_high", ep)
        return bool(v) and _extract_first_frame(v, out_png)
    if config == "dreamgen_single_arm_multiview":
        return _build_grid_frame(config, DATASETS / "single_arm/multiview" / split, ep, out_png)
    if config == "dreamgen_bimanual_multiview":
        return _build_grid_frame(
            config, DATASETS / "bimanual/multiview" / split / entry["task"], ep, out_png)
    return False


def edited_dir(config: str, split: str, entry: dict) -> Path:
    """Where edited PNGs go: `first_frames_edited/` in the augment input tree.

    `AP.dreamgen_input_dir` owns the (dense-mirroring) layout; see its docstring
    for why it is not simply `augment_paths.input_dir`.
    """
    return AP.dreamgen_input_dir(config, split, entry.get("task")) / "first_frames_edited"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True,
                    choices=[k for k in AP.CONFIGS if k.startswith("dreamgen_")])
    ap.add_argument("--split", required=True, choices=["makovian", "non_makovian"])
    ap.add_argument("--branch", required=True, choices=["edit_only", "combo"],
                    help="prompt_only needs no edit (it keeps the original frame).")
    ap.add_argument("--manifest", default=str(REPO / "augment_manifest.json"))
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--image-guidance", type=float, default=1.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only edit the first N entries (quick verification).")
    ap.add_argument("--force", action="store_true", help="Re-edit even if the PNG exists.")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    entries = manifest.get(args.config, {}).get(args.split, {}).get(args.branch, [])
    if not entries:
        print(f"[edit_dreamgen] no entries for {args.config}/{args.split}/{args.branch}")
        return
    if args.limit:
        entries = entries[:args.limit]

    editor = None  # lazy: a fully-cached run needs no GPU
    done = skipped = failed = 0
    print(f"[edit_dreamgen] {args.config}/{args.split}/{args.branch}: {len(entries)} entries")

    for entry in entries:
        leaf = AP.leaf_name(args.config, entry, args.branch)
        out_png = edited_dir(args.config, args.split, entry) / f"{leaf}.png"
        if out_png.exists() and not args.force:
            print(f"  [SKIP] {leaf} (exists)")
            skipped += 1
            continue

        src = _dense_frame(args.config, args.split, entry)
        if not src.exists():
            print(f"  [build] dense first frame missing, rebuilding: {src.name}")
            if not _rebuild_dense_frame(args.config, args.split, entry, src):
                print(f"  [FAIL] could not build source frame for {leaf}")
                failed += 1
                continue

        if editor is None:
            editor = ObjectEditor(args.model, args.steps, args.guidance, args.image_guidance)
        from PIL import Image
        img = Image.open(src).convert("RGB")
        out = editor.edit(img, entry["edit_instruction"], args.seed)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        out.save(out_png)
        # Provenance so any generated video can be traced back to its edit.
        out_png.with_suffix(".json").write_text(json.dumps({
            "config": args.config,
            "split": args.split,
            "branch": args.branch,
            "leaf": leaf,
            "source_frame": str(src),
            "object": entry.get("object"),
            "edit_instruction": entry["edit_instruction"],
            "source_instruction": entry.get("source_instruction"),
            "prompt": entry.get("prompt"),
            "edit_model": args.model,
            "seed": args.seed,
        }, indent=2, ensure_ascii=False))
        print(f"  [OK] {leaf}  <- {entry['edit_instruction']}")
        done += 1

    print(f"[edit_dreamgen] done={done} skipped={skipped} failed={failed} "
          f"-> {edited_dir(args.config, args.split, entries[0])}")


if __name__ == "__main__":
    main()
