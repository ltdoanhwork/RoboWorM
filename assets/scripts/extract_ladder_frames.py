#!/usr/bin/env python3
"""Extract matched frames from the WorldCache quality ladder videos.

Two artifacts, both meant for building paper/web figures without re-decoding video:

  frames/<cell>/<rung>/<episode>/f<idx>.png
      Individual PNGs at MATCHED absolute frame indices. Every rung of an episode
      has the same length (verified), so the same index is the same moment across
      rungs -- that is what makes a rung-to-rung comparison legitimate.

  figures/<cell>__<episode>.png
      One ladder grid per episode: rows = rungs (best to broken, top to bottom),
      columns = timepoints, with the rung name and speedup burned into each row.
      This is the figure you actually show.

PNG (lossless) is deliberate: these are for figures, where JPEG ringing around the
smeared/collapsed regions would be mistaken for model artefacts.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
LADDER = ROOT / "sampling_dataset/data_for_web/qualitative/worldcache_quality_ladder"
CELLS = ["humanoid_singleview_dreamgen",
         "humanoid_multiview_bright_dreamgen",
         "humanoid_multiview_dreamgen"]
# Multiview episodes are folders; the prediction is the file we compare.
PRED_NAME = "full_pred.mp4"


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).is_file():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def videos_of(rung_dir: Path) -> dict[str, Path]:
    """episode label -> prediction video, for both cell layouts."""
    out = {}
    for mp4 in sorted(rung_dir.glob("*.mp4")):        # singleview: one file per ep
        out[mp4.stem] = mp4
    for sub in sorted(p for p in rung_dir.iterdir() if p.is_dir()):
        if (sub / PRED_NAME).is_file():               # multiview: one dir per ep
            out[sub.name] = sub / PRED_NAME
    return out


def rung_meta(rung_dir: Path) -> dict:
    p = rung_dir / "info.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frames", type=int, default=5)
    ap.add_argument("--clean", action="store_true", help="wipe frames/ and figures/ first")
    args = ap.parse_args()

    frames_root, figs_root = LADDER / "frames", LADDER / "figures"
    if args.clean:
        for d in (frames_root, figs_root):
            shutil.rmtree(d, ignore_errors=True)

    index: dict[str, dict] = {}
    n_png = 0

    for cell in CELLS:
        cell_dir = LADDER / cell
        if not cell_dir.is_dir():
            continue
        rungs = sorted(p.name for p in cell_dir.iterdir() if p.is_dir())
        metas = {r: rung_meta(cell_dir / r) for r in rungs}

        # episode -> rung -> decoded frames at the chosen indices
        picked: dict[str, dict[str, np.ndarray]] = {}
        idx_used: dict[str, list[int]] = {}

        for rung in rungs:
            for ep, vid in videos_of(cell_dir / rung).items():
                arr = np.stack([np.asarray(f) for f in iio.imiter(vid)])
                if ep not in idx_used:
                    # Matched indices, first and last always included.
                    idx_used[ep] = np.linspace(0, len(arr) - 1, args.n_frames).astype(int).tolist()
                idxs = idx_used[ep]
                if max(idxs) >= len(arr):
                    print(f"  !! {cell}/{rung}/{ep}: {len(arr)} frames, "
                          f"expected > {max(idxs)} -- rung lengths differ, skipping")
                    continue
                sel = arr[idxs]
                picked.setdefault(ep, {})[rung] = sel
                out_dir = frames_root / cell / rung / ep
                out_dir.mkdir(parents=True, exist_ok=True)
                for i, fr in zip(idxs, sel):
                    Image.fromarray(fr).save(out_dir / f"f{i:04d}.png")
                    n_png += 1

        # Per-episode ladder grid.
        for ep, by_rung in picked.items():
            rows, labels = [], []
            for rung in rungs:
                if rung in by_rung:
                    rows.append(np.concatenate(list(by_rung[rung]), axis=1))
                    m = metas.get(rung, {})
                    sp = m.get("speedup_vs_dense")
                    labels.append((rung.split("_", 1)[1], f"{sp}x" if sp else ""))
            if not rows:
                continue
            w = min(r.shape[1] for r in rows)
            grid = np.concatenate([r[:, :w] for r in rows], axis=0)
            row_h = rows[0].shape[0]

            # Labels go in a dedicated gutter, never on top of the frames: burning
            # text into the image hides exactly the pixels a reader is judging.
            fsz = max(15, row_h // 12)
            f_name, f_sp = font(fsz), font(int(fsz * 0.85))
            gutter = int(max(len(n) for n, _ in labels) * fsz * 0.62) + 24
            canvas = Image.new("RGB", (gutter + grid.shape[1], grid.shape[0]), (250, 250, 250))
            canvas.paste(Image.fromarray(grid), (gutter, 0))
            d = ImageDraw.Draw(canvas)
            for i, (name, sp) in enumerate(labels):
                y = i * row_h
                if i:  # hairline between rungs
                    d.line([(gutter, y), (canvas.width, y)], fill=(210, 210, 210), width=2)
                d.text((12, y + row_h // 2 - fsz), name, font=f_name, fill=(20, 20, 20))
                if sp:
                    d.text((12, y + row_h // 2 + 4), sp, font=f_sp, fill=(120, 120, 120))
            figs_root.mkdir(parents=True, exist_ok=True)
            canvas.save(figs_root / f"{cell}__{ep}.png")
            index.setdefault(cell, {})[ep] = {
                "frame_indices": idx_used[ep],
                "rungs": [f"{n} {s}".strip() for n, s in labels],
                "figure": f"figures/{cell}__{ep}.png",
            }
            print(f"  {cell}/{ep}: {len(labels)} rungs x {len(idx_used[ep])} frames")

    (frames_root / "index.json").write_text(json.dumps({
        "note": ("Frame indices are matched across rungs of the same episode: every "
                 "rung of an episode has identical length, so index i is the same "
                 "moment. Rows in figures/ go best -> broken, top -> bottom."),
        "n_frames_per_video": args.n_frames,
        "cells": index,
    }, indent=2))
    print(f"\n{n_png} PNG frames + {sum(len(v) for v in index.values())} ladder figures")
    print(f"frames : {frames_root}")
    print(f"figures: {figs_root}")


if __name__ == "__main__":
    main()
