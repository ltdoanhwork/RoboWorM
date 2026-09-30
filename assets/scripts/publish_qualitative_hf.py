#!/usr/bin/env python3
"""Publish the qualitative WorldCache quality ladder to a public HF dataset repo.

The folder is small (~112 MB, 39 mp4 + contact sheets + per-rung info.json), so a
single `upload_folder` call is enough -- no LFS bookkeeping needed by hand, the
hub client handles large files.

Auth: needs a WRITE token. Either
    hf auth login                       # interactive, stores ~/.cache/huggingface/token
or  export HF_TOKEN=hf_xxx              # picked up automatically
then
    python3 assets/scripts/publish_qualitative_hf.py            # dry run: shows plan
    python3 assets/scripts/publish_qualitative_hf.py --push     # actually upload
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "sampling_dataset/data_for_web/qualitative"
DEFAULT_REPO = "doanh25032004/video-world-acceleration-qualitative"

CARD = """---
license: apache-2.0
pretty_name: Video World Model Acceleration - Qualitative Quality Ladder
tags:
  - robotics
  - video-generation
  - world-model
  - diffusion-acceleration
  - kv-cache
size_categories:
  - n<1K
---

# Acceleration quality ladder (qualitative)

Videos showing what **WorldCache** (whole-step diffusion caching) does to a video
world model as it is pushed from a quality-preserving operating point all the way
to collapse. Two cells of a robot video-generation benchmark, same episodes at
every rung, so identically-named files are directly comparable.

Backbone: Cosmos-Predict2 **2B** DiT, post-trained on a GR1/AIRBOT humanoid
manipulation corpus, 480p, 35 denoising steps, autoregressive rollout.

## Layout

```
humanoid_singleview_dreamgen/          5 rungs, 3 episodes each (prediction mp4)
humanoid_multiview_bright_dreamgen/    4 rungs, 3 bright episodes  <- prefer this
humanoid_multiview_dreamgen/           4 rungs, 2 episodes (dark; kept for lineage)
                                       (full_pred / full_gt / full_merged + metrics)
figures/<cell>__<episode>.png          ladder grid: rows = rungs, cols = timepoints
frames/<cell>/<rung>/<episode>/*.png   the individual extracted frames (lossless)
frames/index.json                      which frame indices were taken
contact_sheet_*.png                    one-glance overview per cell
```

**Pre-extracted frames.** `figures/` has one ready-to-use ladder grid per episode
(rows top-to-bottom = dense -> broken, each labelled with its speedup), and
`frames/` has the same frames as individual lossless PNGs if you want to lay them
out yourself. Frame indices are **matched across rungs**: every rung of an episode
is exactly the same length, so index *i* is the same moment in every rung -- which
is what makes the row-to-row comparison meaningful. No need to decode video.

Every rung folder has an `info.json` with the exact WorldCache knobs, the measured
time, the speedup over dense, and the fidelity to the dense prediction.

**Two multiview cells, use the bright one.** The original pair of multiview tasks
turned out to be the darkest in the dataset (2x2 grid mean luma ~51/255, ~36%
near-black pixels, because their wrist cameras are unlit), which makes the
degradation hard to see. `humanoid_multiview_bright_dreamgen/` repeats the same
ladder on the three brightest distinct tasks (luma ~138/137/129, i.e. ~2.7x
brighter), spanning both 2x2 grid layouts and both splits. The dark cell is kept
because its numbers are what the tuning sweep was selected on.

| rung | meaning |
|---|---|
| `01_dense` | reference, caching disabled |
| `02_ok` | selected operating point, fidelity at parity with dense |
| `03_near_ok` | slight softening, all structure intact (singleview only) |
| `04_degraded` | visibly smeared, subject still locatable |
| `05_broken` | collapsed: no dense warm-up, every step skipped |

## Measurements

Quality is fidelity to the **dense prediction** of the same episode, not to ground
truth: the question is what the accelerator changed, not how good the model is.

Single-view (3 episodes, dense 259.4 s/episode):

| rung | thresh / ret / probe | s/ep | speedup | PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|---|
| `01_dense` | — | 259.4 | 1.00x | — | — | — |
| `02_ok` | 0.12 / 0.20 / 4 | 121.8 | 2.13x | 16.95 | 0.6445 | 0.2866 |
| `03_near_ok` | 0.20 / 0.15 / 4 | 97.8 | 2.65x | 15.16 | 0.5407 | 0.4037 |
| `04_degraded` | 0.30 / 0.10 / 4 | 87.2 | 2.98x | 11.64 | 0.3765 | 0.5705 |
| `05_broken` | 1.00 / 0.00 / 4 | 63.5 | 4.09x | 12.04 | 0.3951 | 0.5517 |

Multi-view **bright** (3 episodes, dense 118.3 s/segment, steady state):

| rung | thresh / ret / probe | s/seg | speedup | PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|---|
| `01_dense` | — | 118.3 | 1.00x | — | — | — |
| `02_ok` | 0.25 / 0.20 / 2 | 52.7 | 2.25x | 16.03 | 0.7182 | 0.3192 |
| `04_degraded` | 0.30 / 0.10 / 2 | 39.4 | 3.01x | 12.94 | 0.6262 | 0.3968 |
| `05_broken` | 1.00 / 0.00 / 2 | 18.1 | 6.53x | 13.96 | 0.6207 | 0.4317 |

Multi-view dark (2 episodes, dense 118.8 s/segment):

| rung | thresh / ret / probe | s/seg | speedup | PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|---|
| `01_dense` | — | 118.8 | 1.00x | — | — | — |
| `02_ok` | 0.25 / 0.20 / 2 | 53.3 | 2.23x | 18.68 | 0.7850 | 0.2716 |
| `04_degraded` | 0.30 / 0.10 / 2 | 42.9 | 2.77x | 15.05 | 0.6228 | 0.3868 |
| `05_broken` | 1.00 / 0.00 / 2 | 18.1 | 6.56x | 16.35 | 0.6469 | 0.3857 |

## Two findings worth knowing

**The cliff is the dense warm-up, not the skip threshold.** Dropping `ret_ratio`
from 0.15 to 0.10 costs 0.16 SSIM (0.541 -> 0.377) and buys only +0.33x. Raising
`rel_l1_thresh` degrades gracefully by comparison. If you tune one knob, tune the
warm-up.

**Reference metrics stop ranking once the model collapses.** `05_broken` scores
*above* `04_degraded` on SSIM in three of the four cell/metric combinations (0.395
vs 0.377 single-view; 0.647 vs 0.623 multi-view dark) while being visibly
destroyed -- a washed-out low-detail frame correlates better with the reference
than a frame full of wrong high-frequency detail. Only LPIPS on the bright cell
orders the two correctly (0.397 -> 0.432). Past the knee, judge visually; do not
trust the metric ordering. This is why the ladder ships the videos and not just a
table.

## Timing protocol

Single-view: per-episode `LVG Total inference time`. Multi-view: per-segment
`Total inference time`, segment 0 excluded because it pays one-off warm-up.
Single GPU, idle, one run at a time; wall clock is never used (it includes model
load, ffmpeg and metric computation).

## Backbone / settings

Cosmos-Predict2 2B DiT post-trained on humanoid manipulation (GR1 / AIRBOT MMK2),
480p, 35 denoising steps, guidance 7, 5 conditioning frames, autoregressive
rollout. Multi-view renders four synchronised cameras as one 2x2 grid, so a single
multi-view frame already contains four views.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--push", action="store_true", help="actually upload")
    ap.add_argument("--private", action="store_true", help="create private instead of public")
    args = ap.parse_args()

    if not SRC.is_dir():
        sys.exit(f"source folder missing: {SRC}")
    files = sorted(p for p in SRC.rglob("*") if p.is_file())
    size_mb = sum(p.stat().st_size for p in files) / 1e6

    print(f"source : {SRC}")
    print(f"repo   : {args.repo}  ({'private' if args.private else 'PUBLIC'})")
    print(f"payload: {len(files)} files, {size_mb:.1f} MB")
    print(f"  mp4={sum(p.suffix == '.mp4' for p in files)} "
          f"png={sum(p.suffix == '.png' for p in files)} "
          f"json={sum(p.suffix == '.json' for p in files)}")

    from huggingface_hub import HfApi, whoami

    token = os.environ.get("HF_TOKEN") or None
    try:
        me = whoami(token=token)
        print(f"auth   : {me.get('name')} ({me.get('type')})")
    except Exception as e:
        sys.exit(f"\nNOT AUTHENTICATED ({type(e).__name__}).\n"
                 "Run `hf auth login` (needs a WRITE token) or export HF_TOKEN=hf_xxx, "
                 "then re-run.")

    if not args.push:
        print("\nDRY RUN -- nothing uploaded. Re-run with --push to publish.")
        return

    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    # Card first, so the repo is never publicly visible without context.
    card = SRC / "README.md"
    prev = card.read_text() if card.is_file() else None
    card.write_text(CARD)
    try:
        api.upload_folder(folder_path=str(SRC), repo_id=args.repo, repo_type="dataset",
                          commit_message="Add WorldCache qualitative quality ladder")
    finally:
        # Leave the working tree as it was: the local README documents the folder
        # for repo users, the card is for the Hub.
        if prev is not None:
            card.write_text(prev)
        else:
            card.unlink(missing_ok=True)

    vis = "private" if args.private else "public"
    print(f"\ndone -> https://huggingface.co/datasets/{args.repo}  ({vis})")


if __name__ == "__main__":
    main()
