---
name: sampling-dataset-structure
description: Describes sampling_dataset layout in video_gen_physics—especially dense vs future accelerated video runs—and default paths for multiview scripts. Use when editing paths under sampling_dataset, DreamGen batch I/O, or when the user mentions dense, accelerated video, or single_arm sampling output.
disable-model-invocation: true
---

# sampling_dataset structure

Repo root conventions use `sampling_dataset/` as the workspace for **prepared inputs** (grids, `batch_input.json`) and **generated outputs** (mp4, sidecar prompts), separate from raw LeRobot data under `datasets/`.

## Why `dense/`

The folder name **`dense`** marks the **standard / full** generation path: baseline model, full diffusion steps, or the default quality setting you compare against.

Reserve **sibling directories under the same role** (e.g. `sampling_dataset/<variant>/single_arm/…`) for other runs—**accelerated** or sparse inference (FasterCache, SiTo, PISA, fewer steps, etc.)—so outputs stay organized and comparable without overwriting `dense/`.

## Single-arm multiview (DreamGen GR00T droid)

Defaults (overridable with `INPUT_DIR` / `OUTPUT_DIR`):

- Input (grids + batch JSON): `sampling_dataset/dense/single_arm/input/multiview/dreamgen/<makovian|non_makovian>/`
- Output (generated videos): `sampling_dataset/dense/single_arm/output/multiview/dreamgen/<makovian|non_makovian>/`

Scripts: `prepare_single_arm_multiview_dreamgen_batch.py`, `run_single_arm_multiview_dreamgen_batch.sh`, `run_single_arm_multiview_dreamgen_all.sh`.

## Other roots (do not move without user intent)

These follow long-standing paths in this repo; only change if the user asks:

- Humanoid singleview — DreamGen-style batches / legacy: `sampling_dataset/humanoid/singleview/input/…` (often `dense/` under a model bucket)
- Humanoid singleview — DreamDojo dense action-conditioned saves (default shell output): `sampling_dataset/dense/humanoid/output/singleview/dreamdojo/{makovian|non_makovian}/`
- Bimanual: `sampling_dataset/bimanual/input|output/multiview/dreamgen/…`

When adding a new pipeline, mirror the pattern: **`input/…` for artifacts fed to inference**, **`output/…` for mp4 and logs**, and use **`dense/`** (or a clearly named variant folder) at the appropriate level so accelerated runs can sit beside it later.
