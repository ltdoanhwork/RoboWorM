#!/bin/bash
# DreamGen bimanual SINGLEVIEW AUGMENT run — 4 branches x 5 videos per category
# (the extra 4th branch, task_swap, is UNIQUE to this config — see
# .cursor/rules/augment-track-layout.mdc).
#
# Mirrors run_bimanual_singleview_dreamgen_16fps.sh (same ckpt
# bimanual_singleview_16fps, cam_high view, video2world_gr00t_lvg, 16:9,
# per-episode chunk counts taken from the dense batch jsons), and only perturbs
# the input:
#
#   edit_only    the object on the conditioning first frame is RECOLORED,
#                prompt stays byte-identical to dense (refiner off).
#   prompt_only  the frame is untouched, the prompt describes something the
#                recorded actions do not do, refiner on.
#   combo        recolored frame + prompt naming the new color, refiner on.
#   task_swap    the frame is untouched (own episode), but the prompt is a
#                DIFFERENT REAL task's caption borrowed from elsewhere in the
#                dataset (same object pool, different scene) — tests whether
#                the model follows the text or just continues the frame. Refiner on.
#
# Output (leaf gains a __{branch} tag; drop it to get the dense path):
#   sampling_dataset/augment/bimanual/output/singleview/dreamgen/{split}/
#       {iter}_bimanual16fps/{task}__episode_XXXXXX__{branch}.mp4
#
# Augmented inputs have no ground truth, so no GT video and no PSNR/SSIM/LPIPS are
# written — reference-free metrics come from scripts/augment/compute_augment_metrics.py.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_bimanual_singleview_dreamgen_augment.sh
#   SPLITS=makovian BRANCHES=task_swap LIMIT=1 bash scripts/run_bimanual_singleview_dreamgen_augment.sh
#
# Env overrides: SPLITS, BRANCHES, LIMIT, MANIFEST, ITER_NAME, DIT_PATH, NUM_GPUS,
#   GUIDANCE, MODEL_SIZE, ASPECT_RATIO, EDIT_MODEL, EDIT_SEED, SKIP_EDIT=1,
#   SKIP_INFER=1, FORCE_REGENERATE=1.
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CONFIG_KEY="dreamgen_bimanual_singleview"
# This is the ONLY config with a 4th branch (task_swap) — default it here so
# every other config sourcing _dreamgen_augment_core.sh keeps its own 3-branch
# default untouched.
BRANCHES="${BRANCHES:-edit_only prompt_only combo task_swap object_relocation}"
source "${ROOT}/scripts/augment/_dreamgen_augment_core.sh"
