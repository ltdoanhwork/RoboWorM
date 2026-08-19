#!/bin/bash
# DreamGen single_arm SINGLEVIEW AUGMENT run — 3 branches x 5 videos per category,
# plus a hand-curated object_relocation branch (variable count, see gen_manifest.py).
#
# Mirrors run_single_arm_singleview_dreamgen_static16fps.sh (same ckpt
# single_arm_static_singleview_16fps_v2, global static view, video2world_gr00t_lvg,
# 4:3, 1 conditioning frame, per-episode chunk counts taken from the dense batch
# jsons), and only perturbs the input:
#
#   edit_only    the object on the conditioning first frame is RECOLORED,
#                prompt stays byte-identical to dense (refiner off).
#   prompt_only  the frame is untouched, the prompt describes something the
#                recorded actions do not do, refiner on.
#   combo        recolored frame + prompt naming the new color, refiner on.
#   object_relocation  frame untouched, hand-curated prompt naming a REAL loose
#                object AND a REAL container both visible in that frame — this
#                config's scenes are repetitive (2 tasks/split), so the count is
#                small/variable (0 for makovian, 1 for non_makovian). Refiner on.
#
# Output (leaf gains a __{branch} tag; drop it to get the dense path):
#   sampling_dataset/augment/single_arm/output/singleview/dreamgen/{split}/
#       {iter}_static16fps/global/episode_XXXXXX__{branch}.mp4
#
# Augmented inputs have no ground truth, so no GT video and no PSNR/SSIM/LPIPS are
# written — reference-free metrics come from scripts/augment/compute_augment_metrics.py.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_single_arm_singleview_dreamgen_augment.sh
#   SPLITS=makovian BRANCHES=edit_only LIMIT=1 bash scripts/run_single_arm_singleview_dreamgen_augment.sh
#
# Env overrides: SPLITS, BRANCHES, LIMIT, MANIFEST, ITER_NAME, DIT_PATH, NUM_GPUS,
#   GUIDANCE, MODEL_SIZE, ASPECT_RATIO, NUM_CONDITIONAL_FRAMES, EDIT_MODEL, EDIT_SEED,
#   SKIP_EDIT=1, SKIP_INFER=1, FORCE_REGENERATE=1.
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CONFIG_KEY="dreamgen_single_arm_singleview"
BRANCHES="${BRANCHES:-edit_only prompt_only combo object_relocation}"
source "${ROOT}/scripts/augment/_dreamgen_augment_core.sh"
