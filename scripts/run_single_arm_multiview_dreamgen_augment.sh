#!/bin/bash
# DreamGen single_arm MULTIVIEW AUGMENT run — 3 branches x 5 videos per category,
# plus a hand-curated object_relocation branch (variable count, see gen_manifest.py).
#
# Mirrors run_single_arm_multiview_dreamgen_batch.sh (same base GR00T DROID 14B
# checkpoint, single-shot `examples.video2world_gr00t`, FPS 10, 2x2 grid first
# frame), and only perturbs the input:
#
#   edit_only    the object is RECOLORED in the 2x2 grid conditioning frame,
#                prompt stays byte-identical to dense.
#   prompt_only  grid frame untouched, the perturbed behaviour is injected into
#                the grid caption's task slot (the 2x2-layout sentences MUST stay,
#                or the model stops emitting a grid), refiner on.
#   combo        recolored grid + perturbed task phrase naming the new color.
#   object_relocation  grid frame untouched, hand-curated task phrase naming a
#                REAL loose object AND a REAL container both visible in the
#                grid's frame 0 (injected into the same task slot as prompt_only).
#
# Output (leaf gains a __{branch} tag; drop it to get the dense path):
#   sampling_dataset/augment/single_arm/output/multiview/dreamgen/{split}/
#       episode_XXXXXX__{branch}.mp4
#
# Augmented inputs have no ground truth, so no GT video and no PSNR/SSIM/LPIPS are
# written — reference-free metrics come from scripts/augment/compute_augment_metrics.py.
#
# Note: 14B single-GPU is heavy (~490s/episode dense); budget accordingly.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_single_arm_multiview_dreamgen_augment.sh
#   SPLITS=makovian BRANCHES=edit_only LIMIT=1 bash scripts/run_single_arm_multiview_dreamgen_augment.sh
#
# Env overrides: SPLITS, BRANCHES, LIMIT, MANIFEST, MODEL_SIZE, FPS, NUM_GPUS,
#   EDIT_MODEL, EDIT_SEED, SKIP_EDIT=1, SKIP_INFER=1, FORCE_REGENERATE=1.
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CONFIG_KEY="dreamgen_single_arm_multiview"
BRANCHES="${BRANCHES:-edit_only prompt_only combo object_relocation}"
source "${ROOT}/scripts/augment/_dreamgen_augment_core.sh"
