#!/bin/bash
# Ctrl-World single_arm multiview AUGMENT run (object RECOLORED on the first frame).
#
# Mirrors run_single_arm_multiview_ctrlworld.sh (same ckpt / model / infer logic),
# with three deliberate differences:
#   Step 1: reuse the DENSE Ctrl-World annotation tree (dataset_example). It is
#           action-conditioned, so states/joints/actions stay byte-identical —
#           only pixels change.
#   Step 2: build an AUGMENT annotation tree with the object RECOLORED in frame 0
#           of every view (scripts/augment/edit_ctrlworld_views.py, .venv_edit /
#           Qwen-Image-Edit). Frame 0 is Ctrl-World's ONLY visual condition, so
#           editing later frames would cost time without changing conditioning.
#           Recolor (not "swap for another object") keeps the 3 views mutually
#           consistent — see .cursor/rules/augment-track-layout.mdc.
#   Step 3: run the augment infer copy with --no-gt: the recolored scene never
#           happened, so no GT video and no PSNR/SSIM/LPIPS are produced.
#
# 10 videos per category, from augment_manifest.json.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_single_arm_multiview_ctrlworld_augment.sh [makovian|non_makovian|both]
#
# Env overrides: CKPT, EDIT_MODEL, EDIT_SEED, USE_MASK, EDIT_VENV, MANIFEST,
#   LIMIT (first N manifest entries, for verification), SKIP_EDIT=1 / SKIP_INFER=1,
#   OUTPUT_ROOT / INPUT_ROOT (scratch trials).
set -e

SUBSET=${1:-both}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CONFIG_KEY="ctrlworld_single_arm_multiview"
source "${ROOT}/scripts/augment/_ctrlworld_augment_core.sh"
