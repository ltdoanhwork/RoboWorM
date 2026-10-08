#!/bin/bash
# Ctrl-World humanoid multiview 4-VIEW GRID AUGMENT run (object RECOLORED on the
# first frame of all 4 views).
#
# Uses its own config key (ctrlworld_humanoid_multiview_4view_grid); the 3-view
# sibling launcher (run_humanoid_multiview_ctrlworld_augment.sh) and checkpoint
# were removed on 2026-10-08. Same 3 steps:
#   Step 1: reuse the DENSE 4-view-grid Ctrl-World annotation tree
#           (dataset_example/humanoid_multiview_4view_grid) — action-conditioned,
#           so states/joints/actions stay byte-identical, only pixels change.
#   Step 2: build an AUGMENT annotation tree with the object RECOLORED in frame 0
#           of every one of the 4 views (scripts/augment/edit_ctrlworld_views.py).
#   Step 3: run the augment infer copy with --config humanoid_grid --no-gt.
#
# 10 videos per category, from augment_manifest.json — the SAME entries as
# ctrlworld_humanoid_multiview (gen_manifest.py mirrors that section verbatim,
# since the identical episode ids exist in both dataset_example trees), so the
# 4-view-grid recolors match the legacy 3-view augment rollouts episode-for-episode.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run/ctrlworld/run_humanoid_multiview_ctrlworld_4view_grid_augment.sh [makovian|non_makovian|both]
#
# Env overrides: CKPT, DATASET_NAME, DOWN_SAMPLE, EDIT_MODEL, EDIT_SEED, USE_MASK,
#   EDIT_VENV, MANIFEST, LIMIT (first N manifest entries, for verification),
#   SKIP_EDIT=1 / SKIP_INFER=1, OUTPUT_ROOT / INPUT_ROOT (scratch trials).
set -e

SUBSET=${1:-both}
ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "${ROOT}"

CONFIG_KEY="ctrlworld_humanoid_multiview_4view_grid"
source "${ROOT}/scripts/augment/_ctrlworld_augment_core.sh"
