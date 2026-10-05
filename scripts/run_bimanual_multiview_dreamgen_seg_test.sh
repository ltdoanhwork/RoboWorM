#!/usr/bin/env bash
# DreamGen dense multiview inference — bimanual (4-view 2x2 grid),
# using the SEGMENTED post-training checkpoint (bimanual_multiview_seg).
#
# This is a thin wrapper over scripts/run_bimanual_multiview_dreamgen_dense.sh:
# it just points CHECKPOINTS_DIR at the segmented run. The segmented checkpoint
# is distinguished by its iter dir (iter_000010000) in the canonical path, NOT
# by a model-name suffix (per sampling-dataset-layout rule: {model} must stay
# "dreamgen"). Outputs therefore share the standard dense/.../dreamgen tree and
# stay comparable with the all_multiview baseline at the same iter granularity.
#
# Grid layout (--four_views, matches training):
#   TL: cam_high  |  TR: cam_low
#   BL: cam_left_wrist  |  BR: cam_right_wrist
#
# Usage (from repo root) — NUM_GPUS must equal the GPU count you expose:
#   1 GPU:  MAX_EPISODES=5 NUM_GPUS=1 CUDA_VISIBLE_DEVICES=0 bash scripts/run_bimanual_multiview_dreamgen_seg_test.sh
#   2 GPUs: MAX_EPISODES=2 NUM_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_bimanual_multiview_dreamgen_seg_test.sh
#   Prep only (no GPU): PREPARE_ONLY=1 bash scripts/run_bimanual_multiview_dreamgen_seg_test.sh
#
# Overrideable env: NUM_GPUS, GUIDANCE, MAX_EPISODES, MASTER_PORT, DIT_PATH, etc.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Segmented checkpoint dir (latest_checkpoint.txt -> iter_000010000.pt)
# BROKEN (2026-10-05 refactor): dreamgen bimanual_multiview_seg (iter_000010000) does not exist on disk (not in checkpoints/
# nor the HF backup). Set CHECKPOINTS_DIR explicitly to run this script.
export CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/checkpoints/finetuned/dreamgen/posttraining/multiview/bimanual_multiview_seg/checkpoints_40_per}"

# Canonical I/O paths (per sampling-dataset-layout rule). {model} stays
# "dreamgen"; the segmented 10k checkpoint is distinguished by its iter dir,
# which run_..._dense.sh inserts automatically (iter_000010000).
export INPUT_ROOT="${INPUT_ROOT:-${ROOT}/sampling_dataset/dense/bimanual/input/multiview/dreamgen}"
export SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/dense/bimanual/output/multiview/dreamgen}"

# Sensible defaults for a quick test: 2 GPUs, a couple episodes per task.
# NOTE: NUM_GPUS must match the number of GPUs in CUDA_VISIBLE_DEVICES.
export NUM_GPUS="${NUM_GPUS:-2}"
export MASTER_PORT="${MASTER_PORT:-29503}"
export MAX_EPISODES="${MAX_EPISODES:-2}"

echo "=== Bimanual Multiview SEG test ==="
echo "  Checkpoints dir: ${CHECKPOINTS_DIR}"
echo "  Input root:      ${INPUT_ROOT}"
echo "  Save root:       ${SAVE_ROOT}"
echo "  NUM_GPUS=${NUM_GPUS}  MAX_EPISODES=${MAX_EPISODES}  MASTER_PORT=${MASTER_PORT}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "==================================="

exec bash "${ROOT}/scripts/run_bimanual_multiview_dreamgen_dense.sh"
