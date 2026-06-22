#!/usr/bin/env bash
# Test script: single-segment full-length autoregressive rollout.
# Generates ONE continuous rollout from the first GT frame for the entire
# episode length — pure autoregressive, no GT reset between segments.
# Good for testing long-horizon generation quality and physics drift.
#
# Default: runs 1 episode. Set NUM_EPISODES=N for more.
#
# Usage:
#   bash scripts/run_single_arm_singleview_dreamdojo_single_segment_test.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/exp_logs/dreamdojo_logs/dreamdojo/exp_single_arm_sv/single_arm_singleview/checkpoints}"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_single_arm_sv}"
NUM_EPISODES="${NUM_EPISODES:-1}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/tmp/dreamdojo}"

split="${SPLIT:-makovian}"
dataset_dir="${ROOT}/datasets/single_arm/singleview/${split}"

if [ ! -d "${dataset_dir}/data" ]; then
  echo "ERROR: no data directory at ${dataset_dir}/data" >&2
  exit 1
fi

save_base="${SAVE_ROOT}/${split}"
mkdir -p "${save_base}"

echo "=== Single-segment full rollout test: ${split} (${NUM_EPISODES} episodes) -> ${save_base} ==="
python "${ROOT}/scripts/infer_single_arm_singleview_single_segment.py" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --experiment "${EXPERIMENT}" \
  --dataset-path "${dataset_dir}" \
  --save-dir "${save_base}" \
  --num-episodes "${NUM_EPISODES}" \
  --save-fps 15 \
  --output-dir "${ROOT}/output/dreamdojo_single_seg_test_${split}"
