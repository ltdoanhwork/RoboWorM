#!/usr/bin/env bash
# Test script: single-segment full-length autoregressive rollout for bimanual.
# Pure autoregressive from first GT frame — no GT reset between chunks.
# Output: tmp/dreamdojo/bimanual/{split}/iter_*/episode_*/
#
# Usage:
#   bash scripts/run_bimanual_singleview_dreamdojo_single_segment_test.sh
#   NUM_EPISODES=10 SPLIT=non_makovian bash scripts/run_bimanual_singleview_dreamdojo_single_segment_test.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/checkpoints/finetuned/dreamdojo/dreamdojo/exp_aloha_resume70k/aloha_bimanual_singleview_full/checkpoints}"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_aloha_resume70k_bimanual_sv}"
NUM_EPISODES="${NUM_EPISODES:-1}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/tmp/dreamdojo/bimanual}"

split="${SPLIT:-makovian}"
split_dir="${ROOT}/datasets/bimanual/singleview/${split}"

# Collect task directories with parquet data
task_paths=""
for d in $(find "${split_dir}" -maxdepth 1 -mindepth 1 -type d | sort); do
  if find "$d/data" -name "*.parquet" 2>/dev/null | grep -q .; then
    if [ -n "${task_paths}" ]; then
      task_paths="${task_paths},"
    fi
    task_paths="${task_paths}${d}"
  fi
done

if [ -z "${task_paths}" ]; then
  echo "ERROR: no task directories with parquet data found in ${split_dir}" >&2
  exit 1
fi

save_base="${SAVE_ROOT}/${split}"
mkdir -p "${save_base}"

echo "=== Bimanual single-segment full rollout test: ${split} (${NUM_EPISODES} episodes) -> ${save_base} ==="
python "${ROOT}/scripts/infer_bimanual_singleview_single_segment.py" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --experiment "${EXPERIMENT}" \
  --dataset-path "${task_paths}" \
  --save-dir "${save_base}" \
  --num-episodes "${NUM_EPISODES}" \
  --save-fps 10 \
  --output-dir "${ROOT}/output/dreamdojo_single_seg_bimanual_test_${split}"
