#!/usr/bin/env bash
# Full-episode single-segment autoregressive rollout for bimanual singleview DreamDojo.
# Pure autoregressive from first GT frame — no GT reset between chunks.
# Generates video as long as the input episode.
#
# Output: sampling_dataset/dense/bimanual/output/single_view/dreamdojo/{split}/iter_*/episode_*/
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_bimanual_singleview_dreamdojo_full_episode.sh
#   NUM_EPISODES=10 bash scripts/run_bimanual_singleview_dreamdojo_full_episode.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/exp_logs/dreamdojo_logs/dreamdojo/exp_aloha_resume40k/aloha_bimanual_singleview_full/checkpoints}"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_aloha}"
NUM_EPISODES="${NUM_EPISODES:-}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/dense/bimanual/output/single_view/dreamdojo}"

get_task_paths() {
  local split="$1"
  local split_dir="${ROOT}/datasets/bimanual/singleview/${split}"
  local paths=""
  for d in $(find "${split_dir}" -maxdepth 1 -mindepth 1 -type d | sort); do
    if find "$d/data" -name "*.parquet" 2>/dev/null | grep -q .; then
      if [ -n "${paths}" ]; then
        paths="${paths},"
      fi
      paths="${paths}${d}"
    fi
  done
  echo "${paths}"
}

ep_flag=()
if [[ -n "${NUM_EPISODES}" ]]; then
  ep_flag=(--num-episodes "${NUM_EPISODES}")
fi

pids=()
gpu=0

for split in makovian non_makovian; do
  task_paths="$(get_task_paths "${split}")"
  if [ -z "${task_paths}" ]; then
    echo "Skipping ${split}: no task directories found." >&2
    continue
  fi
  save_base="${SAVE_ROOT}/${split}"
  mkdir -p "${save_base}"
  echo "=== Full-episode single-segment rollout: ${split} (GPU ${gpu}) -> ${save_base} ==="
  CUDA_VISIBLE_DEVICES="${gpu}" python "${ROOT}/scripts/infer_bimanual_singleview_single_segment.py" \
    --checkpoints-dir "${CHECKPOINTS_DIR}" \
    --experiment "${EXPERIMENT}" \
    --dataset-path "${task_paths}" \
    --save-dir "${save_base}" \
    --save-fps 10 \
    --output-dir "${ROOT}/output/dreamdojo_full_ep_bimanual_sv_${split}" \
    "${ep_flag[@]}" &
  pids+=($!)
  gpu=$((gpu + 1))
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done
