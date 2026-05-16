#!/usr/bin/env bash
# Full-episode inference for bimanual singleview DreamDojo.
# Each episode is split into non-overlapping 97-frame segments.
# Each segment starts from the GT frame at that position.
# All segments are concatenated into one video matching episode length.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_bimanual_singleview_dreamdojo_full_episode.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/exp_logs/dreamdojo_logs/dreamdojo/exp_aloha_resume40k/aloha_bimanual_singleview_full/checkpoints}"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_aloha}"
NUM_FRAMES="${NUM_FRAMES:-97}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/dense/bimanual/output/single_view/dreamdojo}"
NUM_EPISODES="${NUM_EPISODES:-}"

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

for split in makovian non_makovian; do
  task_paths="$(get_task_paths "${split}")"
  if [ -z "${task_paths}" ]; then
    echo "Skipping ${split}: no task directories found." >&2
    continue
  fi
  save_base="${SAVE_ROOT}/${split}"
  mkdir -p "${save_base}"
  echo "=== Full-episode inference: ${split} -> ${save_base} ==="
  python "${ROOT}/scripts/infer_bimanual_singleview_full_episode.py" \
    --checkpoints-dir "${CHECKPOINTS_DIR}" \
    --experiment "${EXPERIMENT}" \
    --dataset-path "${task_paths}" \
    --save-dir "${save_base}" \
    --num-frames "${NUM_FRAMES}" \
    --output-dir "${ROOT}/output/dreamdojo_full_ep_bimanual_sv_${split}" \
    "${ep_flag[@]}"
done
