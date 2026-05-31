#!/usr/bin/env bash
# DreamDojo full-episode inference with FasterCache acceleration on single-arm (DROID) singleview.
# Runs datasets/single_arm/singleview/makovian and .../non_makovian separately.
# Output: sampling_dataset/fastercache/single_arm/output/singleview/dreamdojo/{split}/iter_*/episode_*/
#
# Usage:
#   bash scripts/run_single_arm_singleview_dreamdojo_fastercache.sh
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_single_arm_singleview_dreamdojo_fastercache.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/exp_logs/dreamdojo_logs/dreamdojo/exp_droid_sv/droid_singlearm_singleview/checkpoints/iter_000100000}"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_droid_sv}"
NUM_FRAMES="${NUM_FRAMES:-97}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/fastercache/single_arm/output/singleview/dreamdojo}"
NUM_EPISODES="${NUM_EPISODES:-}"
GUIDANCE="${GUIDANCE:-1.5}"

# FasterCache parameters
FASTERCACHE_START_STEP="${FASTERCACHE_START_STEP:-0}"
FASTERCACHE_MODEL_INTERVAL="${FASTERCACHE_MODEL_INTERVAL:-5}"
FASTERCACHE_BLOCK_INTERVAL="${FASTERCACHE_BLOCK_INTERVAL:-3}"

ep_flag=()
if [[ -n "${NUM_EPISODES}" ]]; then
  ep_flag=(--num-episodes "${NUM_EPISODES}")
fi

for split in makovian non_makovian; do
  dataset_dir="${ROOT}/datasets/single_arm/singleview/${split}"
  if [ ! -d "${dataset_dir}/data" ]; then
    echo "Skipping ${split}: no data directory found at ${dataset_dir}/data" >&2
    continue
  fi
  save_base="${SAVE_ROOT}/${split}"
  mkdir -p "${save_base}"
  echo "=== DreamDojo FasterCache single-arm SV: ${split} -> ${save_base} ==="
  python "${ROOT}/scripts/infer_humanoid_singleview_full_episode.py" \
    --checkpoints-dir "${CHECKPOINTS_DIR}" \
    --experiment "${EXPERIMENT}" \
    --dataset-path "${dataset_dir}" \
    --save-dir "${save_base}" \
    --num-frames "${NUM_FRAMES}" \
    --save-fps 15 \
    --output-dir "${ROOT}/output/dreamdojo_fastercache_single_arm_sv_${split}" \
    --guidance "${GUIDANCE}" \
    --fastercache-enabled \
    --fastercache-start-step "${FASTERCACHE_START_STEP}" \
    --fastercache-model-interval "${FASTERCACHE_MODEL_INTERVAL}" \
    --fastercache-block-interval "${FASTERCACHE_BLOCK_INTERVAL}" \
    "${ep_flag[@]}"
done
