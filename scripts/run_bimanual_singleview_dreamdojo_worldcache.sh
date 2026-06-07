#!/usr/bin/env bash
# DreamDojo full-episode inference with WorldCache acceleration on bimanual (ALOHA) singleview.
# Runs datasets/bimanual/singleview/makovian and .../non_makovian separately.
# Output: sampling_dataset/worldcache/bimanual/output/singleview/dreamdojo/{split}/iter_*/episode_*/
#
# Usage:
#   bash scripts/run_bimanual_singleview_dreamdojo_worldcache.sh
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_bimanual_singleview_dreamdojo_worldcache.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/exp_logs/dreamdojo_logs/dreamdojo/exp_aloha_resume70k/aloha_bimanual_singleview_full/checkpoints/iter_000100000}"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_aloha_resume70k_bimanual_sv}"
NUM_FRAMES="${NUM_FRAMES:-49}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/worldcache/bimanual/output/singleview/dreamdojo}"
NUM_EPISODES="${NUM_EPISODES:-}"
GUIDANCE="${GUIDANCE:-1.5}"

# WorldCache parameters
WORLDCACHE_NUM_STEPS="${WORLDCACHE_NUM_STEPS:-35}"
WORLDCACHE_REL_L1_THRESH="${WORLDCACHE_REL_L1_THRESH:-0.03}"
WORLDCACHE_RET_RATIO="${WORLDCACHE_RET_RATIO:-0.4}"
WORLDCACHE_PROBE_DEPTH="${WORLDCACHE_PROBE_DEPTH:-4}"
WORLDCACHE_MOTION_SENSITIVITY="${WORLDCACHE_MOTION_SENSITIVITY:-5.0}"

ep_flag=()
if [[ -n "${NUM_EPISODES}" ]]; then
  ep_flag=(--num-episodes "${NUM_EPISODES}")
fi

for split in makovian non_makovian; do
  split_dir="${ROOT}/datasets/bimanual/singleview/${split}"
  if [ ! -d "${split_dir}" ]; then
    echo "Skipping ${split}: directory not found at ${split_dir}" >&2
    continue
  fi

  # Build comma-separated list of all task directories that have data/
  dataset_paths=""
  for task_dir in "${split_dir}"/*/; do
    if [ -d "${task_dir}data" ]; then
      if [ -n "${dataset_paths}" ]; then
        dataset_paths="${dataset_paths},${task_dir%/}"
      else
        dataset_paths="${task_dir%/}"
      fi
    fi
  done

  if [ -z "${dataset_paths}" ]; then
    echo "Skipping ${split}: no task directories with data/ found" >&2
    continue
  fi

  save_base="${SAVE_ROOT}/${split}"
  mkdir -p "${save_base}"
  echo "=== DreamDojo WorldCache bimanual SV: ${split} -> ${save_base} ==="
  python "${ROOT}/scripts/infer_humanoid_singleview_full_episode.py" \
    --checkpoints-dir "${CHECKPOINTS_DIR}" \
    --experiment "${EXPERIMENT}" \
    --dataset-path "${dataset_paths}" \
    --save-dir "${save_base}" \
    --num-frames "${NUM_FRAMES}" \
    --save-fps 15 \
    --output-dir "${ROOT}/output/dreamdojo_worldcache_bimanual_sv_${split}" \
    --guidance "${GUIDANCE}" \
    --worldcache-enabled \
    --worldcache-num-steps "${WORLDCACHE_NUM_STEPS}" \
    --worldcache-rel-l1-thresh "${WORLDCACHE_REL_L1_THRESH}" \
    --worldcache-ret-ratio "${WORLDCACHE_RET_RATIO}" \
    --worldcache-probe-depth "${WORLDCACHE_PROBE_DEPTH}" \
    --worldcache-motion-sensitivity "${WORLDCACHE_MOTION_SENSITIVITY}" \
    "${ep_flag[@]}"
done
