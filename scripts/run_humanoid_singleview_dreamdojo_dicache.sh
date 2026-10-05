#!/usr/bin/env bash
# DreamDojo full-episode inference with DiCache acceleration.
# Runs datasets/humanoid/singleview/makovian and .../non_makovian separately.
# Output: sampling_dataset/dicache/humanoid/output/singleview/dreamdojo/{split}/iter_*/episode_*/
#
# Usage:
#   bash scripts/run_humanoid_singleview_dreamdojo_dicache.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_humanoid_singleview_dreamdojo_dicache.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/checkpoints/base/dreamdojo/2B_GR1_post-train/iter_000050000}"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_gr1}"
NUM_FRAMES="${NUM_FRAMES:-49}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/dicache/humanoid/output/singleview/dreamdojo}"
NUM_EPISODES="${NUM_EPISODES:-}"
GUIDANCE="${GUIDANCE:-1.5}"

# DiCache parameters
DICACHE_NUM_STEPS="${DICACHE_NUM_STEPS:-35}"
DICACHE_REL_L1_THRESH="${DICACHE_REL_L1_THRESH:-0.08}"
DICACHE_RET_RATIO="${DICACHE_RET_RATIO:-0.2}"
DICACHE_PROBE_DEPTH="${DICACHE_PROBE_DEPTH:-2}"

ep_flag=()
if [[ -n "${NUM_EPISODES}" ]]; then
  ep_flag=(--num-episodes "${NUM_EPISODES}")
fi

for split in makovian non_makovian; do
  dataset_dir="${ROOT}/datasets/humanoid/singleview/${split}"
  if [ ! -d "${dataset_dir}/data" ]; then
    echo "Skipping ${split}: no data directory found at ${dataset_dir}/data" >&2
    continue
  fi
  save_base="${SAVE_ROOT}/${split}"
  mkdir -p "${save_base}"
  echo "=== DreamDojo DiCache full-episode: ${split} -> ${save_base} ==="
  # SAVE_FULL=1 adds --save-full (PSNR/SSIM/LPIPS). The sparse-attention
  # scripts for this cell already pass it unconditionally, so without this the
  # cache methods would be the only ones with no quality numbers to pair with
  # their speedup.
  full_flag=()
  if [ "${SAVE_FULL:-0}" = "1" ]; then full_flag=(--save-full); fi
  python "${ROOT}/scripts/infer_humanoid_singleview_full_episode.py" \
    --checkpoints-dir "${CHECKPOINTS_DIR}" \
    --experiment "${EXPERIMENT}" \
    --dataset-path "${dataset_dir}" \
    --save-dir "${save_base}" \
    --num-frames "${NUM_FRAMES}" \
    --output-dir "${ROOT}/output/dreamdojo_dicache_humanoid_sv_${split}" \
    --guidance "${GUIDANCE}" \
    --dicache-enabled \
    --dicache-num-steps "${DICACHE_NUM_STEPS}" \
    --dicache-rel-l1-thresh "${DICACHE_REL_L1_THRESH}" \
    --dicache-ret-ratio "${DICACHE_RET_RATIO}" \
    --dicache-probe-depth "${DICACHE_PROBE_DEPTH}" \
    "${full_flag[@]}" \
    "${ep_flag[@]}"
done
