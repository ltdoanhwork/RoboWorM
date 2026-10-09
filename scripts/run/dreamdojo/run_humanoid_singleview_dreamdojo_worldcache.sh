#!/usr/bin/env bash
# DreamDojo full-episode inference with WorldCache acceleration.
# Runs datasets/humanoid/singleview/makovian and .../non_makovian separately.
# Output: sampling_dataset/worldcache/humanoid/output/singleview/dreamdojo/{split}/iter_*/episode_*/
#
# Usage:
#   bash scripts/run/dreamdojo/run_humanoid_singleview_dreamdojo_worldcache.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamdojo/run_humanoid_singleview_dreamdojo_worldcache.sh
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/checkpoints/base/dreamdojo/2B_GR1_post-train/iter_000050000}"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_gr1}"
NUM_FRAMES="${NUM_FRAMES:-49}"
_SAVE_ROOT_ENV="${SAVE_ROOT:-}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/worldcache/humanoid/output/singleview/dreamdojo}"
# Intermediate --output-dir: OUTPUT_ROOT if set, else <SAVE_ROOT>/_output when SAVE_ROOT
# was given (keeps scratch runs self-contained), else the historical ${ROOT}/output.
OUT_CACHE_ROOT="${OUTPUT_ROOT:-${_SAVE_ROOT_ENV:+${_SAVE_ROOT_ENV}/_output}}"
OUT_CACHE_ROOT="${OUT_CACHE_ROOT:-${ROOT}/output}"
NUM_EPISODES="${NUM_EPISODES:-}"
# Guidance: same as the cell's dense launcher (run_humanoid_singleview_dreamdojo_full_episode.sh: GUIDANCE, default 0 =
# the infer-script default). Was 1.5 before 2026-10-09, so fidelity-vs-dense compared
# different guidance scales.
GUIDANCE="${GUIDANCE:-0}"

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
  dataset_dir="${ROOT}/datasets/humanoid/singleview/${split}"
  if [ ! -d "${dataset_dir}/data" ]; then
    echo "Skipping ${split}: no data directory found at ${dataset_dir}/data" >&2
    continue
  fi
  save_base="${SAVE_ROOT}/${split}"
  mkdir -p "${save_base}"
  echo "=== DreamDojo WorldCache full-episode: ${split} -> ${save_base} ==="
  # SAVE_FULL=1 adds --save-full (PSNR/SSIM/LPIPS). The sparse-attention
  # scripts for this cell already pass it unconditionally, so without this the
  # cache methods would be the only ones with no quality numbers to pair with
  # their speedup.
  full_flag=()
  if [ "${SAVE_FULL:-0}" = "1" ]; then full_flag=(--save-full); fi
  python "${ROOT}/scripts/infer/infer_humanoid_singleview_full_episode.py" \
    --checkpoints-dir "${CHECKPOINTS_DIR}" \
    --experiment "${EXPERIMENT}" \
    --dataset-path "${dataset_dir}" \
    --save-dir "${save_base}" \
    --num-frames "${NUM_FRAMES}" \
    --output-dir "${OUT_CACHE_ROOT}/dreamdojo_worldcache_humanoid_sv_${split}" \
    --guidance "${GUIDANCE}" \
    --worldcache-enabled \
    --worldcache-num-steps "${WORLDCACHE_NUM_STEPS}" \
    --worldcache-rel-l1-thresh "${WORLDCACHE_REL_L1_THRESH}" \
    --worldcache-ret-ratio "${WORLDCACHE_RET_RATIO}" \
    --worldcache-probe-depth "${WORLDCACHE_PROBE_DEPTH}" \
    --worldcache-motion-sensitivity "${WORLDCACHE_MOTION_SENSITIVITY}" \
    "${full_flag[@]}" \
    "${ep_flag[@]}"
done
