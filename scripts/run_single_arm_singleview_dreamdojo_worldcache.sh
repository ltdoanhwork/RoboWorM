#!/usr/bin/env bash
# DreamDojo autoregressive inference with WorldCache acceleration on single-arm singleview.
# Full rollout: video output length matches the GT input episode (chunk-strided autoregressive).
# Runs datasets/single_arm/singleview/makovian and .../non_makovian separately.
# Output: sampling_dataset/worldcache/single_arm/output/singleview/dreamdojo/{split}/iter_*/episode_*/
#
# Usage:
#   bash scripts/run_single_arm_singleview_dreamdojo_worldcache.sh
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_single_arm_singleview_dreamdojo_worldcache.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/checkpoints/finetuned/dreamdojo/dreamdojo/exp_single_arm_sv/single_arm_singleview/checkpoints}"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_single_arm_sv}"
NUM_FRAMES="${NUM_FRAMES:-97}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/worldcache/single_arm/output/singleview/dreamdojo}"
NUM_EPISODES="${NUM_EPISODES:-}"
GUIDANCE="${GUIDANCE:-1.5}"

# WorldCache parameters
# Defaults tuned for STRONGER skipping (faster, slightly lower fidelity).
# Raise rel-l1-thresh and lower ret-ratio / motion-sensitivity to skip more steps.
WORLDCACHE_NUM_STEPS="${WORLDCACHE_NUM_STEPS:-35}"
WORLDCACHE_REL_L1_THRESH="${WORLDCACHE_REL_L1_THRESH:-0.15}"
WORLDCACHE_RET_RATIO="${WORLDCACHE_RET_RATIO:-0.2}"
WORLDCACHE_PROBE_DEPTH="${WORLDCACHE_PROBE_DEPTH:-4}"
WORLDCACHE_MOTION_SENSITIVITY="${WORLDCACHE_MOTION_SENSITIVITY:-2.0}"

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
  echo "=== DreamDojo WorldCache single-arm SV: ${split} -> ${save_base} ==="
  python "${ROOT}/scripts/infer_single_arm_singleview_autoregressive.py" \
    --checkpoints-dir "${CHECKPOINTS_DIR}" \
    --experiment "${EXPERIMENT}" \
    --dataset-path "${dataset_dir}" \
    --save-dir "${save_base}" \
    --num-frames "${NUM_FRAMES}" \
    --save-fps 15 \
    --output-dir "${ROOT}/output/dreamdojo_worldcache_single_arm_sv_${split}" \
    --guidance "${GUIDANCE}" \
    --worldcache-enabled \
    --worldcache-num-steps "${WORLDCACHE_NUM_STEPS}" \
    --worldcache-rel-l1-thresh "${WORLDCACHE_REL_L1_THRESH}" \
    --worldcache-ret-ratio "${WORLDCACHE_RET_RATIO}" \
    --worldcache-probe-depth "${WORLDCACHE_PROBE_DEPTH}" \
    --worldcache-motion-sensitivity "${WORLDCACHE_MOTION_SENSITIVITY}" \
    "${ep_flag[@]}"
done
