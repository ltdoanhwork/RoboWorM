#!/usr/bin/env bash
# DreamDojo dense action-conditioned inference on LeRobot single-arm singleview splits.
# Runs datasets/single_arm/singleview/makovian and .../non_makovian with data_split full.
#
# Pure autoregressive rollout: only the FIRST GT frame is used as conditioning.
# All subsequent frames are generated from the model's own predictions (no GT re-anchoring).
# Produces ONE output video per episode matching the full GT episode length.
#
# Outputs under SAVE_ROOT/{makovian,non_makovian}/<iter_*>/episode_XXXXXX/
#
# Usage:
#   bash scripts/run_single_arm_singleview_dreamdojo_dense.sh
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_single_arm_singleview_dreamdojo_dense.sh
#   NUM_EPISODES=10 bash scripts/run_single_arm_singleview_dreamdojo_dense.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/exp_logs/dreamdojo_logs/dreamdojo/exp_droid_sv/droid_singlearm_singleview/checkpoints}"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_droid_sv}"
NUM_FRAMES="${NUM_FRAMES:-97}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/dense/single_arm/output/singleview/dreamdojo}"
NUM_EPISODES="${NUM_EPISODES:-}"

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
  out_cache="${ROOT}/output/dreamdojo_dense_single_arm_sv_${split}"
  mkdir -p "${save_base}" "${out_cache}"
  echo "=== DreamDojo dense single-arm SV (autoregressive): ${split} -> ${save_base} ==="
  python "${ROOT}/scripts/infer_single_arm_singleview_autoregressive.py" \
    --checkpoints-dir "${CHECKPOINTS_DIR}" \
    --experiment "${EXPERIMENT}" \
    --dataset-path "${dataset_dir}" \
    --save-dir "${save_base}" \
    --num-frames "${NUM_FRAMES}" \
    --save-fps 15 \
    --output-dir "${out_cache}" \
    "${ep_flag[@]}"
done
