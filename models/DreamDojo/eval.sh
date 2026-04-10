#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

OUTPUT_SUBDIR="dreamdojo_hv_eval_14b_g1_gr1"
OUTPUT_DIR="${REPO_ROOT}/outputs/action_conditioned/${OUTPUT_SUBDIR}"
SAVE_DIR="${WORKSPACE_ROOT}/dream_gen_benchmark/${OUTPUT_SUBDIR}"
CHECKPOINTS_DIR="${WORKSPACE_ROOT}/checkpoints/dreamdojo/14B_G1_post-train"
DATASET_PATH="${WORKSPACE_ROOT}/checkpoints/dreamdojo/datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/GR1_robot"

mkdir -p "${OUTPUT_DIR}" "${SAVE_DIR}"

if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_ROOT}"

python scripts/run_action_conditioned_eval_local.py \
  --output-dir "${OUTPUT_DIR}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --experiment "dreamdojo_14b_480_640_gr1" \
  --save-dir "${SAVE_DIR}" \
  --dataset-path "${DATASET_PATH}" \
  --num-frames 49 \
  --num-samples 100 \
  --data-split "test" \
  --deterministic-uniform-sampling
