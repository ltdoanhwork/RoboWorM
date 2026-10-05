#!/usr/bin/env bash
# Test inference for humanoid singleview using the 70k post-training checkpoint.
# Runs on both makovian and non_makovian splits.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_humanoid_singleview_dreamdojo_70k_test.sh
#   NUM_EPISODES=3 CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_humanoid_singleview_dreamdojo_70k_test.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- Environment ---
export VENV="${ROOT}/models/DreamDojo/.venv"
export LD_LIBRARY_PATH="${VENV}/lib/python3.10/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/python3.10/site-packages/nvidia/cusparse/lib:${VENV}/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"

# --- Checkpoint ---
# BROKEN (2026-10-05 refactor): the combined dreamgen all_singleview run does not exist on disk (not in
# checkpoints/finetuned/ nor the HF backup). Set CHECKPOINTS_DIR explicitly to run this script.
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/checkpoints/finetuned/dreamgen/posttraining/singleview/all_singleview/checkpoints_40_per}"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_gr1}"
NUM_FRAMES="${NUM_FRAMES:-49}"
NUM_EPISODES="${NUM_EPISODES:-}"
GUIDANCE="${GUIDANCE:-0}"
NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29502}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/dense/humanoid/output/singleview/dreamdojo_singleview}"

# Optional: limit episodes for quick test
ep_flag=()
if [[ -n "${NUM_EPISODES}" ]]; then
  ep_flag=(--num-episodes "${NUM_EPISODES}")
fi

echo "============================================"
echo "DreamDojo Humanoid Singleview 70k Test"
echo "  Checkpoint: ${CHECKPOINTS_DIR}"
echo "  Experiment: ${EXPERIMENT}"
echo "  Num frames: ${NUM_FRAMES}"
echo "  Guidance:   ${GUIDANCE}"
echo "  GPUs:       ${NUM_GPUS}"
echo "  Save root:  ${SAVE_ROOT}"
echo "  Episodes:   ${NUM_EPISODES:-all}"
echo "============================================"

for split in makovian non_makovian; do
  dataset_dir="${ROOT}/datasets/humanoid/singleview/${split}"
  if [ ! -d "${dataset_dir}/data" ]; then
    echo "Skipping ${split}: no data directory found at ${dataset_dir}/data" >&2
    continue
  fi
  save_base="${SAVE_ROOT}/${split}"
  mkdir -p "${save_base}"
  echo ""
  echo "=== Full-episode inference: ${split} -> ${save_base} ==="
  torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    "${ROOT}/scripts/infer_humanoid_singleview_full_episode.py" \
    --checkpoints-dir "${CHECKPOINTS_DIR}" \
    --experiment "${EXPERIMENT}" \
    --dataset-path "${dataset_dir}" \
    --save-dir "${save_base}" \
    --num-frames "${NUM_FRAMES}" \
    --guidance "${GUIDANCE}" \
    --output-dir "${ROOT}/output/dreamdojo_70k_humanoid_sv_${split}" \
    "${ep_flag[@]}"
  echo "=== Done: ${split} ==="
done

echo ""
echo "All done. Results saved under: ${SAVE_ROOT}"
