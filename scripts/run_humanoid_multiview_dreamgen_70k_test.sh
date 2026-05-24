#!/usr/bin/env bash
# Test inference for humanoid multiview (2x2 grid) using the 70k post-training checkpoint.
# Runs on both makovian and non_makovian splits, all subdatasets.
# Uses 4 GPUs (context parallelism) by default.
#
# Each sample is a 2x2 grid with 4 camera views. Prompt describes all views.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_humanoid_multiview_dreamgen_70k_test.sh
#   MAX_EPISODES=3 CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_humanoid_multiview_dreamgen_70k_test.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- Environment ---
export VENV="${ROOT}/models/DreamDojo/.venv"
export LD_LIBRARY_PATH="${VENV}/lib/python3.10/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/python3.10/site-packages/nvidia/cusparse/lib:${VENV}/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/models/dreamgen:${PYTHONPATH:-}"

# --- Checkpoint ---
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/exp_logs/dreamgen_logs/posttraining/multiview/all_multiview/checkpoints_40_per}"
DIT_PATH="${DIT_PATH:-${CHECKPOINTS_DIR}/iter_000070000/model_ema_bf16.pt}"

# --- Inference settings ---
NUM_GPUS="${NUM_GPUS:-4}"
MODEL_SIZE="${MODEL_SIZE:-2B}"
RESOLUTION="${RESOLUTION:-480}"
FPS="${FPS:-16}"
MASTER_PORT="${MASTER_PORT:-29501}"
GUIDANCE="${GUIDANCE:-7}"
MAX_EPISODES="${MAX_EPISODES:-}"

# --- Output ---
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/dense/humanoid/output/multiview/dreamgen_multiview}"
INPUT_ROOT="${INPUT_ROOT:-${ROOT}/sampling_dataset/humanoid/multiview/input_grid_70k}"

echo "============================================"
echo "DreamGen Humanoid Multiview Grid 70k Test"
echo "  Checkpoint: ${DIT_PATH}"
echo "  Model:      ${MODEL_SIZE} | Resolution: ${RESOLUTION} | FPS: ${FPS}"
echo "  GPUs:       ${NUM_GPUS}"
echo "  Guidance:   ${GUIDANCE}"
echo "  Save root:  ${SAVE_ROOT}"
echo "  Episodes:   ${MAX_EPISODES:-all}"
echo "============================================"

# --- Step 1: Prepare grid input images + batch_input.json ---
echo ""
echo "[STEP 1] Preparing 2x2 grid inputs..."
python "${ROOT}/scripts/prepare_multiview_humanoid_grid_batch.py" \
  --datasets-root "${ROOT}/datasets/humanoid/multiview" \
  --save-root "${SAVE_ROOT}" \
  --input-root "${INPUT_ROOT}" \
  --iter-name "iter_000070000" \
  ${MAX_EPISODES:+--max-episodes "${MAX_EPISODES}"}

# --- Step 2: Run inference on each batch_input.json ---
echo ""
echo "[STEP 2] Running DreamGen inference..."

find "${SAVE_ROOT}" -name "batch_input.json" -path "*/iter_000070000/*" | sort | while read -r batch_json; do
  rel_path="${batch_json#${SAVE_ROOT}/}"
  echo ""
  echo "  === Processing: ${rel_path} ==="

  torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    -m models.dreamgen.examples.video2world \
    --model_size "${MODEL_SIZE}" \
    --resolution "${RESOLUTION}" \
    --fps "${FPS}" \
    --dit_path "${DIT_PATH}" \
    --batch_input_json "${batch_json}" \
    --guidance "${GUIDANCE}" \
    --num_gpus "${NUM_GPUS}" \
    --disable_guardrail

  echo "  === Done: ${rel_path} ==="
done

echo ""
echo "All done. Results saved under: ${SAVE_ROOT}"
