#!/usr/bin/env bash
# =============================================================================
# run_multiview_humanoid_batch.sh
# Run DreamGen multiview inference on a humanoid LeRobot dataset.
#
# Usage (from the repo root):
#   bash scripts/run_multiview_humanoid_batch.sh
#
# Override defaults via environment variables:
#   DATASET_PATH, INPUT_DIR, OUTPUT_DIR, BATCH_JSON,
#   MAX_EPISODES, NUM_GPUS, MODEL_SIZE, MASTER_PORT
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Paths (all relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DATASET_PATH="${DATASET_PATH:-./datasets/humanoid/multiview/non_makovian/Airbot_MMK2_Airbot_MMK2_storage_mango_pomegranate}"
DATASET_NAME="$(basename "${DATASET_PATH}")"

INPUT_DIR="${INPUT_DIR:-./sampling_dataset/humanoid/multiview/input}"
OUTPUT_DIR="${OUTPUT_DIR:-./sampling_dataset/humanoid/multiview/non_makovian/dreamgen/${DATASET_NAME}}"
BATCH_JSON="${BATCH_JSON:-${INPUT_DIR}/${DATASET_NAME}/batch_input.json}"

MODEL_SIZE="${MODEL_SIZE:-14B}"
NUM_GPUS="${NUM_GPUS:-2}"
MASTER_PORT="${MASTER_PORT:-12341}"
FPS="${FPS:-10}"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
echo "============================================================"
echo " DreamGen Multiview Humanoid Inference"
echo "============================================================"
echo " Dataset  : ${DATASET_PATH}"
echo " Input dir: ${INPUT_DIR}/${DATASET_NAME}"
echo " Output   : ${OUTPUT_DIR}"
echo " Batch    : ${BATCH_JSON}"
echo " Model    : ${MODEL_SIZE}  GPUs: ${NUM_GPUS}  FPS: ${FPS}"
echo "============================================================"
echo ""

# Activate venv
source "${REPO_ROOT}/models/DreamDojo/.venv/bin/activate"

# Expose DreamGen modules
export PYTHONPATH="${REPO_ROOT}/models/dreamgen${PYTHONPATH:+:$PYTHONPATH}"

# ---------------------------------------------------------------------------
# Step 1: Prepare grid images + batch_input.json
# ---------------------------------------------------------------------------
echo "[STEP 1] Preparing input grid images and batch_input.json ..."

PREPARE_ARGS=(
    --dataset_path "${DATASET_PATH}"
    --input_dir    "${INPUT_DIR}"
    --output_dir   "${OUTPUT_DIR}"
    --batch_json   "${BATCH_JSON}"
    --skip_existing
)

if [[ -n "${MAX_EPISODES:-}" ]]; then
    PREPARE_ARGS+=(--max_episodes "${MAX_EPISODES}")
fi

python scripts/prepare_multiview_humanoid_batch.py "${PREPARE_ARGS[@]}"

echo ""
echo "[STEP 1] Done. Batch JSON: ${BATCH_JSON}"
echo ""

# ---------------------------------------------------------------------------
# Step 2: Run DreamGen inference
# ---------------------------------------------------------------------------
echo "[STEP 2] Running DreamGen inference (${MODEL_SIZE}, ${NUM_GPUS} GPU(s)) ..."

# Validate GPU count
AVAILABLE_GPUS=$(python - <<'PY'
import torch, sys
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)

if [[ "${AVAILABLE_GPUS}" -lt 1 ]]; then
    echo "[ERROR] No CUDA GPUs visible to PyTorch." >&2
    exit 1
fi
if [[ "${NUM_GPUS}" -gt "${AVAILABLE_GPUS}" ]]; then
    echo "[ERROR] NUM_GPUS=${NUM_GPUS} but only ${AVAILABLE_GPUS} GPU(s) available." >&2
    exit 1
fi

INFERENCE_ARGS=(
    -m examples.video2world_gr00t
    --model_size      "${MODEL_SIZE}"
    --gr00t_variant   droid
    --batch_input_json "${BATCH_JSON}"
    --prompt_prefix   ""
    --num_gpus        "${NUM_GPUS}"
    --disable_guardrail
    --fps             "${FPS}"
)

mkdir -p "${OUTPUT_DIR}"

if [[ "${NUM_GPUS}" -gt 1 ]]; then
    torchrun \
        --nproc_per_node="${NUM_GPUS}" \
        --master_port="${MASTER_PORT}" \
        "${INFERENCE_ARGS[@]}"
else
    python "${INFERENCE_ARGS[@]}"
fi

echo ""
echo "[STEP 2] Inference complete. Videos saved to: ${OUTPUT_DIR}"
