#!/usr/bin/env bash
# =============================================================================
# run_multiview_airbot_all.sh
# Run DreamGen multiview inference on ALL Airbot MMK2 datasets.
#
# Loops over all 6 Airbot datasets, auto-detects camera layout (A vs B),
# prepares grid images, and runs inference with 5 episodes per dataset.
#
# Usage (from the repo root):
#   bash scripts/run_multiview_airbot_all.sh
#
# Override defaults via environment variables:
#   MAX_EPISODES, NUM_GPUS, MODEL_SIZE, MASTER_PORT, FPS
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="./datasets/humanoid/multiview/non_makovian"
INPUT_DIR="./sampling_dataset/humanoid/multiview/input"
OUTPUT_ROOT="./sampling_dataset/humanoid/multiview/non_makovian/dreamgen"

MAX_EPISODES="${MAX_EPISODES:-5}"
MODEL_SIZE="${MODEL_SIZE:-14B}"
NUM_GPUS="${NUM_GPUS:-2}"
MASTER_PORT="${MASTER_PORT:-12341}"
FPS="${FPS:-10}"

# All 6 Airbot datasets (Group A and B)
AIRBOT_DATASETS=(
    "Airbot_MMK2_Airbot_MMK2_storage_mango_pomegranate"
    "Airbot_MMK2_doodled_line"
    "Airbot_MMK2_move_pan"
    "Airbot_MMK2_storage_bottle_part"
    "AIRBOT_MMK2_boxs_storage"
    "AIRBOT_MMK2_mobile_car"
)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
echo "============================================================"
echo " DreamGen Multiview — ALL Airbot MMK2 Datasets"
echo "============================================================"
echo " Data root   : ${DATA_ROOT}"
echo " Output root : ${OUTPUT_ROOT}"
echo " Max episodes: ${MAX_EPISODES} per dataset"
echo " Model       : ${MODEL_SIZE}  GPUs: ${NUM_GPUS}  FPS: ${FPS}"
echo " Datasets    : ${#AIRBOT_DATASETS[@]}"
echo "============================================================"
echo ""

# Activate venv
source "${REPO_ROOT}/models/DreamDojo/.venv/bin/activate"

# Expose DreamGen modules
export PYTHONPATH="${REPO_ROOT}/models/dreamgen${PYTHONPATH:+:$PYTHONPATH}"

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

echo "GPUs: ${NUM_GPUS} / ${AVAILABLE_GPUS} available"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Prepare all datasets (grid images + batch JSON)
# ---------------------------------------------------------------------------
echo "================================================================"
echo " STEP 1: Preparing grid images for all datasets"
echo "================================================================"
echo ""

for DS_NAME in "${AIRBOT_DATASETS[@]}"; do
    DS_PATH="${DATA_ROOT}/${DS_NAME}"
    if [[ ! -d "${DS_PATH}" ]]; then
        echo "[WARN] Dataset not found, skipping: ${DS_NAME}"
        continue
    fi

    echo "--- Preparing: ${DS_NAME} ---"
    python scripts/prepare_multiview_humanoid_batch.py \
        --dataset_path "${DS_PATH}" \
        --input_dir    "${INPUT_DIR}" \
        --output_dir   "${OUTPUT_ROOT}/${DS_NAME}" \
        --cam_layout   auto \
        --max_episodes "${MAX_EPISODES}" \
        --skip_existing
    echo ""
done

echo ""
echo "[STEP 1] All datasets prepared."
echo ""

# ---------------------------------------------------------------------------
# Step 2: Run DreamGen inference for each dataset
# ---------------------------------------------------------------------------
echo "================================================================"
echo " STEP 2: Running inference for all datasets"
echo "================================================================"
echo ""

for DS_NAME in "${AIRBOT_DATASETS[@]}"; do
    BATCH_JSON="${INPUT_DIR}/${DS_NAME}/batch_input.json"
    if [[ ! -f "${BATCH_JSON}" ]]; then
        echo "[WARN] No batch_input.json for ${DS_NAME}, skipping inference."
        continue
    fi

    echo "--- Inference: ${DS_NAME} ---"
    mkdir -p "${OUTPUT_ROOT}/${DS_NAME}"

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

    if [[ "${NUM_GPUS}" -gt 1 ]]; then
        torchrun \
            --nproc_per_node="${NUM_GPUS}" \
            --master_port="${MASTER_PORT}" \
            "${INFERENCE_ARGS[@]}"
    else
        python "${INFERENCE_ARGS[@]}"
    fi

    echo ""
    echo "[OK] ${DS_NAME} — inference complete."
    echo ""
done

echo "============================================================"
echo " ALL DONE — Videos saved under: ${OUTPUT_ROOT}/"
echo "============================================================"
