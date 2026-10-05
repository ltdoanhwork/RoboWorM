#!/bin/bash
# Run Ctrl-World inference with FasterCache on single_arm/multiview dataset (makovian & non_makovian)
#
# Output: sampling_dataset/fastercache/single_arm/{output,input}/multiview/ctrlworld/{subset}/
#
# Usage:
#   bash scripts/run_single_arm_multiview_ctrlworld_fastercache.sh [makovian|non_makovian|both]
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_single_arm_multiview_ctrlworld_fastercache.sh both

set -euo pipefail

SUBSET=${1:-both}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# Activate environment
source "${PROJECT_ROOT}/models/Ctrl-World/.venv/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}/models/Ctrl-World:${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

# Model paths
SVD_MODEL_PATH="./checkpoints/stabilityai/stable-video-diffusion-img2vid"
CLIP_MODEL_PATH="./checkpoints/openai/clip-vit-base-patch32"
CKPT_PATH="${CKPT:-./checkpoints/ctrl_world/Ctrl-World/checkpoint-10000.pt}"
DATASET_META_INFO="./models/Ctrl-World/dataset_meta_info"
DATASET_EXAMPLE_DIR="./models/Ctrl-World/dataset_example"

# FasterCache parameters
FASTERCACHE_START_STEP="${FASTERCACHE_START_STEP:-0}"
FASTERCACHE_MODEL_INTERVAL="${FASTERCACHE_MODEL_INTERVAL:-5}"
FASTERCACHE_BLOCK_INTERVAL="${FASTERCACHE_BLOCK_INTERVAL:-3}"

OUTPUT_BASE="${OUTPUT_ROOT:-${PROJECT_ROOT}/sampling_dataset/fastercache/single_arm/output/multiview/ctrlworld}"
INPUT_BASE="${INPUT_ROOT:-${PROJECT_ROOT}/sampling_dataset/fastercache/single_arm/input/multiview/ctrlworld}"

EXTRA=""
if [ -n "${NUM_EPISODES:-}" ]; then EXTRA="${EXTRA} --num_episodes ${NUM_EPISODES}"; fi
if [ -n "${NUM_INFERENCE_STEPS:-}" ]; then EXTRA="${EXTRA} --num_inference_steps ${NUM_INFERENCE_STEPS}"; fi

echo "=========================================================="
echo "Ctrl-World + FasterCache | Single-Arm Multiview Inference"
echo "Subset: ${SUBSET}"
echo "Params: start_step=${FASTERCACHE_START_STEP}, model_interval=${FASTERCACHE_MODEL_INTERVAL}, block_interval=${FASTERCACHE_BLOCK_INTERVAL}"
echo "=========================================================="

# Step 1: Prepare dataset
if [ "${SKIP_PREPARE:-0}" != "1" ]; then
echo ""
echo "[Step 1] Preparing dataset (LeRoBot -> Ctrl-World format)..."
python scripts/prepare_ctrlworld_single_arm_multiview.py \
    --subset "${SUBSET}" \
    --output-dir "${DATASET_EXAMPLE_DIR}" \
    --dataset-base "${PROJECT_ROOT}/datasets/single_arm/multiview"
fi

# Step 2: Run inference with FasterCache
if [ "${SUBSET}" = "both" ] || [ "${SUBSET}" = "makovian" ]; then
    echo ""
    echo "[Step 2a] Running FasterCache inference on MAKOVIAN subset..."
    python scripts/infer_single_arm_multiview_ctrlworld.py \
        --svd_model_path "${SVD_MODEL_PATH}" \
        --clip_model_path "${CLIP_MODEL_PATH}" \
        --ckpt_path "${CKPT_PATH}" \
        --subset makovian \
        --dataset_example_dir "${DATASET_EXAMPLE_DIR}" \
        --dataset_meta_info_path "${DATASET_META_INFO}" \
        --save_dir "${OUTPUT_BASE}/makovian" \
        --input_save_dir "${INPUT_BASE}/makovian" \
        --use_fastercache \
        --fastercache_start_step "${FASTERCACHE_START_STEP}" \
        --fastercache_model_interval "${FASTERCACHE_MODEL_INTERVAL}" \
        --fastercache_block_interval "${FASTERCACHE_BLOCK_INTERVAL}" ${EXTRA}
fi

if [ "${SUBSET}" = "both" ] || [ "${SUBSET}" = "non_makovian" ]; then
    echo ""
    echo "[Step 2b] Running FasterCache inference on NON_MAKOVIAN subset..."
    python scripts/infer_single_arm_multiview_ctrlworld.py \
        --svd_model_path "${SVD_MODEL_PATH}" \
        --clip_model_path "${CLIP_MODEL_PATH}" \
        --ckpt_path "${CKPT_PATH}" \
        --subset non_makovian \
        --dataset_example_dir "${DATASET_EXAMPLE_DIR}" \
        --dataset_meta_info_path "${DATASET_META_INFO}" \
        --save_dir "${OUTPUT_BASE}/non_makovian" \
        --input_save_dir "${INPUT_BASE}/non_makovian" \
        --use_fastercache \
        --fastercache_start_step "${FASTERCACHE_START_STEP}" \
        --fastercache_model_interval "${FASTERCACHE_MODEL_INTERVAL}" \
        --fastercache_block_interval "${FASTERCACHE_BLOCK_INTERVAL}" ${EXTRA}
fi

echo ""
echo "=========================================================="
echo "Done! Results saved to:"
echo "  Output: ${OUTPUT_BASE}/{makovian,non_makovian}"
echo "  Input:  ${INPUT_BASE}/{makovian,non_makovian}"
echo "=========================================================="
