#!/bin/bash
# Run Ctrl-World inference with DiCache on single_arm/multiview dataset (makovian & non_makovian)
#
# Output: sampling_dataset/dicache/single_arm/{output,input}/multiview/ctrlworld/{subset}/
#
# Usage:
#   bash scripts/run_single_arm_multiview_ctrlworld_dicache.sh [makovian|non_makovian|both]
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_single_arm_multiview_ctrlworld_dicache.sh both

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
SVD_MODEL_PATH="./checkpoints/base/stabilityai/stable-video-diffusion-img2vid"
CLIP_MODEL_PATH="./checkpoints/base/openai/clip-vit-base-patch32"
CKPT_PATH="${CKPT:-./checkpoints/base/ctrl_world/Ctrl-World/checkpoint-10000.pt}"
DATASET_META_INFO="./models/Ctrl-World/dataset_meta_info"
DATASET_EXAMPLE_DIR="./models/Ctrl-World/dataset_example"

# DiCache parameters (quality-focused: stricter threshold, later start)
DICACHE_REL_L1_THRESH="${DICACHE_REL_L1_THRESH:-0.06}"
DICACHE_RET_RATIO="${DICACHE_RET_RATIO:-0.3}"
DICACHE_PROBE_DEPTH="${DICACHE_PROBE_DEPTH:-3}"

OUTPUT_BASE="${OUTPUT_ROOT:-${PROJECT_ROOT}/sampling_dataset/dicache/single_arm/output/multiview/ctrlworld}"
INPUT_BASE="${INPUT_ROOT:-${PROJECT_ROOT}/sampling_dataset/dicache/single_arm/input/multiview/ctrlworld}"

EXTRA=""
if [ -n "${NUM_EPISODES:-}" ]; then EXTRA="${EXTRA} --num_episodes ${NUM_EPISODES}"; fi
if [ -n "${NUM_INFERENCE_STEPS:-}" ]; then EXTRA="${EXTRA} --num_inference_steps ${NUM_INFERENCE_STEPS}"; fi

echo "=========================================================="
echo "Ctrl-World + DiCache | Single-Arm Multiview Inference"
echo "Subset: ${SUBSET}"
echo "Params: thresh=${DICACHE_REL_L1_THRESH}, ret_ratio=${DICACHE_RET_RATIO}, probe_depth=${DICACHE_PROBE_DEPTH}"
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

# Step 2: Run inference with DiCache
if [ "${SUBSET}" = "both" ] || [ "${SUBSET}" = "makovian" ]; then
    echo ""
    echo "[Step 2a] Running DiCache inference on MAKOVIAN subset..."
    python scripts/infer_single_arm_multiview_ctrlworld.py \
        --svd_model_path "${SVD_MODEL_PATH}" \
        --clip_model_path "${CLIP_MODEL_PATH}" \
        --ckpt_path "${CKPT_PATH}" \
        --subset makovian \
        --dataset_example_dir "${DATASET_EXAMPLE_DIR}" \
        --dataset_meta_info_path "${DATASET_META_INFO}" \
        --save_dir "${OUTPUT_BASE}/makovian" \
        --input_save_dir "${INPUT_BASE}/makovian" \
        --use_dicache \
        --dicache_rel_l1_thresh "${DICACHE_REL_L1_THRESH}" \
        --dicache_ret_ratio "${DICACHE_RET_RATIO}" \
        --dicache_probe_depth "${DICACHE_PROBE_DEPTH}" ${EXTRA}
fi

if [ "${SUBSET}" = "both" ] || [ "${SUBSET}" = "non_makovian" ]; then
    echo ""
    echo "[Step 2b] Running DiCache inference on NON_MAKOVIAN subset..."
    python scripts/infer_single_arm_multiview_ctrlworld.py \
        --svd_model_path "${SVD_MODEL_PATH}" \
        --clip_model_path "${CLIP_MODEL_PATH}" \
        --ckpt_path "${CKPT_PATH}" \
        --subset non_makovian \
        --dataset_example_dir "${DATASET_EXAMPLE_DIR}" \
        --dataset_meta_info_path "${DATASET_META_INFO}" \
        --save_dir "${OUTPUT_BASE}/non_makovian" \
        --input_save_dir "${INPUT_BASE}/non_makovian" \
        --use_dicache \
        --dicache_rel_l1_thresh "${DICACHE_REL_L1_THRESH}" \
        --dicache_ret_ratio "${DICACHE_RET_RATIO}" \
        --dicache_probe_depth "${DICACHE_PROBE_DEPTH}" ${EXTRA}
fi

echo ""
echo "=========================================================="
echo "Done! Results saved to:"
echo "  Output: ${OUTPUT_BASE}/{makovian,non_makovian}"
echo "  Input:  ${INPUT_BASE}/{makovian,non_makovian}"
echo "=========================================================="
