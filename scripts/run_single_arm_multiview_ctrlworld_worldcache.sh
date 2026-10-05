#!/bin/bash
# Run Ctrl-World inference with WorldCache on single_arm/multiview dataset (makovian & non_makovian)
#
# Output: sampling_dataset/worldcache/single_arm/{output,input}/multiview/ctrlworld/{subset}/
#
# Usage:
#   bash scripts/run_single_arm_multiview_ctrlworld_worldcache.sh [makovian|non_makovian|both]
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_single_arm_multiview_ctrlworld_worldcache.sh both

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

# WorldCache parameters (speed-focused: higher threshold, earlier start)
WORLDCACHE_REL_L1_THRESH="${WORLDCACHE_REL_L1_THRESH:-0.08}"
WORLDCACHE_RET_RATIO="${WORLDCACHE_RET_RATIO:-0.30}"
WORLDCACHE_PROBE_DEPTH="${WORLDCACHE_PROBE_DEPTH:-2}"
WORLDCACHE_MOTION_SENSITIVITY="${WORLDCACHE_MOTION_SENSITIVITY:-2.5}"

OUTPUT_BASE="${OUTPUT_ROOT:-${PROJECT_ROOT}/sampling_dataset/worldcache/single_arm/output/multiview/ctrlworld}"
INPUT_BASE="${INPUT_ROOT:-${PROJECT_ROOT}/sampling_dataset/worldcache/single_arm/input/multiview/ctrlworld}"

EXTRA=""
if [ -n "${NUM_EPISODES:-}" ]; then EXTRA="${EXTRA} --num_episodes ${NUM_EPISODES}"; fi
if [ -n "${NUM_INFERENCE_STEPS:-}" ]; then EXTRA="${EXTRA} --num_inference_steps ${NUM_INFERENCE_STEPS}"; fi

echo "=========================================================="
echo "Ctrl-World + WorldCache | Single-Arm Multiview Inference"
echo "Subset: ${SUBSET}"
echo "Params: thresh=${WORLDCACHE_REL_L1_THRESH}, ret_ratio=${WORLDCACHE_RET_RATIO}, probe_depth=${WORLDCACHE_PROBE_DEPTH}, motion_sens=${WORLDCACHE_MOTION_SENSITIVITY}"
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

# Step 2: Run inference with WorldCache
if [ "${SUBSET}" = "both" ] || [ "${SUBSET}" = "makovian" ]; then
    echo ""
    echo "[Step 2a] Running WorldCache inference on MAKOVIAN subset..."
    python scripts/infer_single_arm_multiview_ctrlworld.py \
        --svd_model_path "${SVD_MODEL_PATH}" \
        --clip_model_path "${CLIP_MODEL_PATH}" \
        --ckpt_path "${CKPT_PATH}" \
        --subset makovian \
        --dataset_example_dir "${DATASET_EXAMPLE_DIR}" \
        --dataset_meta_info_path "${DATASET_META_INFO}" \
        --save_dir "${OUTPUT_BASE}/makovian" \
        --input_save_dir "${INPUT_BASE}/makovian" \
        --use_worldcache \
        --worldcache_rel_l1_thresh "${WORLDCACHE_REL_L1_THRESH}" \
        --worldcache_ret_ratio "${WORLDCACHE_RET_RATIO}" \
        --worldcache_probe_depth "${WORLDCACHE_PROBE_DEPTH}" \
        --worldcache_motion_sensitivity "${WORLDCACHE_MOTION_SENSITIVITY}" ${EXTRA}
fi

if [ "${SUBSET}" = "both" ] || [ "${SUBSET}" = "non_makovian" ]; then
    echo ""
    echo "[Step 2b] Running WorldCache inference on NON_MAKOVIAN subset..."
    python scripts/infer_single_arm_multiview_ctrlworld.py \
        --svd_model_path "${SVD_MODEL_PATH}" \
        --clip_model_path "${CLIP_MODEL_PATH}" \
        --ckpt_path "${CKPT_PATH}" \
        --subset non_makovian \
        --dataset_example_dir "${DATASET_EXAMPLE_DIR}" \
        --dataset_meta_info_path "${DATASET_META_INFO}" \
        --save_dir "${OUTPUT_BASE}/non_makovian" \
        --input_save_dir "${INPUT_BASE}/non_makovian" \
        --use_worldcache \
        --worldcache_rel_l1_thresh "${WORLDCACHE_REL_L1_THRESH}" \
        --worldcache_ret_ratio "${WORLDCACHE_RET_RATIO}" \
        --worldcache_probe_depth "${WORLDCACHE_PROBE_DEPTH}" \
        --worldcache_motion_sensitivity "${WORLDCACHE_MOTION_SENSITIVITY}" ${EXTRA}
fi

echo ""
echo "=========================================================="
echo "Done! Results saved to:"
echo "  Output: ${OUTPUT_BASE}/{makovian,non_makovian}"
echo "  Input:  ${INPUT_BASE}/{makovian,non_makovian}"
echo "=========================================================="
