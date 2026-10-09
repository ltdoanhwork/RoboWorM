#!/bin/bash
# Run Ctrl-World inference with ITM (Iterative Token Merging) on single_arm/multiview dataset
#
# Output: sampling_dataset/itm/single_arm/{output,input}/multiview/ctrlworld/{subset}/
#
# Usage:
#   bash scripts/run/ctrlworld/run_single_arm_multiview_ctrlworld_itm.sh [makovian|non_makovian|both]
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run/ctrlworld/run_single_arm_multiview_ctrlworld_itm.sh both

set -euo pipefail

SUBSET=${1:-both}

PROJECT_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "${PROJECT_ROOT}"

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

# ITM parameters (spatio-temporal hold)
# Chosen config = spatio-temporal prune-once-hold (shared with SiTo): prune
# spatial tokens once per episode and keep them compressed through both the
# spatial AND temporal blocks. keep_ratio=0.8 across all stages
# (max_downsample_ratio=4) gives ~1.04x speedup at PSNR ~16.4 on episode_000255.
# Lower ITM_ST_KEEP_RATIO (e.g. 0.5) for up to ~1.16x at lower quality.
ITM_MAX_DOWNSAMPLE_RATIO="${ITM_MAX_DOWNSAMPLE_RATIO:-4}"
ITM_ST_KEEP_RATIO="${ITM_ST_KEEP_RATIO:-0.8}"

OUTPUT_BASE="${OUTPUT_ROOT:-${PROJECT_ROOT}/sampling_dataset/itm/single_arm/output/multiview/ctrlworld}"
INPUT_BASE="${INPUT_ROOT:-${PROJECT_ROOT}/sampling_dataset/itm/single_arm/input/multiview/ctrlworld}"

# Steps: same as the dense launcher (run_single_arm_multiview_ctrlworld.sh) — no
# flag unless NUM_INFERENCE_STEPS is set, so the model config default (50) applies.
# (Before 2026-10-09 this launcher forced 25 steps.)
EXTRA=""
if [ -n "${NUM_EPISODES:-}" ]; then EXTRA="${EXTRA} --num_episodes ${NUM_EPISODES}"; fi
if [ -n "${NUM_INFERENCE_STEPS:-}" ]; then EXTRA="${EXTRA} --num_inference_steps ${NUM_INFERENCE_STEPS}"; fi

echo "=========================================================="
echo "Ctrl-World + ITM | Single-Arm Multiview Inference"
echo "Subset: ${SUBSET}"
echo "Params: max_downsample_ratio=${ITM_MAX_DOWNSAMPLE_RATIO}, st_keep_ratio=${ITM_ST_KEEP_RATIO}, steps=${NUM_INFERENCE_STEPS:-config default (50)}"
echo "=========================================================="

# Step 1: Prepare dataset
if [ "${SKIP_PREPARE:-0}" != "1" ]; then
echo ""
echo "[Step 1] Preparing dataset (LeRoBot -> Ctrl-World format)..."
python scripts/prepare/prepare_ctrlworld_single_arm_multiview.py \
    --subset "${SUBSET}" \
    --output-dir "${DATASET_EXAMPLE_DIR}" \
    --dataset-base "${PROJECT_ROOT}/datasets/single_arm/multiview"
fi

# Step 2: Run inference with ITM
if [ "${SUBSET}" = "both" ] || [ "${SUBSET}" = "makovian" ]; then
    echo ""
    echo "[Step 2a] Running ITM inference on MAKOVIAN subset..."
    python scripts/infer/infer_single_arm_multiview_ctrlworld.py \
        --svd_model_path "${SVD_MODEL_PATH}" \
        --clip_model_path "${CLIP_MODEL_PATH}" \
        --ckpt_path "${CKPT_PATH}" \
        --subset makovian \
        --dataset_example_dir "${DATASET_EXAMPLE_DIR}" \
        --dataset_meta_info_path "${DATASET_META_INFO}" \
        --save_dir "${OUTPUT_BASE}/makovian" \
        --input_save_dir "${INPUT_BASE}/makovian" \
        --use_itm \
        --itm_spatiotemporal_hold \
        --itm_st_keep_ratio "${ITM_ST_KEEP_RATIO}" \
        --itm_max_downsample_ratio "${ITM_MAX_DOWNSAMPLE_RATIO}" ${EXTRA}
fi

if [ "${SUBSET}" = "both" ] || [ "${SUBSET}" = "non_makovian" ]; then
    echo ""
    echo "[Step 2b] Running ITM inference on NON_MAKOVIAN subset..."
    python scripts/infer/infer_single_arm_multiview_ctrlworld.py \
        --svd_model_path "${SVD_MODEL_PATH}" \
        --clip_model_path "${CLIP_MODEL_PATH}" \
        --ckpt_path "${CKPT_PATH}" \
        --subset non_makovian \
        --dataset_example_dir "${DATASET_EXAMPLE_DIR}" \
        --dataset_meta_info_path "${DATASET_META_INFO}" \
        --save_dir "${OUTPUT_BASE}/non_makovian" \
        --input_save_dir "${INPUT_BASE}/non_makovian" \
        --use_itm \
        --itm_spatiotemporal_hold \
        --itm_st_keep_ratio "${ITM_ST_KEEP_RATIO}" \
        --itm_max_downsample_ratio "${ITM_MAX_DOWNSAMPLE_RATIO}" ${EXTRA}
fi

echo ""
echo "=========================================================="
echo "Done! Results saved to:"
echo "  Output: ${OUTPUT_BASE}/{makovian,non_makovian}"
echo "  Input:  ${INPUT_BASE}/{makovian,non_makovian}"
echo "=========================================================="
