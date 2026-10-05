#!/bin/bash
# Run Ctrl-World inference with SiTo (token pruning) on single_arm/multiview dataset
#
# Output: sampling_dataset/sito/single_arm/{output,input}/multiview/ctrlworld/{subset}/
#
# Usage:
#   bash scripts/run_single_arm_multiview_ctrlworld_sito.sh [makovian|non_makovian|both]
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_single_arm_multiview_ctrlworld_sito.sh both

set -euo pipefail

SUBSET=${1:-both}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

source "${PROJECT_ROOT}/models/Ctrl-World/.venv/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}/models/Ctrl-World:${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

# Model paths
SVD_MODEL_PATH="./checkpoints/base/stabilityai/stable-video-diffusion-img2vid"
CLIP_MODEL_PATH="./checkpoints/base/openai/clip-vit-base-patch32"
CKPT_PATH="./checkpoints/base/ctrl_world/Ctrl-World/checkpoint-10000.pt"
DATASET_META_INFO="./models/Ctrl-World/dataset_meta_info"
DATASET_EXAMPLE_DIR="./models/Ctrl-World/dataset_example"

# SiTo parameters (spatio-temporal hold)
# Chosen config = spatio-temporal prune-once-hold: prune spatial tokens once per
# episode and keep them compressed through both the spatial AND temporal blocks.
# keep_ratio=0.9 on the largest stage only (max_downsample_ratio=1) is the
# quality-leaning point of this method (PSNR ~18.9 on episode_000255). Push
# ST_KEEP_RATIO lower (e.g. 0.5, max_downsample_ratio=4) for up to ~1.16x speed
# at lower quality. Overridable via env.
SITO_MAX_DOWNSAMPLE_RATIO="${SITO_MAX_DOWNSAMPLE_RATIO:-1}"
SITO_ST_KEEP_RATIO="${SITO_ST_KEEP_RATIO:-0.9}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-25}"

EXTRA=""
if [ -n "${NUM_EPISODES:-}" ]; then EXTRA="${EXTRA} --num_episodes ${NUM_EPISODES}"; fi

OUTPUT_BASE="${OUTPUT_ROOT:-${PROJECT_ROOT}/sampling_dataset/sito/single_arm/output/multiview/ctrlworld}"
INPUT_BASE="${INPUT_ROOT:-${PROJECT_ROOT}/sampling_dataset/sito/single_arm/input/multiview/ctrlworld}"

echo "=========================================================="
echo "Ctrl-World + SiTo | Single-Arm Multiview Inference"
echo "Subset: ${SUBSET}"
echo "Params: max_downsample_ratio=${SITO_MAX_DOWNSAMPLE_RATIO}, st_keep_ratio=${SITO_ST_KEEP_RATIO}, steps=${NUM_INFERENCE_STEPS}"
echo "=========================================================="

# Step 1: Prepare dataset
echo ""
if [ "${SKIP_PREPARE:-0}" = "1" ]; then echo "[Step 1] skipped (SKIP_PREPARE=1)"; else
echo "[Step 1] Preparing dataset (LeRoBot -> Ctrl-World format)..."
python scripts/prepare_ctrlworld_single_arm_multiview.py \
    --subset "${SUBSET}" \
    --output-dir "${DATASET_EXAMPLE_DIR}" \
    --dataset-base "${PROJECT_ROOT}/datasets/single_arm/multiview"
fi

# Step 2: Run inference with SiTo
if [ "${SUBSET}" = "both" ] || [ "${SUBSET}" = "makovian" ]; then
    echo ""
    echo "[Step 2a] Running SiTo inference on MAKOVIAN subset..."
    python scripts/infer_single_arm_multiview_ctrlworld.py \
        --svd_model_path "${SVD_MODEL_PATH}" \
        --clip_model_path "${CLIP_MODEL_PATH}" \
        --ckpt_path "${CKPT_PATH}" \
        --subset makovian \
        --dataset_example_dir "${DATASET_EXAMPLE_DIR}" \
        --dataset_meta_info_path "${DATASET_META_INFO}" \
        --save_dir "${OUTPUT_BASE}/makovian" \
        --input_save_dir "${INPUT_BASE}/makovian" \
        --num_inference_steps "${NUM_INFERENCE_STEPS}" ${EXTRA} \
        --use_sito \
        --sito_spatiotemporal_hold \
        --sito_st_keep_ratio "${SITO_ST_KEEP_RATIO}" \
        --sito_max_downsample_ratio "${SITO_MAX_DOWNSAMPLE_RATIO}"
fi

if [ "${SUBSET}" = "both" ] || [ "${SUBSET}" = "non_makovian" ]; then
    echo ""
    echo "[Step 2b] Running SiTo inference on NON_MAKOVIAN subset..."
    python scripts/infer_single_arm_multiview_ctrlworld.py \
        --svd_model_path "${SVD_MODEL_PATH}" \
        --clip_model_path "${CLIP_MODEL_PATH}" \
        --ckpt_path "${CKPT_PATH}" \
        --subset non_makovian \
        --dataset_example_dir "${DATASET_EXAMPLE_DIR}" \
        --dataset_meta_info_path "${DATASET_META_INFO}" \
        --save_dir "${OUTPUT_BASE}/non_makovian" \
        --input_save_dir "${INPUT_BASE}/non_makovian" \
        --num_inference_steps "${NUM_INFERENCE_STEPS}" ${EXTRA} \
        --use_sito \
        --sito_spatiotemporal_hold \
        --sito_st_keep_ratio "${SITO_ST_KEEP_RATIO}" \
        --sito_max_downsample_ratio "${SITO_MAX_DOWNSAMPLE_RATIO}"
fi

echo ""
echo "=========================================================="
echo "Done! Results saved to:"
echo "  Output: ${OUTPUT_BASE}/{makovian,non_makovian}"
echo "  Input:  ${INPUT_BASE}/{makovian,non_makovian}"
echo "=========================================================="
