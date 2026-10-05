#!/bin/bash
# Ctrl-World inference with Radial Attention on single_arm/multiview.
# Mirrors run_single_arm_multiview_ctrlworld.sh (dense), swapping
# sampling_dataset/dense -> radial and adding Radial flags.
#
# Output: sampling_dataset/radial/single_arm/{output,input}/multiview/ctrlworld/{subset}/
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_single_arm_multiview_ctrlworld_radial.sh [makovian|non_makovian|both]
# Env: CKPT, NUM_EPISODES, NUM_INFERENCE_STEPS, OUTPUT_ROOT, INPUT_ROOT, SKIP_PREPARE,
#      RADIAL_{DECAY,BLOCK_SIZE,FIRST_LAYERS_FP,MODEL_TYPE}

set -euo pipefail

SUBSET=${1:-both}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

source "${PROJECT_ROOT}/models/Ctrl-World/.venv/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}/models/Ctrl-World:${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

SVD_MODEL_PATH="./checkpoints/base/stabilityai/stable-video-diffusion-img2vid"
CLIP_MODEL_PATH="./checkpoints/base/openai/clip-vit-base-patch32"
CKPT_PATH="${CKPT:-./checkpoints/base/ctrl_world/Ctrl-World/checkpoint-10000.pt}"
DATASET_META_INFO="./models/Ctrl-World/dataset_meta_info"
DATASET_EXAMPLE_DIR="./models/Ctrl-World/dataset_example"

# Radial -- TUNED (see ctrlworld-single-arm-multiview-sparse-attention-insights).
# fp4 is the best sparse quality point (PSNR 21.38 ≈ dense), still 0.92x (slower)
# — net-negative on this short-seq SVD UNet, kept as a valid benchmark point.
RADIAL_DECAY="${RADIAL_DECAY:-1.0}"
RADIAL_BLOCK_SIZE="${RADIAL_BLOCK_SIZE:-64}"
RADIAL_FIRST_LAYERS_FP="${RADIAL_FIRST_LAYERS_FP:-4}"
RADIAL_MODEL_TYPE="${RADIAL_MODEL_TYPE:-hunyuan}"

OUTPUT_BASE="${OUTPUT_ROOT:-${PROJECT_ROOT}/sampling_dataset/radial/single_arm/output/multiview/ctrlworld}"
INPUT_BASE="${INPUT_ROOT:-${PROJECT_ROOT}/sampling_dataset/radial/single_arm/input/multiview/ctrlworld}"

EXTRA=""
if [ -n "${NUM_EPISODES:-}" ]; then EXTRA="${EXTRA} --num_episodes ${NUM_EPISODES}"; fi
if [ -n "${NUM_INFERENCE_STEPS:-}" ]; then EXTRA="${EXTRA} --num_inference_steps ${NUM_INFERENCE_STEPS}"; fi

echo "=========================================================="
echo "Ctrl-World + Radial | Single-Arm Multiview Inference"
echo "Subset: ${SUBSET}"
echo "  ckpt=${CKPT_PATH}"
echo "  Params: decay=${RADIAL_DECAY}, block_size=${RADIAL_BLOCK_SIZE}, first_fp=${RADIAL_FIRST_LAYERS_FP}, model_type=${RADIAL_MODEL_TYPE}"
echo "  output=${OUTPUT_BASE}"
echo "=========================================================="

if [ "${SKIP_PREPARE:-0}" != "1" ]; then
    echo ""
    echo "[Step 1] Preparing dataset (LeRoBot -> Ctrl-World format)..."
    python scripts/prepare_ctrlworld_single_arm_multiview.py \
        --subset "${SUBSET}" \
        --output-dir "${DATASET_EXAMPLE_DIR}" \
        --dataset-base "${PROJECT_ROOT}/datasets/single_arm/multiview"
fi

run_subset() {
    local subset="$1"
    echo ""
    echo "[Step 2] Running Radial inference on ${subset}..."
    python scripts/infer_single_arm_multiview_ctrlworld.py \
        --svd_model_path "${SVD_MODEL_PATH}" \
        --clip_model_path "${CLIP_MODEL_PATH}" \
        --ckpt_path "${CKPT_PATH}" \
        --subset "${subset}" \
        --dataset_example_dir "${DATASET_EXAMPLE_DIR}" \
        --dataset_meta_info_path "${DATASET_META_INFO}" \
        --save_dir "${OUTPUT_BASE}/${subset}" \
        --input_save_dir "${INPUT_BASE}/${subset}" \
        --use_radial \
        --radial_decay_factor "${RADIAL_DECAY}" \
        --radial_block_size "${RADIAL_BLOCK_SIZE}" \
        --radial_first_layers_fp "${RADIAL_FIRST_LAYERS_FP}" \
        --radial_model_type "${RADIAL_MODEL_TYPE}" ${EXTRA}
}

if [ "${SUBSET}" = "both" ] || [ "${SUBSET}" = "makovian" ]; then
    run_subset makovian
fi
if [ "${SUBSET}" = "both" ] || [ "${SUBSET}" = "non_makovian" ]; then
    run_subset non_makovian
fi

echo ""
echo "=========================================================="
echo "Done! Results saved to: ${OUTPUT_BASE}"
echo "=========================================================="
