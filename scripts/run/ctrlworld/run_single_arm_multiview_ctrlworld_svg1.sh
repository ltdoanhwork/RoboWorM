#!/bin/bash
# Ctrl-World inference with SVG1 (stripe-sparse + MSE profiling) on
# single_arm/multiview. Mirrors run_single_arm_multiview_ctrlworld.sh (dense),
# swapping sampling_dataset/dense -> svg1 and adding SVG1 flags.
#
# SVG1 uses masked-SDPA BY DESIGN (bool mask), not because flex is missing.
# Env has flashinfer + flex; this is still the SVG1 path.
#
# Output: sampling_dataset/svg1/single_arm/{output,input}/multiview/ctrlworld/{subset}/
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run/ctrlworld/run_single_arm_multiview_ctrlworld_svg1.sh [makovian|non_makovian|both]
# Env: CKPT, NUM_EPISODES, NUM_INFERENCE_STEPS, OUTPUT_ROOT, INPUT_ROOT, SKIP_PREPARE,
#      SVG_{SPARSITY,NUM_SAMPLED_ROWS,FIRST_LAYERS_FP}

set -euo pipefail

SUBSET=${1:-both}

PROJECT_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
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

# SVG1 -- TUNED (see ctrlworld-single-arm-multiview-sparse-attention-insights).
SVG_SPARSITY="${SVG_SPARSITY:-0.25}"
SVG_NUM_SAMPLED_ROWS="${SVG_NUM_SAMPLED_ROWS:-32}"
SVG_FIRST_LAYERS_FP="${SVG_FIRST_LAYERS_FP:-2}"

OUTPUT_BASE="${OUTPUT_ROOT:-${PROJECT_ROOT}/sampling_dataset/svg1/single_arm/output/multiview/ctrlworld}"
INPUT_BASE="${INPUT_ROOT:-${PROJECT_ROOT}/sampling_dataset/svg1/single_arm/input/multiview/ctrlworld}"

EXTRA=""
if [ -n "${NUM_EPISODES:-}" ]; then EXTRA="${EXTRA} --num_episodes ${NUM_EPISODES}"; fi
if [ -n "${NUM_INFERENCE_STEPS:-}" ]; then EXTRA="${EXTRA} --num_inference_steps ${NUM_INFERENCE_STEPS}"; fi

echo "=========================================================="
echo "Ctrl-World + SVG1 | Single-Arm Multiview Inference"
echo "Subset: ${SUBSET}"
echo "  ckpt=${CKPT_PATH}"
echo "  Params: sparsity=${SVG_SPARSITY}, sampled_rows=${SVG_NUM_SAMPLED_ROWS}, first_fp=${SVG_FIRST_LAYERS_FP}"
echo "  output=${OUTPUT_BASE}"
echo "=========================================================="

if [ "${SKIP_PREPARE:-0}" != "1" ]; then
    echo ""
    echo "[Step 1] Preparing dataset (LeRoBot -> Ctrl-World format)..."
    python scripts/prepare/prepare_ctrlworld_single_arm_multiview.py \
        --subset "${SUBSET}" \
        --output-dir "${DATASET_EXAMPLE_DIR}" \
        --dataset-base "${PROJECT_ROOT}/datasets/single_arm/multiview"
fi

run_subset() {
    local subset="$1"
    echo ""
    echo "[Step 2] Running SVG1 inference on ${subset}..."
    python scripts/infer/infer_single_arm_multiview_ctrlworld.py \
        --svd_model_path "${SVD_MODEL_PATH}" \
        --clip_model_path "${CLIP_MODEL_PATH}" \
        --ckpt_path "${CKPT_PATH}" \
        --subset "${subset}" \
        --dataset_example_dir "${DATASET_EXAMPLE_DIR}" \
        --dataset_meta_info_path "${DATASET_META_INFO}" \
        --save_dir "${OUTPUT_BASE}/${subset}" \
        --input_save_dir "${INPUT_BASE}/${subset}" \
        --use_svg \
        --svg_pattern SVG1 \
        --svg_sparsity "${SVG_SPARSITY}" \
        --svg_num_sampled_rows "${SVG_NUM_SAMPLED_ROWS}" \
        --svg_first_layers_fp "${SVG_FIRST_LAYERS_FP}" ${EXTRA}
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
