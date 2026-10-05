#!/bin/bash
# Ctrl-World inference with SVG2 (SAP / K-Means) on single_arm/multiview.
# Mirrors run_single_arm_multiview_ctrlworld.sh (dense), swapping
# sampling_dataset/dense -> svg2 and adding SVG2 flags.
#
# NOTE: env DOES have flashinfer (0.3.1) + triton (3.3.1) — verified 2026-08-17.
# SVG2 genuinely runs the flashinfer block-sparse path (log: "Installed SVG
# FlashInfer gated monkey patch"), NOT an SDPA fallback.
#
# Output: sampling_dataset/svg2/single_arm/{output,input}/multiview/ctrlworld/{subset}/
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_single_arm_multiview_ctrlworld_svg2.sh [makovian|non_makovian|both]
# Env: VENV, CKPT, NUM_EPISODES, NUM_INFERENCE_STEPS, OUTPUT_ROOT, INPUT_ROOT, SKIP_PREPARE,
#      SVG_{NUM_Q_CENTROIDS,NUM_K_CENTROIDS,TOP_P_KMEANS,FIRST_LAYERS_FP}

set -euo pipefail

SUBSET=${1:-both}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

VENV="${VENV:-${PROJECT_ROOT}/models/Ctrl-World/.venv}"
source "${VENV}/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}/models/Ctrl-World:${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

SVD_MODEL_PATH="./checkpoints/base/stabilityai/stable-video-diffusion-img2vid"
CLIP_MODEL_PATH="./checkpoints/base/openai/clip-vit-base-patch32"
CKPT_PATH="${CKPT:-./checkpoints/base/ctrl_world/Ctrl-World/checkpoint-10000.pt}"
DATASET_META_INFO="./models/Ctrl-World/dataset_meta_info"
DATASET_EXAMPLE_DIR="./models/Ctrl-World/dataset_example"

# SVG2 -- TUNED (see ctrlworld-single-arm-multiview-sparse-attention-insights).
SVG_NUM_Q_CENTROIDS="${SVG_NUM_Q_CENTROIDS:-32}"
SVG_NUM_K_CENTROIDS="${SVG_NUM_K_CENTROIDS:-64}"
SVG_TOP_P_KMEANS="${SVG_TOP_P_KMEANS:-0.7}"
SVG_FIRST_LAYERS_FP="${SVG_FIRST_LAYERS_FP:-2}"

OUTPUT_BASE="${OUTPUT_ROOT:-${PROJECT_ROOT}/sampling_dataset/svg2/single_arm/output/multiview/ctrlworld}"
INPUT_BASE="${INPUT_ROOT:-${PROJECT_ROOT}/sampling_dataset/svg2/single_arm/input/multiview/ctrlworld}"

EXTRA=""
if [ -n "${NUM_EPISODES:-}" ]; then EXTRA="${EXTRA} --num_episodes ${NUM_EPISODES}"; fi
if [ -n "${NUM_INFERENCE_STEPS:-}" ]; then EXTRA="${EXTRA} --num_inference_steps ${NUM_INFERENCE_STEPS}"; fi

echo "=========================================================="
echo "Ctrl-World + SVG2 (SAP) | Single-Arm Multiview Inference"
echo "Subset: ${SUBSET}"
echo "  ckpt=${CKPT_PATH}  venv=${VENV}"
echo "  Params: q_centroids=${SVG_NUM_Q_CENTROIDS}, k_centroids=${SVG_NUM_K_CENTROIDS}, top_p=${SVG_TOP_P_KMEANS}"
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
    echo "[Step 2] Running SVG2 inference on ${subset}..."
    python scripts/infer_single_arm_multiview_ctrlworld.py \
        --svd_model_path "${SVD_MODEL_PATH}" \
        --clip_model_path "${CLIP_MODEL_PATH}" \
        --ckpt_path "${CKPT_PATH}" \
        --subset "${subset}" \
        --dataset_example_dir "${DATASET_EXAMPLE_DIR}" \
        --dataset_meta_info_path "${DATASET_META_INFO}" \
        --save_dir "${OUTPUT_BASE}/${subset}" \
        --input_save_dir "${INPUT_BASE}/${subset}" \
        --use_svg \
        --svg_pattern SVG2 \
        --svg_num_q_centroids "${SVG_NUM_Q_CENTROIDS}" \
        --svg_num_k_centroids "${SVG_NUM_K_CENTROIDS}" \
        --svg_top_p_kmeans "${SVG_TOP_P_KMEANS}" \
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
