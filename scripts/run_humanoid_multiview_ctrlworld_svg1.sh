#!/bin/bash
# Ctrl-World inference with SVG1 (stripe-sparse + MSE profiling) sparse attention
# on humanoid/multiview. Mirrors run_humanoid_multiview_ctrlworld.sh (dense),
# swapping sampling_dataset/dense -> svg1 and adding SVG1 flags. CONFIG selects
# 3-view strip vs 4-view grid.
#
# Output: sampling_dataset/svg1/humanoid/{output,input}/multiview/ctrlworld[...]/{category}/
#
# Env overrides: CKPT, DATASET_NAME, DATASET_DIR, DOWN_SAMPLE, SPLIT, CONFIG,
#   NUM_EPISODES, NUM_INFERENCE_STEPS, GUIDANCE, OUTPUT_ROOT, INPUT_ROOT,
#   NUM_SHARDS/SHARD_ID, SVG_{SPARSITY,NUM_SAMPLED_ROWS,FIRST_LAYERS_FP}

set -euo pipefail

ROOT="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics"
cd "${ROOT}"

source "${ROOT}/models/Ctrl-World/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/Ctrl-World:${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

SVD_MODEL_PATH="${ROOT}/checkpoints/base/stabilityai/stable-video-diffusion-img2vid"
CLIP_MODEL_PATH="${ROOT}/checkpoints/base/openai/clip-vit-base-patch32"
CKPT="${CKPT:-${ROOT}/checkpoints/finetuned/ctrlworld/humanoid_multiview/checkpoint-200000.pt}"

DATASET_NAME="${DATASET_NAME:-humanoid_multiview}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/models/Ctrl-World/dataset_example/${DATASET_NAME}}"
META_INFO="${ROOT}/models/Ctrl-World/dataset_meta_info"
DOWN_SAMPLE="${DOWN_SAMPLE:-6}"
SPLIT="${SPLIT:-all}"
CONFIG="${CONFIG:-humanoid}"

# SVG1 parameters -- TUNED per model (see the 4view-grid sparse-attention rule).
SVG_SPARSITY="${SVG_SPARSITY:-0.15}"
SVG_NUM_SAMPLED_ROWS="${SVG_NUM_SAMPLED_ROWS:-32}"
SVG_FIRST_LAYERS_FP="${SVG_FIRST_LAYERS_FP:-2}"

OUTPUT_BASE="${OUTPUT_ROOT:-${ROOT}/sampling_dataset/svg1/humanoid/output/multiview/ctrlworld}"
INPUT_BASE="${INPUT_ROOT:-${ROOT}/sampling_dataset/svg1/humanoid/input/multiview/ctrlworld}"

EXTRA=""
if [ -n "${NUM_EPISODES:-}" ]; then EXTRA="${EXTRA} --num_episodes ${NUM_EPISODES}"; fi
# Balanced subset for the timing pass: N episodes of EACH category. Plain
# NUM_EPISODES slices a stem-sorted list where every makovian_* precedes every
# non_makovian_*, so a small value silently yields makovian only.
if [ -n "${NUM_EPISODES_PER_CATEGORY:-}" ]; then EXTRA="${EXTRA} --num_episodes_per_category ${NUM_EPISODES_PER_CATEGORY}"; fi
if [ -n "${NUM_INFERENCE_STEPS:-}" ]; then EXTRA="${EXTRA} --num_inference_steps ${NUM_INFERENCE_STEPS}"; fi
if [ -n "${GUIDANCE:-}" ]; then EXTRA="${EXTRA} --guidance_scale ${GUIDANCE}"; fi
if [ -n "${NUM_SHARDS:-}" ]; then EXTRA="${EXTRA} --num_shards ${NUM_SHARDS} --shard_id ${SHARD_ID:-0}"; fi

echo "=========================================================="
echo "Ctrl-World + SVG1 | Humanoid Multiview Inference"
echo "  ckpt=${CKPT}  config=${CONFIG}"
echo "  Params: sparsity=${SVG_SPARSITY}, sampled_rows=${SVG_NUM_SAMPLED_ROWS}, first_fp=${SVG_FIRST_LAYERS_FP}"
echo "  output=${OUTPUT_BASE}"
echo "=========================================================="

python scripts/infer_bimanual_multiview_ctrlworld.py \
    --config "${CONFIG}" \
    --svd_model_path "${SVD_MODEL_PATH}" \
    --clip_model_path "${CLIP_MODEL_PATH}" \
    --ckpt_path "${CKPT}" \
    --dataset_dir "${DATASET_DIR}" \
    --dataset_meta_info_path "${META_INFO}" \
    --dataset_name "${DATASET_NAME}" \
    --save_dir "${OUTPUT_BASE}" \
    --input_save_dir "${INPUT_BASE}" \
    --infer_down_sample "${DOWN_SAMPLE}" \
    --split "${SPLIT}" \
    --use_svg \
    --svg_pattern SVG1 \
    --svg_sparsity "${SVG_SPARSITY}" \
    --svg_num_sampled_rows "${SVG_NUM_SAMPLED_ROWS}" \
    --svg_first_layers_fp "${SVG_FIRST_LAYERS_FP}" ${EXTRA}

echo ""
echo "Done! Results saved to: ${OUTPUT_BASE}"
