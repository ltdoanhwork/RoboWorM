#!/bin/bash
# Ctrl-World inference with PISA sparse attention on humanoid/multiview.
#
# Mirrors run_humanoid_multiview_ctrlworld.sh (dense) exactly in logic / paths /
# checkpoint, only swapping sampling_dataset/dense -> pisa and adding PISA flags
# (acceleration-benchmark + sampling-dataset-layout rules). CONFIG (humanoid |
# humanoid_grid) selects the 3-view strip vs 4-view 2x2 grid; the 4-view grid
# wrapper passes CONFIG=humanoid_grid.
#
# Output: sampling_dataset/pisa/humanoid/{output,input}/multiview/ctrlworld[...]/{category}/
#
# Env overrides: CKPT, DATASET_NAME, DATASET_DIR, DOWN_SAMPLE, SPLIT, CONFIG,
#   NUM_EPISODES, NUM_INFERENCE_STEPS, GUIDANCE, OUTPUT_ROOT, INPUT_ROOT,
#   NUM_SHARDS/SHARD_ID, PISA_{DENSITY,BLOCK_SIZE,START_LAYER}

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

# PISA parameters -- TUNED per model (see the 4view-grid sparse-attention rule).
PISA_DENSITY="${PISA_DENSITY:-0.2}"
PISA_BLOCK_SIZE="${PISA_BLOCK_SIZE:-64}"
PISA_START_LAYER="${PISA_START_LAYER:-2}"

OUTPUT_BASE="${OUTPUT_ROOT:-${ROOT}/sampling_dataset/pisa/humanoid/output/multiview/ctrlworld}"
INPUT_BASE="${INPUT_ROOT:-${ROOT}/sampling_dataset/pisa/humanoid/input/multiview/ctrlworld}"

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
echo "Ctrl-World + PISA | Humanoid Multiview Inference"
echo "  ckpt=${CKPT}  config=${CONFIG}"
echo "  Params: density=${PISA_DENSITY}, block_size=${PISA_BLOCK_SIZE}, start_layer=${PISA_START_LAYER}"
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
    --use_pisa \
    --pisa_density "${PISA_DENSITY}" \
    --pisa_block_size "${PISA_BLOCK_SIZE}" \
    --pisa_start_layer_idx "${PISA_START_LAYER}" ${EXTRA}

echo ""
echo "Done! Results saved to: ${OUTPUT_BASE}"
