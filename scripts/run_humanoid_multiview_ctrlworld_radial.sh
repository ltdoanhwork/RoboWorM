#!/bin/bash
# Ctrl-World inference with Radial Attention on humanoid/multiview. Mirrors
# run_humanoid_multiview_ctrlworld.sh (dense), swapping sampling_dataset/dense ->
# radial and adding Radial flags. CONFIG selects 3-view strip vs 4-view grid.
#
# Output: sampling_dataset/radial/humanoid/{output,input}/multiview/ctrlworld[...]/{category}/
#
# Env overrides: CKPT, DATASET_NAME, DATASET_DIR, DOWN_SAMPLE, SPLIT, CONFIG,
#   NUM_EPISODES, NUM_INFERENCE_STEPS, GUIDANCE, OUTPUT_ROOT, INPUT_ROOT,
#   NUM_SHARDS/SHARD_ID, RADIAL_{DECAY,BLOCK_SIZE,FIRST_LAYERS_FP,MODEL_TYPE}

set -euo pipefail

ROOT="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics"
cd "${ROOT}"

source "${ROOT}/models/Ctrl-World/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/Ctrl-World:${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

SVD_MODEL_PATH="${ROOT}/checkpoints/stabilityai/stable-video-diffusion-img2vid"
CLIP_MODEL_PATH="${ROOT}/checkpoints/openai/clip-vit-base-patch32"
CKPT="${CKPT:-${ROOT}/exp_logs/ctrlworld_logs/humanoid_multiview/checkpoint-200000.pt}"

DATASET_NAME="${DATASET_NAME:-humanoid_multiview}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/models/Ctrl-World/dataset_example/${DATASET_NAME}}"
META_INFO="${ROOT}/models/Ctrl-World/dataset_meta_info"
DOWN_SAMPLE="${DOWN_SAMPLE:-6}"
SPLIT="${SPLIT:-all}"
CONFIG="${CONFIG:-humanoid}"

# Radial parameters -- TUNED per model (see the 4view-grid sparse-attention rule).
RADIAL_DECAY="${RADIAL_DECAY:-1.0}"
RADIAL_BLOCK_SIZE="${RADIAL_BLOCK_SIZE:-64}"
RADIAL_FIRST_LAYERS_FP="${RADIAL_FIRST_LAYERS_FP:-4}"
RADIAL_MODEL_TYPE="${RADIAL_MODEL_TYPE:-hunyuan}"

OUTPUT_BASE="${OUTPUT_ROOT:-${ROOT}/sampling_dataset/radial/humanoid/output/multiview/ctrlworld}"
INPUT_BASE="${INPUT_ROOT:-${ROOT}/sampling_dataset/radial/humanoid/input/multiview/ctrlworld}"

EXTRA=""
if [ -n "${NUM_EPISODES:-}" ]; then EXTRA="${EXTRA} --num_episodes ${NUM_EPISODES}"; fi
if [ -n "${NUM_INFERENCE_STEPS:-}" ]; then EXTRA="${EXTRA} --num_inference_steps ${NUM_INFERENCE_STEPS}"; fi
if [ -n "${GUIDANCE:-}" ]; then EXTRA="${EXTRA} --guidance_scale ${GUIDANCE}"; fi
if [ -n "${NUM_SHARDS:-}" ]; then EXTRA="${EXTRA} --num_shards ${NUM_SHARDS} --shard_id ${SHARD_ID:-0}"; fi

echo "=========================================================="
echo "Ctrl-World + Radial | Humanoid Multiview Inference"
echo "  ckpt=${CKPT}  config=${CONFIG}"
echo "  Params: decay=${RADIAL_DECAY}, block_size=${RADIAL_BLOCK_SIZE}, first_fp=${RADIAL_FIRST_LAYERS_FP}, model_type=${RADIAL_MODEL_TYPE}"
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
    --use_radial \
    --radial_decay_factor "${RADIAL_DECAY}" \
    --radial_block_size "${RADIAL_BLOCK_SIZE}" \
    --radial_first_layers_fp "${RADIAL_FIRST_LAYERS_FP}" \
    --radial_model_type "${RADIAL_MODEL_TYPE}" ${EXTRA}

echo ""
echo "Done! Results saved to: ${OUTPUT_BASE}"
