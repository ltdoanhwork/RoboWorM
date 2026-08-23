#!/bin/bash
# Run finetuned Ctrl-World inference on bimanual/multiview (replay GT 14-D qpos).
#
# Prereq: dataset prepared via prepare_ctrlworld_bimanual_multiview.py and a
# finetuned checkpoint from run_bimanual_multiview_ctrlworld_train.sh.
#
# Usage:
#   CKPT=/path/to/checkpoint-XXXXX.pt bash scripts/run_bimanual_multiview_ctrlworld.sh
#
# Env overrides:
#   CKPT (finetuned checkpoint .pt; default = DROID base as a smoke fallback)
#   DATASET_NAME (default bimanual_multiview), DATASET_DIR
#   NUM_EPISODES, SPLIT (val|train|all), NUM_INFERENCE_STEPS
#   DOWN_SAMPLE (must match training; default 1 = 30fps), CUDA_VISIBLE_DEVICES
#   CONFIG (bimanual = 3-view strip [default] | bimanual_grid = 4-view 2x2 grid)
#   OUTPUT_ROOT / INPUT_ROOT (override output/input trees, e.g. for a variant model)
#   NUM_SHARDS / SHARD_ID (split episodes across parallel GPU processes)

set -e

ROOT="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics"
cd "${ROOT}"

source "${ROOT}/models/Ctrl-World/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/Ctrl-World:${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

SVD_MODEL_PATH="${ROOT}/checkpoints/stabilityai/stable-video-diffusion-img2vid"
CLIP_MODEL_PATH="${ROOT}/checkpoints/openai/clip-vit-base-patch32"
CKPT="${CKPT:-${ROOT}/checkpoints/ctrl_world/Ctrl-World/checkpoint-10000.pt}"

DATASET_NAME="${DATASET_NAME:-bimanual_multiview}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/models/Ctrl-World/dataset_example/${DATASET_NAME}}"
META_INFO="${ROOT}/models/Ctrl-World/dataset_meta_info"
DOWN_SAMPLE="${DOWN_SAMPLE:-1}"

CONFIG="${CONFIG:-bimanual}"
OUTPUT_BASE="${OUTPUT_ROOT:-${ROOT}/sampling_dataset/dense/bimanual/output/multiview/ctrlworld}"
INPUT_BASE="${INPUT_ROOT:-${ROOT}/sampling_dataset/dense/bimanual/input/multiview/ctrlworld}"

EXTRA=""
if [ -n "${NUM_EPISODES:-}" ]; then EXTRA="${EXTRA} --num_episodes ${NUM_EPISODES}"; fi
if [ -n "${NUM_INFERENCE_STEPS:-}" ]; then EXTRA="${EXTRA} --num_inference_steps ${NUM_INFERENCE_STEPS}"; fi
if [ -n "${GUIDANCE:-}" ]; then EXTRA="${EXTRA} --guidance_scale ${GUIDANCE}"; fi
if [ -n "${NUM_SHARDS:-}" ]; then EXTRA="${EXTRA} --num_shards ${NUM_SHARDS} --shard_id ${SHARD_ID:-0}"; fi

echo "=========================================="
echo "Ctrl-World Bimanual Multiview Inference"
echo "  ckpt=${CKPT}"
echo "  config=${CONFIG}"
echo "  dataset=${DATASET_DIR}  split=${SPLIT:-all}  down_sample=${DOWN_SAMPLE}"
echo "  output=${OUTPUT_BASE}"
echo "=========================================="

python scripts/infer_bimanual_multiview_ctrlworld.py \
    --svd_model_path "${SVD_MODEL_PATH}" \
    --clip_model_path "${CLIP_MODEL_PATH}" \
    --ckpt_path "${CKPT}" \
    --dataset_dir "${DATASET_DIR}" \
    --dataset_meta_info_path "${META_INFO}" \
    --dataset_name "${DATASET_NAME}" \
    --save_dir "${OUTPUT_BASE}" \
    --input_save_dir "${INPUT_BASE}" \
    --infer_down_sample "${DOWN_SAMPLE}" \
    --config "${CONFIG}" \
    --split "${SPLIT:-all}" ${EXTRA}

echo ""
echo "Done! Results saved to: ${OUTPUT_BASE}"
