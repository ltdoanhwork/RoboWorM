#!/bin/bash
# Run finetuned Ctrl-World inference on humanoid/multiview (replay GT 36-D state).
#
# Reuses infer_bimanual_multiview_ctrlworld.py with --config humanoid (3 views,
# dim-agnostic state replay). Prereq: dataset prepared via
# prepare_ctrlworld_humanoid_multiview.py + a finetuned checkpoint.
#
# Usage:
#   CKPT=/path/to/checkpoint-XXXXX.pt bash scripts/run/ctrlworld/run_humanoid_multiview_ctrlworld.sh
#
# Env overrides:
#   CONFIG (humanoid | humanoid_grid, default humanoid),
#   CKPT, DATASET_NAME (default humanoid_multiview), DATASET_DIR, NUM_EPISODES,
#   SPLIT (val|train|all), NUM_INFERENCE_STEPS, GUIDANCE, DOWN_SAMPLE (default 6),
#   NUM_SHARDS / SHARD_ID (split episodes across parallel GPU processes),
#   OUTPUT_ROOT / INPUT_ROOT, CUDA_VISIBLE_DEVICES

set -e

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "${ROOT}"

source "${ROOT}/models/Ctrl-World/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/Ctrl-World:${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

SVD_MODEL_PATH="${ROOT}/checkpoints/base/stabilityai/stable-video-diffusion-img2vid"
CLIP_MODEL_PATH="${ROOT}/checkpoints/base/openai/clip-vit-base-patch32"
CKPT="${CKPT:-${ROOT}/checkpoints/base/ctrl_world/Ctrl-World/checkpoint-10000.pt}"

DATASET_NAME="${DATASET_NAME:-humanoid_multiview}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/models/Ctrl-World/dataset_example/${DATASET_NAME}}"
META_INFO="${ROOT}/models/Ctrl-World/dataset_meta_info"
DOWN_SAMPLE="${DOWN_SAMPLE:-6}"

OUTPUT_BASE="${OUTPUT_ROOT:-${ROOT}/sampling_dataset/dense/humanoid/output/multiview/ctrlworld}"
INPUT_BASE="${INPUT_ROOT:-${ROOT}/sampling_dataset/dense/humanoid/input/multiview/ctrlworld}"

EXTRA=""
if [ -n "${NUM_EPISODES:-}" ]; then EXTRA="${EXTRA} --num_episodes ${NUM_EPISODES}"; fi
# Balanced subset for the timing pass: N episodes of EACH category. Plain
# NUM_EPISODES slices a stem-sorted list where every makovian_* precedes every
# non_makovian_*, so a small value silently yields makovian only.
if [ -n "${NUM_EPISODES_PER_CATEGORY:-}" ]; then EXTRA="${EXTRA} --num_episodes_per_category ${NUM_EPISODES_PER_CATEGORY}"; fi
if [ -n "${NUM_INFERENCE_STEPS:-}" ]; then EXTRA="${EXTRA} --num_inference_steps ${NUM_INFERENCE_STEPS}"; fi
if [ -n "${GUIDANCE:-}" ]; then EXTRA="${EXTRA} --guidance_scale ${GUIDANCE}"; fi
if [ -n "${NUM_SHARDS:-}" ]; then EXTRA="${EXTRA} --num_shards ${NUM_SHARDS} --shard_id ${SHARD_ID:-0}"; fi

echo "=========================================="
echo "Ctrl-World Humanoid Multiview Inference"
echo "  config=${CONFIG:-humanoid}"
echo "  ckpt=${CKPT}"
echo "  dataset=${DATASET_DIR}  split=${SPLIT:-all}  down_sample=${DOWN_SAMPLE}"
echo "  output=${OUTPUT_BASE}"
echo "=========================================="

python scripts/infer/infer_bimanual_multiview_ctrlworld.py \
    --config "${CONFIG:-humanoid}" \
    --svd_model_path "${SVD_MODEL_PATH}" \
    --clip_model_path "${CLIP_MODEL_PATH}" \
    --ckpt_path "${CKPT}" \
    --dataset_dir "${DATASET_DIR}" \
    --dataset_meta_info_path "${META_INFO}" \
    --dataset_name "${DATASET_NAME}" \
    --save_dir "${OUTPUT_BASE}" \
    --input_save_dir "${INPUT_BASE}" \
    --infer_down_sample "${DOWN_SAMPLE}" \
    --split "${SPLIT:-all}" ${EXTRA}

echo ""
echo "Done! Results saved to: ${OUTPUT_BASE}"
