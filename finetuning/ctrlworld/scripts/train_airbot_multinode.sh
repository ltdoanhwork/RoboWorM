#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/models/DreamDojo/.venv/bin/activate" ]]; then
  source "${REPO_ROOT}/models/DreamDojo/.venv/bin/activate"
fi

PROCESSED_ROOT="${PROCESSED_ROOT:-datasets/processed/ctrl_world_robo_multiview}"
SVD_MODEL_PATH="${SVD_MODEL_PATH:-checkpoints/base/stabilityai/stable-video-diffusion-img2vid}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-checkpoints/base/openai/clip-vit-base-patch32}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROCESSED_ROOT}/checkpoints}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"

TRAIN_ARGS=(
  -m finetuning.train_ctrl_world
  --processed-root "${PROCESSED_ROOT}"
  --manifest-path "${PROCESSED_ROOT}/manifest.json"
  --svd-model-path "${SVD_MODEL_PATH}"
  --clip-model-path "${CLIP_MODEL_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --action-key eef_sim_pose_action
  --state-key eef_sim_pose_state
  --action-dim 12
  --n-views 4
  --fps 5
)

if [[ "${NNODES}" -gt 1 || "${GPUS_PER_NODE}" -gt 1 ]]; then
  accelerate launch \
    --num_processes "$((NNODES * GPUS_PER_NODE))" \
    --num_machines "${NNODES}" \
    --machine_rank "${NODE_RANK}" \
    --main_process_ip "${MASTER_ADDR}" \
    --main_process_port "${MASTER_PORT}" \
    "${TRAIN_ARGS[@]}" "$@"
else
  python "${TRAIN_ARGS[@]}" "$@"
fi
