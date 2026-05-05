#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/models/DreamDojo/.venv/bin/activate" ]]; then
  source "${REPO_ROOT}/models/DreamDojo/.venv/bin/activate"
fi

PROCESSED_ROOT="${PROCESSED_ROOT:-datasets/processed/ctrl_world_robo_multiview}"
SVD_MODEL_PATH="${SVD_MODEL_PATH:-checkpoints/stabilityai/stable-video-diffusion-img2vid}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-checkpoints/openai/clip-vit-base-patch32}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROCESSED_ROOT}/checkpoints}"
CKPT="${CKPT:-${OUTPUT_DIR}/latest.pt}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"

python -m finetuning.eval_replay \
  --processed-root "${PROCESSED_ROOT}" \
  --manifest-path "${PROCESSED_ROOT}/manifest.json" \
  --svd-model-path "${SVD_MODEL_PATH}" \
  --clip-model-path "${CLIP_MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --ckpt "${CKPT}" \
  --num-samples "${NUM_SAMPLES}" \
  "$@"
