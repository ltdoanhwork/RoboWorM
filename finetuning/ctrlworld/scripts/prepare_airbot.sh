#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/models/DreamDojo/.venv/bin/activate" ]]; then
  source "${REPO_ROOT}/models/DreamDojo/.venv/bin/activate"
fi

DATASET_ROOT="${DATASET_ROOT:-datasets/humanoid/multiview/non_makovian}"
PROCESSED_ROOT="${PROCESSED_ROOT:-datasets/processed/ctrl_world_robo_multiview}"
SVD_MODEL_PATH="${SVD_MODEL_PATH:-checkpoints/base/stabilityai/stable-video-diffusion-img2vid}"
MAX_EPISODES_PER_DATASET="${MAX_EPISODES_PER_DATASET:-}"
LIMIT_RECORDS="${LIMIT_RECORDS:-}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-64}"
FPS="${FPS:-5}"
DOWNSAMPLE_STRIDE="${DOWNSAMPLE_STRIDE:-6}"

ARGS=(
  --dataset-root "${DATASET_ROOT}"
  --processed-root "${PROCESSED_ROOT}"
  --manifest-path "${PROCESSED_ROOT}/manifest.json"
  --include-pattern "Airbot_MMK2*,AIRBOT_MMK2*"
  --svd-model-path "${SVD_MODEL_PATH}"
  --fps "${FPS}"
  --downsample-stride "${DOWNSAMPLE_STRIDE}"
  --action-key eef_sim_pose_action
  --state-key eef_sim_pose_state
  --action-dim 12
  --n-views 4
  --encode-batch-size "${ENCODE_BATCH_SIZE}"
)

if [[ -n "${MAX_EPISODES_PER_DATASET}" ]]; then
  ARGS+=(--max-episodes-per-dataset "${MAX_EPISODES_PER_DATASET}")
fi
if [[ -n "${LIMIT_RECORDS}" ]]; then
  ARGS+=(--limit-records "${LIMIT_RECORDS}")
fi
if [[ "${WRITE_RESIZED_VIDEOS:-0}" == "1" ]]; then
  ARGS+=(--write-resized-videos)
fi

python -m finetuning.data.prepare_latents "${ARGS[@]}" "$@"
