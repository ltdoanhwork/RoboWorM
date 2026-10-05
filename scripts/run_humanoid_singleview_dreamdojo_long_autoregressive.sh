#!/usr/bin/env bash
# Long autoregressive inference for humanoid singleview DreamDojo.
#
# Uses generate_vid2world_long with chunk_overlap for smooth long-horizon generation.
# Pure autoregressive: only the first GT frame is used, no re-anchoring.
#
# Key parameters:
#   CHUNK_OVERLAP    - frames overlapping between chunks (default 4, higher = smoother but slower)
#   NUM_COND_FRAMES  - num_latent_conditional_frames (1=image2world, 2=video2world 5-frame context)
#
# Usage:
#   bash scripts/run_humanoid_singleview_dreamdojo_long_autoregressive.sh
#   CHUNK_OVERLAP=8 bash scripts/run_humanoid_singleview_dreamdojo_long_autoregressive.sh
#   NUM_COND_FRAMES=2 bash scripts/run_humanoid_singleview_dreamdojo_long_autoregressive.sh
#   NUM_EPISODES=5 CUDA_VISIBLE_DEVICES=0 bash scripts/run_humanoid_singleview_dreamdojo_long_autoregressive.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/checkpoints/base/dreamdojo/2B_GR1_post-train/iter_000050000}"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_gr1}"
NUM_FRAMES="${NUM_FRAMES:-49}"
CHUNK_OVERLAP="${CHUNK_OVERLAP:-4}"
NUM_COND_FRAMES="${NUM_COND_FRAMES:-1}"
GUIDANCE="${GUIDANCE:-0}"
RESOLUTION="${RESOLUTION:-480,640}"
SAVE_FPS="${SAVE_FPS:-10}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/tmp_2}"
NUM_EPISODES="${NUM_EPISODES:-}"

ep_flag=()
if [[ -n "${NUM_EPISODES}" ]]; then
  ep_flag=(--num-episodes "${NUM_EPISODES}")
fi

for split in makovian non_makovian; do
  dataset_dir="${ROOT}/datasets/humanoid/singleview/${split}"
  if [ ! -d "${dataset_dir}/data" ]; then
    echo "Skipping ${split}: no data directory found at ${dataset_dir}/data" >&2
    continue
  fi
  save_base="${SAVE_ROOT}/${split}"
  mkdir -p "${save_base}"
  echo "=== Long autoregressive (chunk_overlap=${CHUNK_OVERLAP}, cond_frames=${NUM_COND_FRAMES}): ${split} ==="
  python "${ROOT}/scripts/infer_humanoid_singleview_long_autoregressive.py" \
    --checkpoints-dir "${CHECKPOINTS_DIR}" \
    --experiment "${EXPERIMENT}" \
    --dataset-path "${dataset_dir}" \
    --save-dir "${save_base}" \
    --num-frames "${NUM_FRAMES}" \
    --chunk-overlap "${CHUNK_OVERLAP}" \
    --num-latent-conditional-frames "${NUM_COND_FRAMES}" \
    --guidance "${GUIDANCE}" \
    --resolution "${RESOLUTION}" \
    --save-fps "${SAVE_FPS}" \
    --output-dir "${ROOT}/output/dreamdojo_long_ar_humanoid_sv_${split}" \
    "${ep_flag[@]}"
done
