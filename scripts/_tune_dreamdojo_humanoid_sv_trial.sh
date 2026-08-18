#!/usr/bin/env bash
# Helper: run ONE sparse-attention trial for humanoid singleview dreamdojo into a
# scratch dir and print the timing + metrics line. Used only for tuning sweeps.
#
# Usage: bash scripts/_tune_dreamdojo_humanoid_sv_trial.sh <tag> <gpu> <extra args...>
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TAG="$1"; shift
GPU="$1"; shift

SCRATCH="${SCRATCH:-/tmp/dd_tune}"
NUM_EPISODES="${NUM_EPISODES:-1}"
SPLIT="${SPLIT:-makovian}"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"

rm -rf "${SCRATCH}/${TAG}"
mkdir -p "${SCRATCH}/${TAG}" "${SCRATCH}/logs"

CUDA_VISIBLE_DEVICES="${GPU}" python "${ROOT}/scripts/infer_humanoid_singleview_full_episode.py" \
  --checkpoints-dir "${ROOT}/checkpoints/dreamdojo/2B_GR1_post-train/iter_000050000" \
  --experiment dreamdojo_2b_480_640_gr1 \
  --dataset-path "${ROOT}/datasets/humanoid/singleview/${SPLIT}" \
  --save-dir "${SCRATCH}/${TAG}" \
  --num-frames 49 \
  --output-dir "${SCRATCH}/out_${TAG}" \
  --guidance 1.5 \
  --num-episodes "${NUM_EPISODES}" \
  --save-full \
  "$@" > "${SCRATCH}/logs/${TAG}.log" 2>&1

echo "=== ${TAG} ==="
grep -E "\[Timing\] Generated|PSNR:|SSIM:|LPIPS:" "${SCRATCH}/logs/${TAG}.log" || tail -20 "${SCRATCH}/logs/${TAG}.log"
