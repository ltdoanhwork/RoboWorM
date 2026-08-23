#!/usr/bin/env bash
# Helper: run ONE sparse-attention trial for bimanual singleview dreamdojo into a
# scratch dir and print the timing + metrics line. Used only for tuning sweeps.
# Mirrors _tune_dreamdojo_humanoid_sv_trial.sh (see dreamdojo-humanoid-singleview-
# sparse-attention-insights rule) but points at the bimanual ckpt/dataset/script.
#
# Usage: bash scripts/_tune_dreamdojo_bimanual_sv_trial.sh <tag> <gpu> <extra args...>
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TAG="$1"; shift
GPU="$1"; shift

SCRATCH="${SCRATCH:-/tmp/dd_tune_bimanual}"
NUM_EPISODES="${NUM_EPISODES:-1}"
SPLIT="${SPLIT:-makovian}"
TASK="${TASK:-close_cardboard_box}"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"

rm -rf "${SCRATCH}/${TAG}"
mkdir -p "${SCRATCH}/${TAG}" "${SCRATCH}/logs"

CUDA_VISIBLE_DEVICES="${GPU}" python "${ROOT}/scripts/infer_bimanual_singleview_single_segment.py" \
  --checkpoints-dir "${ROOT}/exp_logs/dreamdojo_logs/dreamdojo/exp_aloha_resume70k/aloha_bimanual_singleview_full/checkpoints" \
  --experiment dreamdojo_2b_480_640_aloha_resume70k_bimanual_sv \
  --dataset-path "${ROOT}/datasets/bimanual/singleview/${SPLIT}/${TASK}" \
  --save-dir "${SCRATCH}/${TAG}" \
  --chunk-size 12 \
  --save-fps 10 \
  --output-dir "${SCRATCH}/out_${TAG}" \
  --guidance 1.5 \
  --num-episodes "${NUM_EPISODES}" \
  --save-full \
  "$@" > "${SCRATCH}/logs/${TAG}.log" 2>&1

echo "=== ${TAG} ==="
grep -E "\[Timing\] Episode|PSNR:|SSIM:|LPIPS:" "${SCRATCH}/logs/${TAG}.log" || tail -30 "${SCRATCH}/logs/${TAG}.log"
