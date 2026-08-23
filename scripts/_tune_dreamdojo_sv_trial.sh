#!/usr/bin/env bash
# Helper: run ONE sparse-attention trial for DreamDojo singleview into a scratch dir.
# Used only for tuning sweeps — NEVER write into sampling_dataset/.
#
# Usage:
#   bash scripts/_tune_dreamdojo_sv_trial.sh <embodiment> <tag> <gpu> [extra args...]
#
# embodiment: single_arm | bimanual
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EMB="$1"; shift
TAG="$1"; shift
GPU="$1"; shift

SCRATCH="${SCRATCH:-/tmp/dd_tune_${EMB}}"
NUM_EPISODES="${NUM_EPISODES:-1}"
SPLIT="${SPLIT:-makovian}"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"

rm -rf "${SCRATCH}/${TAG}"
mkdir -p "${SCRATCH}/${TAG}" "${SCRATCH}/logs"

case "${EMB}" in
  single_arm)
    CKPT="${ROOT}/exp_logs/dreamdojo_logs/dreamdojo/exp_single_arm_sv/single_arm_singleview/checkpoints"
    EXP="dreamdojo_2b_480_640_single_arm_sv"
    DATASET="${ROOT}/datasets/single_arm/singleview/${SPLIT}"
    python "${ROOT}/scripts/infer_single_arm_singleview_autoregressive.py" \
      --checkpoints-dir "${CKPT}" \
      --experiment "${EXP}" \
      --dataset-path "${DATASET}" \
      --save-dir "${SCRATCH}/${TAG}" \
      --num-frames 97 \
      --save-fps 15 \
      --output-dir "${SCRATCH}/out_${TAG}" \
      --guidance 1.5 \
      --num-episodes "${NUM_EPISODES}" \
      "$@" > "${SCRATCH}/logs/${TAG}.log" 2>&1
    ;;
  bimanual)
    CKPT="${ROOT}/exp_logs/dreamdojo_logs/dreamdojo/exp_aloha_resume70k/aloha_bimanual_singleview_full/checkpoints"
    EXP="dreamdojo_2b_480_640_aloha_resume70k_bimanual_sv"
    paths=""
    for d in $(find "${ROOT}/datasets/bimanual/singleview/${SPLIT}" -maxdepth 1 -mindepth 1 -type d | sort); do
      if find "$d/data" -name "*.parquet" 2>/dev/null | grep -q .; then
        if [ -n "${paths}" ]; then paths="${paths},"; fi
        paths="${paths}${d}"
      fi
    done
    python "${ROOT}/scripts/infer_bimanual_singleview_single_segment.py" \
      --checkpoints-dir "${CKPT}" \
      --experiment "${EXP}" \
      --dataset-path "${paths}" \
      --save-dir "${SCRATCH}/${TAG}" \
      --chunk-size 12 \
      --save-fps 10 \
      --output-dir "${SCRATCH}/out_${TAG}" \
      --guidance 1.5 \
      --num-episodes "${NUM_EPISODES}" \
      --save-full \
      "$@" > "${SCRATCH}/logs/${TAG}.log" 2>&1
    ;;
  *)
    echo "unknown embodiment '${EMB}' (want single_arm or bimanual)" >&2
    exit 2
    ;;
esac

echo "=== ${EMB} ${TAG} ==="
grep -E "\[Timing\] Generated|^PSNR:|^SSIM:|^LPIPS:|Attention profile" "${SCRATCH}/logs/${TAG}.log" \
  || tail -25 "${SCRATCH}/logs/${TAG}.log"
