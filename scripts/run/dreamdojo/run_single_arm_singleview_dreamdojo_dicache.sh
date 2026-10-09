#!/usr/bin/env bash
# DreamDojo autoregressive inference with DiCache acceleration on single-arm singleview.
# Full rollout: video output length matches the GT input episode (chunk-strided autoregressive).
# Runs datasets/single_arm/singleview/makovian and .../non_makovian separately.
# Output: sampling_dataset/dicache/single_arm/output/singleview/dreamdojo/{split}/iter_*/episode_*/
#
# Usage:
#   bash scripts/run/dreamdojo/run_single_arm_singleview_dreamdojo_dicache.sh
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run/dreamdojo/run_single_arm_singleview_dreamdojo_dicache.sh
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/checkpoints/finetuned/dreamdojo/dreamdojo/exp_single_arm_sv/single_arm_singleview/checkpoints}"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_single_arm_sv}"
NUM_FRAMES="${NUM_FRAMES:-97}"
_SAVE_ROOT_ENV="${SAVE_ROOT:-}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/dicache/single_arm/output/singleview/dreamdojo}"
# Intermediate --output-dir: OUTPUT_ROOT if set, else <SAVE_ROOT>/_output when SAVE_ROOT
# was given (keeps scratch runs self-contained), else the historical ${ROOT}/output.
OUT_CACHE_ROOT="${OUTPUT_ROOT:-${_SAVE_ROOT_ENV:+${_SAVE_ROOT_ENV}/_output}}"
OUT_CACHE_ROOT="${OUT_CACHE_ROOT:-${ROOT}/output}"
NUM_EPISODES="${NUM_EPISODES:-}"
# Guidance: same as the cell's dense launcher (run_single_arm_singleview_dreamdojo_dense.sh: GUIDANCE, default 0 =
# the infer-script default). Was 1.5 before 2026-10-09, so fidelity-vs-dense compared
# different guidance scales.
GUIDANCE="${GUIDANCE:-0}"

# DiCache parameters
DICACHE_NUM_STEPS="${DICACHE_NUM_STEPS:-35}"
DICACHE_REL_L1_THRESH="${DICACHE_REL_L1_THRESH:-0.08}"
DICACHE_RET_RATIO="${DICACHE_RET_RATIO:-0.2}"
DICACHE_PROBE_DEPTH="${DICACHE_PROBE_DEPTH:-2}"

ep_flag=()
if [[ -n "${NUM_EPISODES}" ]]; then
  ep_flag=(--num-episodes "${NUM_EPISODES}")
fi

for split in makovian non_makovian; do
  dataset_dir="${ROOT}/datasets/single_arm/singleview/${split}"
  if [ ! -d "${dataset_dir}/data" ]; then
    echo "Skipping ${split}: no data directory found at ${dataset_dir}/data" >&2
    continue
  fi
  save_base="${SAVE_ROOT}/${split}"
  mkdir -p "${save_base}"
  echo "=== DreamDojo DiCache single-arm SV: ${split} -> ${save_base} ==="
  python "${ROOT}/scripts/infer/infer_single_arm_singleview_autoregressive.py" \
    --checkpoints-dir "${CHECKPOINTS_DIR}" \
    --experiment "${EXPERIMENT}" \
    --dataset-path "${dataset_dir}" \
    --save-dir "${save_base}" \
    --num-frames "${NUM_FRAMES}" \
    --save-fps 15 \
    --output-dir "${OUT_CACHE_ROOT}/dreamdojo_dicache_single_arm_sv_${split}" \
    --guidance "${GUIDANCE}" \
    --dicache-enabled \
    --dicache-num-steps "${DICACHE_NUM_STEPS}" \
    --dicache-rel-l1-thresh "${DICACHE_REL_L1_THRESH}" \
    --dicache-ret-ratio "${DICACHE_RET_RATIO}" \
    --dicache-probe-depth "${DICACHE_PROBE_DEPTH}" \
    "${ep_flag[@]}"
done
