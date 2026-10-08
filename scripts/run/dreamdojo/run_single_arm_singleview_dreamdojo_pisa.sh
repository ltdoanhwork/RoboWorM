#!/usr/bin/env bash
# DreamDojo single-arm singleview (DROID) inference with PISA (piecewise sparse attention).
# Mirrors run_single_arm_singleview_dreamdojo_dense.sh / _dicache.sh exactly
# (same ckpt / dataset / inference script), only adding the PISA flags and
# routing output to the pisa/ method tree.
#
# Output: sampling_dataset/pisa/single_arm/output/singleview/dreamdojo/{split}/iter_*/episode_*/
#
# Usage:
#   bash scripts/run/dreamdojo/run_single_arm_singleview_dreamdojo_pisa.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamdojo/run_single_arm_singleview_dreamdojo_pisa.sh
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/checkpoints/finetuned/dreamdojo/dreamdojo/exp_single_arm_sv/single_arm_singleview/checkpoints}"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_single_arm_sv}"
NUM_FRAMES="${NUM_FRAMES:-97}"
_SAVE_ROOT_ENV="${SAVE_ROOT:-}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/pisa/single_arm/output/singleview/dreamdojo}"
# Intermediate --output-dir: OUTPUT_ROOT if set, else <SAVE_ROOT>/_output when SAVE_ROOT
# was given (keeps scratch runs self-contained), else the historical ${ROOT}/output.
OUT_CACHE_ROOT="${OUTPUT_ROOT:-${_SAVE_ROOT_ENV:+${_SAVE_ROOT_ENV}/_output}}"
OUT_CACHE_ROOT="${OUT_CACHE_ROOT:-${ROOT}/output}"
NUM_EPISODES="${NUM_EPISODES:-}"
GUIDANCE="${GUIDANCE:-1.5}"
SPLITS="${SPLITS:-makovian non_makovian}"

# PISA parameters (tuned; see dreamdojo-single-arm-singleview-sparse-attention-insights)
# density 0.2 is the LAST point before the quality break -> 142.38s vs dense 144.80s
# (1.02x) at PSNR 27.30 / SSIM 0.916 (== dense 27.57 / 0.920).
# BREAK POINT is density <= 0.10: PSNR 25.50 @0.10, 21.16 @0.05, 17.14 @0.02 (melted).
# Speed saturates at ~1.10x (Amdahl ceiling 1.20x; self-attn is 17% of gen time).
PISA_DENSITY="${PISA_DENSITY:-0.2}"
PISA_BLOCK_SIZE="${PISA_BLOCK_SIZE:-64}"
PISA_START_LAYER_IDX="${PISA_START_LAYER_IDX:-2}"

ep_flag=()
if [[ -n "${NUM_EPISODES}" ]]; then
  ep_flag=(--num-episodes "${NUM_EPISODES}")
fi

for split in ${SPLITS}; do
  dataset_dir="${ROOT}/datasets/single_arm/singleview/${split}"
  if [ ! -d "${dataset_dir}/data" ]; then
    echo "Skipping ${split}: no data directory found at ${dataset_dir}/data" >&2
    continue
  fi
  save_base="${SAVE_ROOT}/${split}"
  mkdir -p "${save_base}"
  echo "=== DreamDojo PISA single-arm SV: ${split} -> ${save_base} ==="
  python "${ROOT}/scripts/infer/infer_single_arm_singleview_autoregressive.py" \
    --checkpoints-dir "${CHECKPOINTS_DIR}" \
    --experiment "${EXPERIMENT}" \
    --dataset-path "${dataset_dir}" \
    --save-dir "${save_base}" \
    --num-frames "${NUM_FRAMES}" \
    --save-fps 15 \
    --output-dir "${OUT_CACHE_ROOT}/dreamdojo_pisa_single_arm_sv_${split}" \
    --guidance "${GUIDANCE}" \
    --use-pisa \
    --pisa-density "${PISA_DENSITY}" \
    --pisa-block-size "${PISA_BLOCK_SIZE}" \
    --pisa-start-layer-idx "${PISA_START_LAYER_IDX}" \
    "${ep_flag[@]}"
done

echo ""
echo "=========================================================="
echo "Done! Results saved to: ${SAVE_ROOT}"
echo "=========================================================="
