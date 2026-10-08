#!/usr/bin/env bash
# DreamDojo single-arm singleview (DROID) inference with SVG1 (Sparse-VideoGen v1,
# stripe sparse + MSE profiling). Mirrors run_single_arm_singleview_dreamdojo_dense.sh
# exactly (same ckpt / dataset / inference script), only adding the SVG1 flags and
# routing output to svg1/.
#
# Output: sampling_dataset/svg1/single_arm/output/singleview/dreamdojo/{split}/iter_*/episode_*/
#
# Usage:
#   bash scripts/run/dreamdojo/run_single_arm_singleview_dreamdojo_svg1.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamdojo/run_single_arm_singleview_dreamdojo_svg1.sh
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/checkpoints/finetuned/dreamdojo/dreamdojo/exp_single_arm_sv/single_arm_singleview/checkpoints}"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_single_arm_sv}"
NUM_FRAMES="${NUM_FRAMES:-97}"
_SAVE_ROOT_ENV="${SAVE_ROOT:-}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/svg1/single_arm/output/singleview/dreamdojo}"
# Intermediate --output-dir: OUTPUT_ROOT if set, else <SAVE_ROOT>/_output when SAVE_ROOT
# was given (keeps scratch runs self-contained), else the historical ${ROOT}/output.
OUT_CACHE_ROOT="${OUTPUT_ROOT:-${_SAVE_ROOT_ENV:+${_SAVE_ROOT_ENV}/_output}}"
OUT_CACHE_ROOT="${OUT_CACHE_ROOT:-${ROOT}/output}"
NUM_EPISODES="${NUM_EPISODES:-}"
GUIDANCE="${GUIDANCE:-1.5}"
SPLITS="${SPLITS:-makovian non_makovian}"

# SVG1 parameters (tuned; see dreamdojo-single-arm-singleview-sparse-attention-insights)
# sparsity 0.25 is the quality-leaning point among SVG1 configs: 158.51s vs dense
# 144.80s = 0.91x (SLOWER) at PSNR 25.85 / SSIM 0.898 (already below dense 27.57 / 0.920).
# SVG1 never beats dense here (same kernel-bound S=4800 story as humanoid).
# BREAK POINT is sparsity <= 0.05: PSNR 23.43, last-frame visibly smeared.
# sparsity 0.10 is equal-quality to 0.25 at 1.00x (no speedup). Kept runnable as a
# negative benchmark point. Profile is cached (recompute_every=999999).
SVG_START_LAYER_IDX="${SVG_START_LAYER_IDX:-2}"
SVG_DENSE_STEP_FRAC="${SVG_DENSE_STEP_FRAC:-0.05}"
SVG_SPARSITY="${SVG_SPARSITY:-0.25}"
SVG_NUM_SAMPLED_ROWS="${SVG_NUM_SAMPLED_ROWS:-16}"
SVG_SAMPLE_MSE_MAX_ROW="${SVG_SAMPLE_MSE_MAX_ROW:-1200}"
SVG_PROFILE_RECOMPUTE_EVERY="${SVG_PROFILE_RECOMPUTE_EVERY:-999999}"

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
  echo "=== DreamDojo SVG1 single-arm SV: ${split} -> ${save_base} ==="
  python "${ROOT}/scripts/infer/infer_single_arm_singleview_autoregressive.py" \
    --checkpoints-dir "${CHECKPOINTS_DIR}" \
    --experiment "${EXPERIMENT}" \
    --dataset-path "${dataset_dir}" \
    --save-dir "${save_base}" \
    --num-frames "${NUM_FRAMES}" \
    --save-fps 15 \
    --output-dir "${OUT_CACHE_ROOT}/dreamdojo_svg1_single_arm_sv_${split}" \
    --guidance "${GUIDANCE}" \
    --use-svg \
    --svg-variant svg1 \
    --svg-start-layer-idx "${SVG_START_LAYER_IDX}" \
    --svg-dense-step-frac "${SVG_DENSE_STEP_FRAC}" \
    --svg-sparsity "${SVG_SPARSITY}" \
    --svg-num-sampled-rows "${SVG_NUM_SAMPLED_ROWS}" \
    --svg-sample-mse-max-row "${SVG_SAMPLE_MSE_MAX_ROW}" \
    --svg-profile-recompute-every "${SVG_PROFILE_RECOMPUTE_EVERY}" \
    "${ep_flag[@]}"
done

echo ""
echo "=========================================================="
echo "Done! Results saved to: ${SAVE_ROOT}"
echo "=========================================================="
