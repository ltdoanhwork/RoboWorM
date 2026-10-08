#!/usr/bin/env bash
# DreamDojo single-arm singleview (DROID) inference with SVG2 (Sparse-VideoGen v2,
# semantic-aware permutation + kmeans block-sparse flashinfer attention). Mirrors
# run_single_arm_singleview_dreamdojo_dense.sh exactly (same ckpt / dataset /
# inference script), only adding the SVG2 flags and routing output to svg2/.
#
# NOTE: SVG2 needs flashinfer (>=0.3) for the VariableBlockSparse wrapper. Without it
# the vendored kmeans_utils falls back to a dense (BH,S,S) mask SDPA which OOMs.
#
# Output: sampling_dataset/svg2/single_arm/output/singleview/dreamdojo/{split}/iter_*/episode_*/
#
# Usage:
#   bash scripts/run/dreamdojo/run_single_arm_singleview_dreamdojo_svg2.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamdojo/run_single_arm_singleview_dreamdojo_svg2.sh
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/checkpoints/finetuned/dreamdojo/dreamdojo/exp_single_arm_sv/single_arm_singleview/checkpoints}"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_single_arm_sv}"
NUM_FRAMES="${NUM_FRAMES:-97}"
_SAVE_ROOT_ENV="${SAVE_ROOT:-}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/svg2/single_arm/output/singleview/dreamdojo}"
# Intermediate --output-dir: OUTPUT_ROOT if set, else <SAVE_ROOT>/_output when SAVE_ROOT
# was given (keeps scratch runs self-contained), else the historical ${ROOT}/output.
OUT_CACHE_ROOT="${OUTPUT_ROOT:-${_SAVE_ROOT_ENV:+${_SAVE_ROOT_ENV}/_output}}"
OUT_CACHE_ROOT="${OUT_CACHE_ROOT:-${ROOT}/output}"
NUM_EPISODES="${NUM_EPISODES:-}"
GUIDANCE="${GUIDANCE:-1.5}"
SPLITS="${SPLITS:-makovian non_makovian}"

# SVG2 parameters (tuned; see dreamdojo-single-arm-singleview-sparse-attention-insights)
# top_p 0.6 keeps the least-bad quality (PSNR 25.70 / SSIM 0.886 vs dense 27.57 / 0.920)
# but costs 210.04s vs 144.80s = 0.69x. SVG2 cannot win: block-sparse kernel at
# S=4800 is slower than dense flash even before kmeans overhead.
# BREAK POINT is top_p <= 0.3: PSNR 23.08, LPIPS 0.150 -> 0.300, still 0.77x.
SVG_START_LAYER_IDX="${SVG_START_LAYER_IDX:-4}"
SVG_DENSE_STEP_FRAC="${SVG_DENSE_STEP_FRAC:-0.05}"
SVG_NUM_Q_CENTROIDS="${SVG_NUM_Q_CENTROIDS:-32}"
SVG_NUM_K_CENTROIDS="${SVG_NUM_K_CENTROIDS:-64}"
SVG_TOP_P_KMEANS="${SVG_TOP_P_KMEANS:-0.6}"
SVG_MIN_KC_RATIO="${SVG_MIN_KC_RATIO:-0.15}"
SVG_KMEANS_ITER_INIT="${SVG_KMEANS_ITER_INIT:-8}"
SVG_KMEANS_ITER_STEP="${SVG_KMEANS_ITER_STEP:-2}"

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
  echo "=== DreamDojo SVG2 single-arm SV: ${split} -> ${save_base} ==="
  python "${ROOT}/scripts/infer/infer_single_arm_singleview_autoregressive.py" \
    --checkpoints-dir "${CHECKPOINTS_DIR}" \
    --experiment "${EXPERIMENT}" \
    --dataset-path "${dataset_dir}" \
    --save-dir "${save_base}" \
    --num-frames "${NUM_FRAMES}" \
    --save-fps 15 \
    --output-dir "${OUT_CACHE_ROOT}/dreamdojo_svg2_single_arm_sv_${split}" \
    --guidance "${GUIDANCE}" \
    --use-svg \
    --svg-variant svg2 \
    --svg-start-layer-idx "${SVG_START_LAYER_IDX}" \
    --svg-dense-step-frac "${SVG_DENSE_STEP_FRAC}" \
    --svg-num-q-centroids "${SVG_NUM_Q_CENTROIDS}" \
    --svg-num-k-centroids "${SVG_NUM_K_CENTROIDS}" \
    --svg-top-p-kmeans "${SVG_TOP_P_KMEANS}" \
    --svg-min-kc-ratio "${SVG_MIN_KC_RATIO}" \
    --svg-kmeans-iter-init "${SVG_KMEANS_ITER_INIT}" \
    --svg-kmeans-iter-step "${SVG_KMEANS_ITER_STEP}" \
    "${ep_flag[@]}"
done

echo ""
echo "=========================================================="
echo "Done! Results saved to: ${SAVE_ROOT}"
echo "=========================================================="
