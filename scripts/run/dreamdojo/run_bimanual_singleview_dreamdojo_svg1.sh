#!/usr/bin/env bash
# DreamDojo bimanual (ALOHA) singleview inference with SVG1 (Sparse-VideoGen v1,
# stripe sparse + MSE profiling). Mirrors run_bimanual_singleview_dreamdojo_full_episode.sh
# exactly (same ckpt / dataset / inference script), only adding the SVG1 flags and
# routing output to svg1/.
#
# Output: sampling_dataset/svg1/bimanual/output/singleview/dreamdojo/{split}/iter_*/{task}__episode_*/
#
# Usage:
#   bash scripts/run/dreamdojo/run_bimanual_singleview_dreamdojo_svg1.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamdojo/run_bimanual_singleview_dreamdojo_svg1.sh
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/checkpoints/finetuned/dreamdojo/dreamdojo/exp_aloha_resume70k/aloha_bimanual_singleview_full/checkpoints}"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_aloha_resume70k_bimanual_sv}"
CHUNK_SIZE="${CHUNK_SIZE:-12}"
SAVE_FPS="${SAVE_FPS:-10}"
_SAVE_ROOT_ENV="${SAVE_ROOT:-}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/svg1/bimanual/output/singleview/dreamdojo}"
# Intermediate --output-dir: OUTPUT_ROOT if set, else <SAVE_ROOT>/_output when SAVE_ROOT
# was given (keeps scratch runs self-contained), else the historical ${ROOT}/output.
OUT_CACHE_ROOT="${OUTPUT_ROOT:-${_SAVE_ROOT_ENV:+${_SAVE_ROOT_ENV}/_output}}"
OUT_CACHE_ROOT="${OUT_CACHE_ROOT:-${ROOT}/output}"
NUM_EPISODES="${NUM_EPISODES:-}"
GUIDANCE="${GUIDANCE:-1.5}"
SPLITS="${SPLITS:-makovian non_makovian}"
SAVE_MODE="${SAVE_MODE:-video_only}"

# SVG1 parameters (tuned; see dreamdojo-bimanual-singleview-sparse-attention-insights)
# sparsity 0.25 / start 2 / profile cached: 402.00s vs dense 390.08s = 0.97x (SLOWER)
# at PSNR 17.99 / SSIM 0.691 (already below dense 18.44 / 0.710). SVG1 never beats
# dense here (same S=4800 flex_attention overhead as humanoid / single_arm).
# BREAK POINT is sparsity <= 0.05: PSNR 14.21, SSIM 0.529, last-frame DESTROYED.
SVG_START_LAYER_IDX="${SVG_START_LAYER_IDX:-2}"
SVG_DENSE_STEP_FRAC="${SVG_DENSE_STEP_FRAC:-0.05}"
SVG_SPARSITY="${SVG_SPARSITY:-0.25}"
SVG_NUM_SAMPLED_ROWS="${SVG_NUM_SAMPLED_ROWS:-16}"
SVG_SAMPLE_MSE_MAX_ROW="${SVG_SAMPLE_MSE_MAX_ROW:-1200}"
SVG_PROFILE_RECOMPUTE_EVERY="${SVG_PROFILE_RECOMPUTE_EVERY:-999999}"

get_task_paths() {
  local split="$1"
  local split_dir="${ROOT}/datasets/bimanual/singleview/${split}"
  local paths=""
  for d in $(find "${split_dir}" -maxdepth 1 -mindepth 1 -type d | sort); do
    if find "$d/data" -name "*.parquet" 2>/dev/null | grep -q .; then
      if [ -n "${paths}" ]; then
        paths="${paths},"
      fi
      paths="${paths}${d}"
    fi
  done
  echo "${paths}"
}

ep_flag=()
if [[ -n "${NUM_EPISODES}" ]]; then
  ep_flag=(--num-episodes "${NUM_EPISODES}")
fi
save_flag=()
if [[ "${SAVE_MODE}" == "full" ]]; then
  save_flag=(--save-full)
fi

for split in ${SPLITS}; do
  split_dir="${ROOT}/datasets/bimanual/singleview/${split}"
  if [ ! -d "${split_dir}" ]; then
    echo "Skipping ${split}: directory not found at ${split_dir}" >&2
    continue
  fi

  dataset_paths="$(get_task_paths "${split}")"
  if [ -z "${dataset_paths}" ]; then
    echo "Skipping ${split}: no task directories with data/ found" >&2
    continue
  fi

  save_base="${SAVE_ROOT}/${split}"
  mkdir -p "${save_base}"
  echo "=== DreamDojo SVG1 bimanual SV: ${split} -> ${save_base} ==="
  python "${ROOT}/scripts/infer/infer_bimanual_singleview_single_segment.py" \
    --checkpoints-dir "${CHECKPOINTS_DIR}" \
    --experiment "${EXPERIMENT}" \
    --dataset-path "${dataset_paths}" \
    --save-dir "${save_base}" \
    --chunk-size "${CHUNK_SIZE}" \
    --save-fps "${SAVE_FPS}" \
    --output-dir "${OUT_CACHE_ROOT}/dreamdojo_svg1_bimanual_sv_${split}" \
    --guidance "${GUIDANCE}" \
    --use-svg \
    --svg-variant svg1 \
    --svg-start-layer-idx "${SVG_START_LAYER_IDX}" \
    --svg-dense-step-frac "${SVG_DENSE_STEP_FRAC}" \
    --svg-sparsity "${SVG_SPARSITY}" \
    --svg-num-sampled-rows "${SVG_NUM_SAMPLED_ROWS}" \
    --svg-sample-mse-max-row "${SVG_SAMPLE_MSE_MAX_ROW}" \
    --svg-profile-recompute-every "${SVG_PROFILE_RECOMPUTE_EVERY}" \
    "${ep_flag[@]}" "${save_flag[@]}"
done

echo ""
echo "=========================================================="
echo "Done! Results saved to: ${SAVE_ROOT}"
echo "=========================================================="
