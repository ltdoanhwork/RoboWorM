#!/usr/bin/env bash
# DreamDojo bimanual (ALOHA) singleview inference with SVG2 (Sparse-VideoGen v2,
# semantic-aware permutation + kmeans block-sparse flashinfer attention). Mirrors
# run_bimanual_singleview_dreamdojo_full_episode.sh exactly (same ckpt / dataset /
# inference script), only adding the SVG2 flags and routing output to svg2/.
#
# NOTE: SVG2 needs flashinfer (>=0.3) for the VariableBlockSparse wrapper. Without it
# the vendored kmeans_utils falls back to a dense (BH,S,S) mask SDPA which OOMs.
#
# Output: sampling_dataset/svg2/bimanual/output/singleview/dreamdojo/{split}/iter_*/{task}__episode_*/
#
# Usage:
#   bash scripts/run/dreamdojo/run_bimanual_singleview_dreamdojo_svg2.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamdojo/run_bimanual_singleview_dreamdojo_svg2.sh
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/checkpoints/finetuned/dreamdojo/dreamdojo/exp_aloha_resume70k/aloha_bimanual_singleview_full/checkpoints}"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_aloha_resume70k_bimanual_sv}"
CHUNK_SIZE="${CHUNK_SIZE:-12}"
SAVE_FPS="${SAVE_FPS:-10}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/svg2/bimanual/output/singleview/dreamdojo}"
NUM_EPISODES="${NUM_EPISODES:-}"
GUIDANCE="${GUIDANCE:-1.5}"
SPLITS="${SPLITS:-makovian non_makovian}"
SAVE_MODE="${SAVE_MODE:-video_only}"

# SVG2 parameters (tuned; see dreamdojo-bimanual-singleview-sparse-attention-insights)
# top_p 0.6 keeps quality (PSNR 18.86 / SSIM 0.712 vs dense 18.44 / 0.710) but costs
# 567.92s vs 390.08s = 0.69x. SVG2 cannot win: block-sparse kernel at S=4800 is
# slower than dense flash even before kmeans overhead.
# BREAK POINT is top_p <= 0.3: SSIM 0.671, LPIPS 0.305 -> 0.388, last-frame smear.
SVG_START_LAYER_IDX="${SVG_START_LAYER_IDX:-4}"
SVG_DENSE_STEP_FRAC="${SVG_DENSE_STEP_FRAC:-0.05}"
SVG_NUM_Q_CENTROIDS="${SVG_NUM_Q_CENTROIDS:-32}"
SVG_NUM_K_CENTROIDS="${SVG_NUM_K_CENTROIDS:-64}"
SVG_TOP_P_KMEANS="${SVG_TOP_P_KMEANS:-0.6}"
SVG_MIN_KC_RATIO="${SVG_MIN_KC_RATIO:-0.15}"
SVG_KMEANS_ITER_INIT="${SVG_KMEANS_ITER_INIT:-8}"
SVG_KMEANS_ITER_STEP="${SVG_KMEANS_ITER_STEP:-2}"

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
  echo "=== DreamDojo SVG2 bimanual SV: ${split} -> ${save_base} ==="
  python "${ROOT}/scripts/infer/infer_bimanual_singleview_single_segment.py" \
    --checkpoints-dir "${CHECKPOINTS_DIR}" \
    --experiment "${EXPERIMENT}" \
    --dataset-path "${dataset_paths}" \
    --save-dir "${save_base}" \
    --chunk-size "${CHUNK_SIZE}" \
    --save-fps "${SAVE_FPS}" \
    --output-dir "${ROOT}/output/dreamdojo_svg2_bimanual_sv_${split}" \
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
    "${ep_flag[@]}" "${save_flag[@]}"
done

echo ""
echo "=========================================================="
echo "Done! Results saved to: ${SAVE_ROOT}"
echo "=========================================================="
