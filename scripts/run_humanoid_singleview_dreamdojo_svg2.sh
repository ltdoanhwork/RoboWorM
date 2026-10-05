#!/usr/bin/env bash
# DreamDojo full-episode inference with SVG2 (Sparse-VideoGen v2, semantic-aware permutation
# + kmeans block-sparse flashinfer attention).
# Mirrors run_humanoid_singleview_dreamdojo_dense.sh exactly (same ckpt / dataset /
# inference script), only adding the SVG2 flags and routing output to svg2/.
#
# NOTE: SVG2 needs flashinfer (>=0.3) for the VariableBlockSparse wrapper. Without it the
# vendored kmeans_utils falls back to a dense (BH,S,S) mask SDPA which OOMs.
#
# Output: sampling_dataset/svg2/humanoid/output/singleview/dreamdojo/{split}/iter_*/episode_*/
#
# Usage:
#   bash scripts/run_humanoid_singleview_dreamdojo_svg2.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_humanoid_singleview_dreamdojo_svg2.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/checkpoints/base/dreamdojo/2B_GR1_post-train/iter_000050000}"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_gr1}"
NUM_FRAMES="${NUM_FRAMES:-49}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/svg2/humanoid/output/singleview/dreamdojo}"
NUM_EPISODES="${NUM_EPISODES:-}"
GUIDANCE="${GUIDANCE:-1.5}"
SPLITS="${SPLITS:-makovian non_makovian}"
SAVE_MODE="${SAVE_MODE:-video_only}"

# SVG2 parameters (tuned defaults; see dreamdojo-humanoid-singleview-sparse-attention rule)
SVG_START_LAYER_IDX="${SVG_START_LAYER_IDX:-4}"
SVG_DENSE_STEP_FRAC="${SVG_DENSE_STEP_FRAC:-0.05}"
SVG_NUM_Q_CENTROIDS="${SVG_NUM_Q_CENTROIDS:-32}"
SVG_NUM_K_CENTROIDS="${SVG_NUM_K_CENTROIDS:-64}"
# Tuned (1 makovian ep, iter_000050000): top_p 0.6 keeps quality closest to dense
# (PSNR 19.14 / SSIM 0.713 vs dense 19.24 / 0.743) but costs 95.44s vs 56.41s = 0.59x.
# SVG2 is the SLOWEST method here: per-step k-means + permute overhead dwarfs the
# attention saved (self-attn is only ~15% of gen time on this 2B DiT).
# BREAK POINT is top_p <= 0.3: PSNR 18.65 but LPIPS jumps 0.272 -> 0.350, still 0.72x.
# Kept runnable as a negative benchmark point (acceleration-benchmark rule).
SVG_TOP_P_KMEANS="${SVG_TOP_P_KMEANS:-0.6}"
SVG_MIN_KC_RATIO="${SVG_MIN_KC_RATIO:-0.15}"
SVG_KMEANS_ITER_INIT="${SVG_KMEANS_ITER_INIT:-8}"
SVG_KMEANS_ITER_STEP="${SVG_KMEANS_ITER_STEP:-2}"

ep_flag=()
if [[ -n "${NUM_EPISODES}" ]]; then
  ep_flag=(--num-episodes "${NUM_EPISODES}")
fi
save_flag=()
if [[ "${SAVE_MODE}" == "full" ]]; then
  save_flag=(--save-full)
fi

for split in ${SPLITS}; do
  dataset_dir="${ROOT}/datasets/humanoid/singleview/${split}"
  if [ ! -d "${dataset_dir}/data" ]; then
    echo "Skipping ${split}: no data directory found at ${dataset_dir}/data" >&2
    continue
  fi
  save_base="${SAVE_ROOT}/${split}"
  mkdir -p "${save_base}"
  echo "=== DreamDojo SVG2 full-episode: ${split} -> ${save_base} ==="
  python "${ROOT}/scripts/infer_humanoid_singleview_full_episode.py" \
    --checkpoints-dir "${CHECKPOINTS_DIR}" \
    --experiment "${EXPERIMENT}" \
    --dataset-path "${dataset_dir}" \
    --save-dir "${save_base}" \
    --num-frames "${NUM_FRAMES}" \
    --output-dir "${ROOT}/output/dreamdojo_svg2_humanoid_sv_${split}" \
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
