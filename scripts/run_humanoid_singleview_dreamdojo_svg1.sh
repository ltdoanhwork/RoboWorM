#!/usr/bin/env bash
# DreamDojo full-episode inference with SVG1 (Sparse-VideoGen v1, stripe sparse + MSE profiling).
# Mirrors run_humanoid_singleview_dreamdojo_dense.sh exactly (same ckpt / dataset /
# inference script), only adding the SVG1 flags and routing output to svg1/.
#
# Output: sampling_dataset/svg1/humanoid/output/singleview/dreamdojo/{split}/iter_*/episode_*/
#
# Usage:
#   bash scripts/run_humanoid_singleview_dreamdojo_svg1.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_humanoid_singleview_dreamdojo_svg1.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/checkpoints/base/dreamdojo/2B_GR1_post-train/iter_000050000}"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_gr1}"
NUM_FRAMES="${NUM_FRAMES:-49}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/svg1/humanoid/output/singleview/dreamdojo}"
NUM_EPISODES="${NUM_EPISODES:-}"
GUIDANCE="${GUIDANCE:-1.5}"
SPLITS="${SPLITS:-makovian non_makovian}"
SAVE_MODE="${SAVE_MODE:-video_only}"

# SVG1 parameters (tuned defaults; see dreamdojo-humanoid-singleview-sparse-attention rule)
SVG_START_LAYER_IDX="${SVG_START_LAYER_IDX:-4}"
SVG_DENSE_STEP_FRAC="${SVG_DENSE_STEP_FRAC:-0.05}"
# Tuned (1 makovian ep, iter_000050000): sparsity 0.25 is the best point measured
# -> 60.63s vs dense 56.41s = 0.93x (still SLOWER) at PSNR 18.42 / SSIM 0.708.
# SVG1 cannot beat dense on this target, and stage-level profiling explains why
# (scripts/_diag_dreamdojo_svg.py, at B=1/H=16/S=4800/D=128):
#   dense flash              1.241 ms/call   <-- must beat this
#   compiled flex_attention  1.027 ms (0.83x dense)  <-- the real sparse win
#   rearrange + placement    0.276 ms        <-- unavoidable, eats the win
# BREAK POINT is sparsity <= 0.15 (PSNR 17.91 @0.15, 17.06 @0.10). Lowering sparsity
# barely adds speed because T=4 caps the diagonal mask width: blocks kept only drops
# 51.4% -> 36.8% going from sparsity 0.25 -> 0.10.
SVG_SPARSITY="${SVG_SPARSITY:-0.25}"
SVG_NUM_SAMPLED_ROWS="${SVG_NUM_SAMPLED_ROWS:-16}"
SVG_SAMPLE_MSE_MAX_ROW="${SVG_SAMPLE_MSE_MAX_ROW:-1200}"
# BUG FIX: upstream re-ran the spatial/temporal MSE head profile on EVERY attention
# call (0.589 ms = ~47% of a dense call). Caching it recovered 0.86x -> 0.93x.
# 999999 = profile once per episode. Set to 1 to restore upstream behaviour.
SVG_PROFILE_RECOMPUTE_EVERY="${SVG_PROFILE_RECOMPUTE_EVERY:-999999}"

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
  echo "=== DreamDojo SVG1 full-episode: ${split} -> ${save_base} ==="
  python "${ROOT}/scripts/infer_humanoid_singleview_full_episode.py" \
    --checkpoints-dir "${CHECKPOINTS_DIR}" \
    --experiment "${EXPERIMENT}" \
    --dataset-path "${dataset_dir}" \
    --save-dir "${save_base}" \
    --num-frames "${NUM_FRAMES}" \
    --output-dir "${ROOT}/output/dreamdojo_svg1_humanoid_sv_${split}" \
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
