#!/usr/bin/env bash
# DreamDojo full-episode inference with PISA (piecewise sparse attention).
# Mirrors run_humanoid_singleview_dreamdojo_full_episode.sh / _dicache.sh exactly
# (same ckpt / dataset / inference script), only adding the PISA flags and
# routing output to the pisa/ method tree.
#
# Output: sampling_dataset/pisa/humanoid/output/singleview/dreamdojo/{split}/iter_*/episode_*/
#
# Usage:
#   bash scripts/run/dreamdojo/run_humanoid_singleview_dreamdojo_pisa.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamdojo/run_humanoid_singleview_dreamdojo_pisa.sh
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/checkpoints/base/dreamdojo/2B_GR1_post-train/iter_000050000}"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_gr1}"
NUM_FRAMES="${NUM_FRAMES:-49}"
_SAVE_ROOT_ENV="${SAVE_ROOT:-}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/pisa/humanoid/output/singleview/dreamdojo}"
# Intermediate --output-dir: OUTPUT_ROOT if set, else <SAVE_ROOT>/_output when SAVE_ROOT
# was given (keeps scratch runs self-contained), else the historical ${ROOT}/output.
OUT_CACHE_ROOT="${OUTPUT_ROOT:-${_SAVE_ROOT_ENV:+${_SAVE_ROOT_ENV}/_output}}"
OUT_CACHE_ROOT="${OUT_CACHE_ROOT:-${ROOT}/output}"
NUM_EPISODES="${NUM_EPISODES:-}"
# Guidance: same as the cell's dense launcher (run_humanoid_singleview_dreamdojo_full_episode.sh: GUIDANCE, default 0 =
# the infer-script default). Was 1.5 before 2026-10-09, so fidelity-vs-dense compared
# different guidance scales.
GUIDANCE="${GUIDANCE:-0}"
SPLITS="${SPLITS:-makovian non_makovian}"
SAVE_MODE="${SAVE_MODE:-video_only}"   # video_only | full (full also writes gt/merged/metrics.json)

# PISA parameters (tuned defaults; see dreamdojo-humanoid-singleview-sparse-attention rule)
# Tuned (1 makovian ep, iter_000050000): density 0.2 is the LAST point before the
# quality break -> 57.21s vs dense 56.41s (0.99x) at PSNR 17.78 / SSIM 0.660.
# BREAK POINT is density <= 0.10: PSNR 16.93 @0.10, 16.14 @0.05, 14.48 @0.02 (melted).
# Speed saturates at ~1.07x no matter how low the density goes, because self-attention
# is only ~15% of generation time on this model (Amdahl ceiling 1.18x).
PISA_DENSITY="${PISA_DENSITY:-0.2}"
PISA_BLOCK_SIZE="${PISA_BLOCK_SIZE:-64}"
PISA_START_LAYER_IDX="${PISA_START_LAYER_IDX:-2}"

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
  echo "=== DreamDojo PISA full-episode: ${split} -> ${save_base} ==="
  python "${ROOT}/scripts/infer/infer_humanoid_singleview_full_episode.py" \
    --checkpoints-dir "${CHECKPOINTS_DIR}" \
    --experiment "${EXPERIMENT}" \
    --dataset-path "${dataset_dir}" \
    --save-dir "${save_base}" \
    --num-frames "${NUM_FRAMES}" \
    --output-dir "${OUT_CACHE_ROOT}/dreamdojo_pisa_humanoid_sv_${split}" \
    --guidance "${GUIDANCE}" \
    --use-pisa \
    --pisa-density "${PISA_DENSITY}" \
    --pisa-block-size "${PISA_BLOCK_SIZE}" \
    --pisa-start-layer-idx "${PISA_START_LAYER_IDX}" \
    "${ep_flag[@]}" "${save_flag[@]}"
done

echo ""
echo "=========================================================="
echo "Done! Results saved to: ${SAVE_ROOT}/{${SPLITS// /,}}"
echo "=========================================================="
