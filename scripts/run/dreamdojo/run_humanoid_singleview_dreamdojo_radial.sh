#!/usr/bin/env bash
# DreamDojo full-episode inference with Radial Attention (log-decay block-sparse mask).
# Mirrors run_humanoid_singleview_dreamdojo_full_episode.sh exactly (same ckpt / dataset /
# inference script), only adding the Radial flags and routing output to radial/.
#
# NOTE: Radial needs flashinfer (BlockSparseAttentionWrapper).
#
# Output: sampling_dataset/radial/humanoid/output/singleview/dreamdojo/{split}/iter_*/episode_*/
#
# Usage:
#   bash scripts/run/dreamdojo/run_humanoid_singleview_dreamdojo_radial.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamdojo/run_humanoid_singleview_dreamdojo_radial.sh
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/checkpoints/base/dreamdojo/2B_GR1_post-train/iter_000050000}"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_gr1}"
NUM_FRAMES="${NUM_FRAMES:-49}"
_SAVE_ROOT_ENV="${SAVE_ROOT:-}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/radial/humanoid/output/singleview/dreamdojo}"
# Intermediate --output-dir: OUTPUT_ROOT if set, else <SAVE_ROOT>/_output when SAVE_ROOT
# was given (keeps scratch runs self-contained), else the historical ${ROOT}/output.
OUT_CACHE_ROOT="${OUTPUT_ROOT:-${_SAVE_ROOT_ENV:+${_SAVE_ROOT_ENV}/_output}}"
OUT_CACHE_ROOT="${OUT_CACHE_ROOT:-${ROOT}/output}"
NUM_EPISODES="${NUM_EPISODES:-}"
GUIDANCE="${GUIDANCE:-1.5}"
SPLITS="${SPLITS:-makovian non_makovian}"
SAVE_MODE="${SAVE_MODE:-video_only}"

# Radial parameters (tuned defaults; see dreamdojo-humanoid-singleview-sparse-attention rule)
RADIAL_DECAY_FACTOR="${RADIAL_DECAY_FACTOR:-1.0}"
RADIAL_BLOCK_SIZE="${RADIAL_BLOCK_SIZE:-64}"
# Tuned knee (1 makovian ep, iter_000050000): start_layer_idx=2 / hunyuan / decay 1.0
# -> 54.78s vs dense 56.41s (1.03x) at PSNR 19.32 / SSIM 0.747 (== dense 19.24 / 0.743).
# decay_factor <= 0.5 is the BREAK point: quality drops (PSNR 18.35 @0.5, 17.91 @0.25)
# AND it gets SLOWER than dense (0.96x / 0.98x) because mask-build cost grows.
RADIAL_START_LAYER_IDX="${RADIAL_START_LAYER_IDX:-2}"
RADIAL_MODEL_TYPE="${RADIAL_MODEL_TYPE:-hunyuan}"

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
  echo "=== DreamDojo Radial full-episode: ${split} -> ${save_base} ==="
  python "${ROOT}/scripts/infer/infer_humanoid_singleview_full_episode.py" \
    --checkpoints-dir "${CHECKPOINTS_DIR}" \
    --experiment "${EXPERIMENT}" \
    --dataset-path "${dataset_dir}" \
    --save-dir "${save_base}" \
    --num-frames "${NUM_FRAMES}" \
    --output-dir "${OUT_CACHE_ROOT}/dreamdojo_radial_humanoid_sv_${split}" \
    --guidance "${GUIDANCE}" \
    --use-radial \
    --radial-decay-factor "${RADIAL_DECAY_FACTOR}" \
    --radial-block-size "${RADIAL_BLOCK_SIZE}" \
    --radial-start-layer-idx "${RADIAL_START_LAYER_IDX}" \
    --radial-model-type "${RADIAL_MODEL_TYPE}" \
    "${ep_flag[@]}" "${save_flag[@]}"
done

echo ""
echo "=========================================================="
echo "Done! Results saved to: ${SAVE_ROOT}"
echo "=========================================================="
