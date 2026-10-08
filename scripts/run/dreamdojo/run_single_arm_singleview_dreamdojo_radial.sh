#!/usr/bin/env bash
# DreamDojo single-arm singleview (DROID) inference with Radial Attention
# (log-decay block-sparse mask). Mirrors run_single_arm_singleview_dreamdojo_dense.sh
# exactly (same ckpt / dataset / inference script), only adding the Radial flags
# and routing output to radial/.
#
# NOTE: Radial needs flashinfer (BlockSparseAttentionWrapper).
#
# Output: sampling_dataset/radial/single_arm/output/singleview/dreamdojo/{split}/iter_*/episode_*/
#
# Usage:
#   bash scripts/run/dreamdojo/run_single_arm_singleview_dreamdojo_radial.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamdojo/run_single_arm_singleview_dreamdojo_radial.sh
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/checkpoints/finetuned/dreamdojo/dreamdojo/exp_single_arm_sv/single_arm_singleview/checkpoints}"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_single_arm_sv}"
NUM_FRAMES="${NUM_FRAMES:-97}"
_SAVE_ROOT_ENV="${SAVE_ROOT:-}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/radial/single_arm/output/singleview/dreamdojo}"
# Intermediate --output-dir: OUTPUT_ROOT if set, else <SAVE_ROOT>/_output when SAVE_ROOT
# was given (keeps scratch runs self-contained), else the historical ${ROOT}/output.
OUT_CACHE_ROOT="${OUTPUT_ROOT:-${_SAVE_ROOT_ENV:+${_SAVE_ROOT_ENV}/_output}}"
OUT_CACHE_ROOT="${OUT_CACHE_ROOT:-${ROOT}/output}"
NUM_EPISODES="${NUM_EPISODES:-}"
GUIDANCE="${GUIDANCE:-1.5}"
SPLITS="${SPLITS:-makovian non_makovian}"

# Radial parameters (tuned; see dreamdojo-single-arm-singleview-sparse-attention-insights)
RADIAL_DECAY_FACTOR="${RADIAL_DECAY_FACTOR:-0.5}"
RADIAL_BLOCK_SIZE="${RADIAL_BLOCK_SIZE:-64}"
# Tuned knee (1 makovian ep, traj_id=0, iter_000030000): start_layer_idx=2 / hunyuan /
# decay 0.5 -> 142.74s vs dense 144.80s (1.01x) at PSNR 27.40 / SSIM 0.918
# (== dense 27.57 / 0.920). decay 0.25 is the faster-but-lossy point (1.04x,
# PSNR 26.90). decay 1.0 (old default) is quality-safe but 0.96x (SLOWER).
RADIAL_START_LAYER_IDX="${RADIAL_START_LAYER_IDX:-2}"
RADIAL_MODEL_TYPE="${RADIAL_MODEL_TYPE:-hunyuan}"

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
  echo "=== DreamDojo Radial single-arm SV: ${split} -> ${save_base} ==="
  python "${ROOT}/scripts/infer/infer_single_arm_singleview_autoregressive.py" \
    --checkpoints-dir "${CHECKPOINTS_DIR}" \
    --experiment "${EXPERIMENT}" \
    --dataset-path "${dataset_dir}" \
    --save-dir "${save_base}" \
    --num-frames "${NUM_FRAMES}" \
    --save-fps 15 \
    --output-dir "${OUT_CACHE_ROOT}/dreamdojo_radial_single_arm_sv_${split}" \
    --guidance "${GUIDANCE}" \
    --use-radial \
    --radial-decay-factor "${RADIAL_DECAY_FACTOR}" \
    --radial-block-size "${RADIAL_BLOCK_SIZE}" \
    --radial-start-layer-idx "${RADIAL_START_LAYER_IDX}" \
    --radial-model-type "${RADIAL_MODEL_TYPE}" \
    "${ep_flag[@]}"
done

echo ""
echo "=========================================================="
echo "Done! Results saved to: ${SAVE_ROOT}"
echo "=========================================================="
