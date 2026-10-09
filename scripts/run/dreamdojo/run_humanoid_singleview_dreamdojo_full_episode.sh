#!/usr/bin/env bash
# Full-episode inference for humanoid singleview DreamDojo.
# Each episode is split into non-overlapping segments of NUM_FRAMES frames.
# Each segment starts from the GT frame at that position → no error accumulation.
# Episodes shorter than NUM_FRAMES are skipped.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamdojo/run_humanoid_singleview_dreamdojo_full_episode.sh
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/checkpoints/base/dreamdojo/2B_GR1_post-train/iter_000050000}"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_gr1}"
NUM_FRAMES="${NUM_FRAMES:-49}"
_SAVE_ROOT_ENV="${SAVE_ROOT:-}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/dense/humanoid/output/singleview/dreamdojo}"
# Intermediate --output-dir: OUTPUT_ROOT if set, else <SAVE_ROOT>/_output when SAVE_ROOT
# was given (keeps scratch runs self-contained), else the historical ${ROOT}/output.
OUT_CACHE_ROOT="${OUTPUT_ROOT:-${_SAVE_ROOT_ENV:+${_SAVE_ROOT_ENV}/_output}}"
OUT_CACHE_ROOT="${OUT_CACHE_ROOT:-${ROOT}/output}"
NUM_EPISODES="${NUM_EPISODES:-}"
# Guidance of the dense reference. 0 = the infer-script default this launcher has
# always used (run_config.json of the released rollouts: "guidance": 0). Every
# method launcher of this cell reads the same GUIDANCE variable with the same default.
GUIDANCE="${GUIDANCE:-0}"

ep_flag=()
if [[ -n "${NUM_EPISODES}" ]]; then
  ep_flag=(--num-episodes "${NUM_EPISODES}")
fi

for split in makovian non_makovian; do
  dataset_dir="${ROOT}/datasets/humanoid/singleview/${split}"
  if [ ! -d "${dataset_dir}/data" ]; then
    echo "Skipping ${split}: no data directory found at ${dataset_dir}/data" >&2
    continue
  fi
  save_base="${SAVE_ROOT}/${split}"
  mkdir -p "${save_base}"
  echo "=== Full-episode inference: ${split} -> ${save_base} ==="
  # SAVE_FULL=1 adds --save-full, which is what makes the script emit
  # PSNR/SSIM/LPIPS. Off by default so the canonical bulk runs stay video-only
  # (that is why sampling_dataset has 216 humanoid mp4s but no metrics.json);
  # the timing pass turns it on so speed and quality come from the same run.
  full_flag=()
  if [ "${SAVE_FULL:-0}" = "1" ]; then full_flag=(--save-full); fi
  # GT_ONLY=1 backfills full_gt.mp4 next to predictions that were written
  # video-only, without running diffusion -- needed because the reference-based
  # (vs_gt) half of the DR grid has no reference for this cell otherwise. It
  # loads no checkpoint and needs no GPU.
  if [ "${GT_ONLY:-0}" = "1" ]; then full_flag=(--gt-only); fi
  python "${ROOT}/scripts/infer/infer_humanoid_singleview_full_episode.py" \
    --checkpoints-dir "${CHECKPOINTS_DIR}" \
    --experiment "${EXPERIMENT}" \
    --dataset-path "${dataset_dir}" \
    --save-dir "${save_base}" \
    --num-frames "${NUM_FRAMES}" \
    --output-dir "${OUT_CACHE_ROOT}/dreamdojo_full_ep_humanoid_sv_${split}" \
    --guidance "${GUIDANCE}" \
    "${full_flag[@]}" \
    "${ep_flag[@]}"
done
