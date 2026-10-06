#!/usr/bin/env bash
# DreamDojo inference with FasterCache acceleration on bimanual (ALOHA) singleview.
# Aligned with scripts/run/dreamdojo/run_bimanual_singleview_dreamdojo_worldcache.sh:
#   - same inference script (infer_bimanual_singleview_single_segment.py, pure autoregressive)
#   - same chunk-size / save-fps / output layout / checkpoint resolution
#   - episodes named by REAL trajectory_id (matches dense / worldcache / dicache)
#   - only difference: FasterCache acceleration enabled.
# Runs datasets/bimanual/singleview/makovian and .../non_makovian separately.
# Output: sampling_dataset/fastercache/bimanual/output/singleview/dreamdojo/{split}/iter_*/episode_*/
#
# Usage:
#   bash scripts/run/dreamdojo/run_bimanual_singleview_dreamdojo_fastercache.sh
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run/dreamdojo/run_bimanual_singleview_dreamdojo_fastercache.sh
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/checkpoints/finetuned/dreamdojo/dreamdojo/exp_aloha_resume70k/aloha_bimanual_singleview_full/checkpoints}"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_aloha_resume70k_bimanual_sv}"
CHUNK_SIZE="${CHUNK_SIZE:-12}"
SAVE_FPS="${SAVE_FPS:-10}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/fastercache/bimanual/output/singleview/dreamdojo}"
NUM_EPISODES="${NUM_EPISODES:-}"
GUIDANCE="${GUIDANCE:-0}"

# FasterCache parameters
FASTERCACHE_START_STEP="${FASTERCACHE_START_STEP:-0}"
FASTERCACHE_MODEL_INTERVAL="${FASTERCACHE_MODEL_INTERVAL:-5}"
FASTERCACHE_BLOCK_INTERVAL="${FASTERCACHE_BLOCK_INTERVAL:-3}"

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

for split in makovian non_makovian; do
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
  echo "=== DreamDojo FasterCache bimanual SV: ${split} -> ${save_base} ==="
  # SAVE_FULL=1 adds --save-full (PSNR/SSIM/LPIPS). The sparse-attention scripts
  # for this cell already pass it, so without this the cache methods would be the
  # only ones with no quality numbers to pair with their speedup.
  full_flag=()
  if [ "${SAVE_FULL:-0}" = "1" ]; then full_flag=(--save-full); fi
  python "${ROOT}/scripts/infer/infer_bimanual_singleview_single_segment.py" \
    --checkpoints-dir "${CHECKPOINTS_DIR}" \
    --experiment "${EXPERIMENT}" \
    --dataset-path "${dataset_paths}" \
    --save-dir "${save_base}" \
    --chunk-size "${CHUNK_SIZE}" \
    --save-fps "${SAVE_FPS}" \
    --output-dir "${ROOT}/output/dreamdojo_fastercache_bimanual_sv_${split}" \
    --guidance "${GUIDANCE}" \
    --fastercache-enabled \
    --fastercache-start-step "${FASTERCACHE_START_STEP}" \
    --fastercache-model-interval "${FASTERCACHE_MODEL_INTERVAL}" \
    --fastercache-block-interval "${FASTERCACHE_BLOCK_INTERVAL}" \
    "${full_flag[@]}" \
    "${ep_flag[@]}"
done
