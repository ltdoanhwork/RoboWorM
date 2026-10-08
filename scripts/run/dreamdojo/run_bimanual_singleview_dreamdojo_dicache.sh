#!/usr/bin/env bash
# DreamDojo full-episode inference with DiCache acceleration on bimanual (ALOHA) singleview.
# Aligned with scripts/run/dreamdojo/run_bimanual_singleview_dreamdojo_full_episode.sh:
#   - same inference script (infer_bimanual_singleview_single_segment.py, pure autoregressive)
#   - same chunk-size / save-fps / output layout / checkpoint resolution
#   - only difference: DiCache acceleration enabled.
# Runs datasets/bimanual/singleview/makovian and .../non_makovian separately.
# Output: sampling_dataset/dicache/bimanual/output/singleview/dreamdojo/{split}/iter_*/episode_*/
#
# Usage:
#   bash scripts/run/dreamdojo/run_bimanual_singleview_dreamdojo_dicache.sh
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run/dreamdojo/run_bimanual_singleview_dreamdojo_dicache.sh
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
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/dicache/bimanual/output/singleview/dreamdojo}"
# Intermediate --output-dir: OUTPUT_ROOT if set, else <SAVE_ROOT>/_output when SAVE_ROOT
# was given (keeps scratch runs self-contained), else the historical ${ROOT}/output.
OUT_CACHE_ROOT="${OUTPUT_ROOT:-${_SAVE_ROOT_ENV:+${_SAVE_ROOT_ENV}/_output}}"
OUT_CACHE_ROOT="${OUT_CACHE_ROOT:-${ROOT}/output}"
NUM_EPISODES="${NUM_EPISODES:-}"
GUIDANCE="${GUIDANCE:-0}"

# DiCache parameters
DICACHE_NUM_STEPS="${DICACHE_NUM_STEPS:-35}"
DICACHE_REL_L1_THRESH="${DICACHE_REL_L1_THRESH:-0.08}"
DICACHE_RET_RATIO="${DICACHE_RET_RATIO:-0.2}"
DICACHE_PROBE_DEPTH="${DICACHE_PROBE_DEPTH:-2}"

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
  echo "=== DreamDojo DiCache bimanual SV: ${split} -> ${save_base} ==="
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
    --output-dir "${OUT_CACHE_ROOT}/dreamdojo_dicache_bimanual_sv_${split}" \
    --guidance "${GUIDANCE}" \
    --dicache-enabled \
    --dicache-num-steps "${DICACHE_NUM_STEPS}" \
    --dicache-rel-l1-thresh "${DICACHE_REL_L1_THRESH}" \
    --dicache-ret-ratio "${DICACHE_RET_RATIO}" \
    --dicache-probe-depth "${DICACHE_PROBE_DEPTH}" \
    "${full_flag[@]}" \
    "${ep_flag[@]}"
done
