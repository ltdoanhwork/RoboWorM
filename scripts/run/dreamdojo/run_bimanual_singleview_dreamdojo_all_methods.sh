#!/usr/bin/env bash
# =============================================================================
# DreamDojo bimanual singleview — full benchmark run over ALL acceleration
# methods in a single script: dense, worldcache, fastercache, dicache.
#
# All methods share the SAME inference script
# (scripts/infer/infer_bimanual_singleview_single_segment.py, pure autoregressive),
# same checkpoint / experiment / chunk-size / save-fps, and the SAME episode
# discovery (every task dir under datasets/bimanual/singleview/{split}).
# The only per-method difference is the cache flags + the top-level output dir.
#
# Episode naming matches DreamGen singleview: {task}__episode_{id:06d}/full_pred.mp4
# so every episode (150 makovian) is kept — no trajectory_id collision.
#
# Only the generated prediction video is saved (--save-video-only, default on).
#
# Output layout (aligned with sampling-dataset-layout rule):
#   sampling_dataset/{method}/bimanual/output/singleview/dreamdojo/{split}/iter_*/{task}__episode_*/full_pred.mp4
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run/dreamdojo/run_bimanual_singleview_dreamdojo_all_methods.sh
#   METHODS="dense worldcache" bash scripts/run/dreamdojo/run_bimanual_singleview_dreamdojo_all_methods.sh
#   SPLITS="makovian" NUM_EPISODES=5 bash scripts/run/dreamdojo/run_bimanual_singleview_dreamdojo_all_methods.sh
# =============================================================================
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"
# Intermediate --output-dir root (override with OUTPUT_ROOT).
OUT_CACHE_ROOT="${OUTPUT_ROOT:-${ROOT}/output}"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# --- Shared inference config (identical across methods) ---
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/checkpoints/finetuned/dreamdojo/dreamdojo/exp_aloha_resume70k/aloha_bimanual_singleview_full/checkpoints}"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_aloha_resume70k_bimanual_sv}"
CHUNK_SIZE="${CHUNK_SIZE:-12}"
SAVE_FPS="${SAVE_FPS:-10}"
GUIDANCE="${GUIDANCE:-0}"
NUM_EPISODES="${NUM_EPISODES:-}"

METHODS="${METHODS:-dense worldcache fastercache dicache}"
SPLITS="${SPLITS:-makovian non_makovian}"

INFER="${ROOT}/scripts/infer/infer_bimanual_singleview_single_segment.py"

# --- WorldCache params ---
WORLDCACHE_NUM_STEPS="${WORLDCACHE_NUM_STEPS:-35}"
WORLDCACHE_REL_L1_THRESH="${WORLDCACHE_REL_L1_THRESH:-0.03}"
WORLDCACHE_RET_RATIO="${WORLDCACHE_RET_RATIO:-0.4}"
WORLDCACHE_PROBE_DEPTH="${WORLDCACHE_PROBE_DEPTH:-4}"
WORLDCACHE_MOTION_SENSITIVITY="${WORLDCACHE_MOTION_SENSITIVITY:-5.0}"

# --- FasterCache params ---
FASTERCACHE_START_STEP="${FASTERCACHE_START_STEP:-0}"
FASTERCACHE_MODEL_INTERVAL="${FASTERCACHE_MODEL_INTERVAL:-5}"
FASTERCACHE_BLOCK_INTERVAL="${FASTERCACHE_BLOCK_INTERVAL:-3}"

# --- DiCache params ---
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

# Emit the method-specific cache flags for the inference script.
method_flags() {
  local method="$1"
  case "${method}" in
    dense)
      ;;  # no cache flags
    worldcache)
      printf '%s\n' \
        --worldcache-enabled \
        --worldcache-num-steps "${WORLDCACHE_NUM_STEPS}" \
        --worldcache-rel-l1-thresh "${WORLDCACHE_REL_L1_THRESH}" \
        --worldcache-ret-ratio "${WORLDCACHE_RET_RATIO}" \
        --worldcache-probe-depth "${WORLDCACHE_PROBE_DEPTH}" \
        --worldcache-motion-sensitivity "${WORLDCACHE_MOTION_SENSITIVITY}"
      ;;
    fastercache)
      printf '%s\n' \
        --fastercache-enabled \
        --fastercache-start-step "${FASTERCACHE_START_STEP}" \
        --fastercache-model-interval "${FASTERCACHE_MODEL_INTERVAL}" \
        --fastercache-block-interval "${FASTERCACHE_BLOCK_INTERVAL}"
      ;;
    dicache)
      printf '%s\n' \
        --dicache-enabled \
        --dicache-num-steps "${DICACHE_NUM_STEPS}" \
        --dicache-rel-l1-thresh "${DICACHE_REL_L1_THRESH}" \
        --dicache-ret-ratio "${DICACHE_RET_RATIO}" \
        --dicache-probe-depth "${DICACHE_PROBE_DEPTH}"
      ;;
    *)
      echo "Unknown method: ${method}" >&2
      return 1
      ;;
  esac
}

ep_flag=()
if [[ -n "${NUM_EPISODES}" ]]; then
  ep_flag=(--num-episodes "${NUM_EPISODES}")
fi

for method in ${METHODS}; do
  SAVE_ROOT="${ROOT}/sampling_dataset/${method}/bimanual/output/singleview/dreamdojo"
  echo "############################################################"
  echo "# METHOD: ${method}"
  echo "# SAVE_ROOT: ${SAVE_ROOT}"
  echo "############################################################"

  # Read method flags into an array (one flag per line).
  mapfile -t cache_flags < <(method_flags "${method}")

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
    echo "=== DreamDojo ${method} bimanual SV: ${split} -> ${save_base} ==="
    python "${INFER}" \
      --checkpoints-dir "${CHECKPOINTS_DIR}" \
      --experiment "${EXPERIMENT}" \
      --dataset-path "${dataset_paths}" \
      --save-dir "${save_base}" \
      --chunk-size "${CHUNK_SIZE}" \
      --save-fps "${SAVE_FPS}" \
      --output-dir "${OUT_CACHE_ROOT}/dreamdojo_${method}_bimanual_sv_${split}" \
      --guidance "${GUIDANCE}" \
      --save-video-only \
      "${cache_flags[@]}" \
      "${ep_flag[@]}"
  done
done

echo "All methods complete: ${METHODS}"
