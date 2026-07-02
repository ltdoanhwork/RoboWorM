#!/usr/bin/env bash
# =============================================================================
# DreamDojo humanoid singleview — full benchmark run over ALL acceleration
# methods in a single script: dense, worldcache, fastercache, dicache.
#
# Mirrors scripts/run_bimanual_singleview_dreamdojo_all_methods.sh:
#   - all methods share the SAME inference script
#     (scripts/infer_humanoid_singleview_full_episode.py, autoregressive rollout),
#   - same checkpoint / experiment / num-frames / save-fps,
#   - same episode discovery, only cache flags + output dir differ per method.
#
# Episode naming matches DreamGen humanoid singleview: episode_{id:06d}/full_pred.mp4
# where id == the REAL trajectory_id (NOT the enumeration index). Humanoid is a
# single LeRobot dataset (no task subdirs), so no task prefix is needed and there
# is no trajectory_id collision — unlike bimanual which needs {task}__ prefix.
#
# Only the generated prediction video is saved (--save-video-only, default on).
#
# Output layout (aligned with sampling-dataset-layout rule):
#   sampling_dataset/{method}/humanoid/output/singleview/dreamdojo/{split}/iter_*/episode_*/full_pred.mp4
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_humanoid_singleview_dreamdojo_all_methods.sh
#   METHODS="dense worldcache" bash scripts/run_humanoid_singleview_dreamdojo_all_methods.sh
#   SPLITS="makovian" NUM_EPISODES=5 bash scripts/run_humanoid_singleview_dreamdojo_all_methods.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# --- Shared inference config (identical across methods) ---
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/checkpoints/dreamdojo/2B_GR1_post-train/iter_000050000}"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_gr1}"
NUM_FRAMES="${NUM_FRAMES:-49}"
SAVE_FPS="${SAVE_FPS:-10}"
GUIDANCE="${GUIDANCE:-1.5}"
NUM_EPISODES="${NUM_EPISODES:-}"

METHODS="${METHODS:-dense worldcache fastercache dicache}"
SPLITS="${SPLITS:-makovian non_makovian}"

INFER="${ROOT}/scripts/infer_humanoid_singleview_full_episode.py"

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
  SAVE_ROOT="${ROOT}/sampling_dataset/${method}/humanoid/output/singleview/dreamdojo"
  echo "############################################################"
  echo "# METHOD: ${method}"
  echo "# SAVE_ROOT: ${SAVE_ROOT}"
  echo "############################################################"

  # Read method flags into an array (one flag per line).
  mapfile -t cache_flags < <(method_flags "${method}")

  for split in ${SPLITS}; do
    dataset_dir="${ROOT}/datasets/humanoid/singleview/${split}"
    if [ ! -d "${dataset_dir}/data" ]; then
      echo "Skipping ${split}: no data directory found at ${dataset_dir}/data" >&2
      continue
    fi

    save_base="${SAVE_ROOT}/${split}"
    mkdir -p "${save_base}"
    echo "=== DreamDojo ${method} humanoid SV: ${split} -> ${save_base} ==="
    python "${INFER}" \
      --checkpoints-dir "${CHECKPOINTS_DIR}" \
      --experiment "${EXPERIMENT}" \
      --dataset-path "${dataset_dir}" \
      --save-dir "${save_base}" \
      --num-frames "${NUM_FRAMES}" \
      --save-fps "${SAVE_FPS}" \
      --output-dir "${ROOT}/output/dreamdojo_${method}_humanoid_sv_${split}" \
      --guidance "${GUIDANCE}" \
      --save-video-only \
      "${cache_flags[@]}" \
      "${ep_flag[@]}"
  done
done

echo "All methods complete: ${METHODS}"
