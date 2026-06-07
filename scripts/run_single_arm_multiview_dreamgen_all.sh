#!/usr/bin/env bash
# =============================================================================
# run_single_arm_multiview_dreamgen_all.sh
# Run prepare + DreamGen for both single_arm multiview categories (makovian,
# non_makovian) under DATASET_ROOT.
#
# Usage (repo root):
#   NUM_GPUS=2 bash scripts/run_single_arm_multiview_dreamgen_all.sh
#
# Environment:
#   DATASET_ROOT   default ./datasets/single_arm/multiview
#   INPUT_DIR      default ./sampling_dataset/dense/single_arm/input/multiview/dreamgen
#   CATEGORY       empty = both; or makovian | non_makovian only
#   MAX_EPISODES, MIN_EPISODE_INDEX, EPISODE_SOURCE (meta|disk, default disk in Python)
#   USE_LVG, NUM_CHUNKS   (longer temporal output via gr00t_lvg)
#   FORCE_REGENERATE_OUTPUTS=1  include episodes whose output mp4 already exists
#   PREPARE_ONLY=1 / SKIP_PREPARE=1  same semantics as bimanual all-runner
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DATASET_ROOT="${DATASET_ROOT:-./datasets/single_arm/multiview}"
INPUT_DIR="${INPUT_DIR:-./sampling_dataset/dense/single_arm/input/multiview/dreamgen}"
CATEGORY_FILTER="${CATEGORY:-}"
MODEL_SIZE="${MODEL_SIZE:-14B}"
NUM_GPUS="${NUM_GPUS:-2}"
MASTER_PORT="${MASTER_PORT:-12341}"
FPS="${FPS:-10}"
USE_LVG="${USE_LVG:-0}"
NUM_CHUNKS="${NUM_CHUNKS:-6}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
FORCE_REGENERATE_OUTPUTS="${FORCE_REGENERATE_OUTPUTS:-0}"
MAX_EPISODES="${MAX_EPISODES:-}"
MIN_EPISODE_INDEX="${MIN_EPISODE_INDEX:-}"
EPISODE_SOURCE="${EPISODE_SOURCE:-}"

CATEGORIES=()
if [[ -n "${CATEGORY_FILTER}" ]]; then
  CATEGORIES+=("${CATEGORY_FILTER}")
else
  CATEGORIES=(makovian non_makovian)
fi

echo "============================================================"
echo " DreamGen multiview — single_arm (all categories)"
echo "============================================================"
echo " Data root: ${DATASET_ROOT}"
echo " Categories: ${CATEGORIES[*]}"
echo " Model: ${MODEL_SIZE}  GPUs: ${NUM_GPUS}  USE_LVG: ${USE_LVG}"
echo " PREPARE_ONLY: ${PREPARE_ONLY}  SKIP_PREPARE: ${SKIP_PREPARE}"
echo "============================================================"

source "${REPO_ROOT}/models/dreamgen/.venv/bin/activate"
export PYTHONPATH="${REPO_ROOT}/models/dreamgen${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${PREPARE_ONLY}" != "1" ]]; then
  AVAILABLE_GPUS=$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
  )
  if [[ "${AVAILABLE_GPUS}" -lt 1 ]]; then
    echo "[ERROR] No CUDA GPUs visible." >&2
    exit 1
  fi
  if [[ "${NUM_GPUS}" -gt "${AVAILABLE_GPUS}" ]]; then
    echo "[ERROR] NUM_GPUS=${NUM_GPUS} > available ${AVAILABLE_GPUS}" >&2
    exit 1
  fi
fi

for CAT in "${CATEGORIES[@]}"; do
  DS_PATH="${DATASET_ROOT}/${CAT}"
  if [[ ! -d "${DS_PATH}" ]]; then
    echo "[WARN] Skip missing: ${DS_PATH}"
    continue
  fi
  if [[ ! -f "${DS_PATH}/meta/episodes.jsonl" ]]; then
    echo "[WARN] No meta/episodes.jsonl in ${DS_PATH}, skip"
    continue
  fi

  echo ""
  echo "========== ${CAT} =========="
  export DATASET_PATH="${DS_PATH}"
  export CATEGORY="${CAT}"
  export INPUT_DIR
  export MODEL_SIZE NUM_GPUS MASTER_PORT FPS USE_LVG NUM_CHUNKS SKIP_PREPARE FORCE_REGENERATE_OUTPUTS
  if [[ -n "${MAX_EPISODES}" ]]; then
    export MAX_EPISODES
  else
    unset MAX_EPISODES 2>/dev/null || true
  fi
  if [[ -n "${MIN_EPISODE_INDEX}" ]]; then
    export MIN_EPISODE_INDEX
  else
    unset MIN_EPISODE_INDEX 2>/dev/null || true
  fi
  if [[ -n "${EPISODE_SOURCE}" ]]; then
    export EPISODE_SOURCE
  else
    unset EPISODE_SOURCE 2>/dev/null || true
  fi

  if [[ "${PREPARE_ONLY}" == "1" ]]; then
    BATCH_JSON="${INPUT_DIR}/${CAT}/batch_input.json"
    PREPARE_ARGS=(
      --dataset_path "${DS_PATH}"
      --input_dir    "${INPUT_DIR}"
      --category     "${CAT}"
      --batch_json   "${BATCH_JSON}"
      --skip_existing
    )
    if [[ -n "${MAX_EPISODES:-}" ]]; then
      PREPARE_ARGS+=(--max_episodes "${MAX_EPISODES}")
    fi
    if [[ -n "${MIN_EPISODE_INDEX:-}" ]]; then
      PREPARE_ARGS+=(--min_episode_index "${MIN_EPISODE_INDEX}")
    fi
    if [[ -n "${EPISODE_SOURCE:-}" ]]; then
      PREPARE_ARGS+=(--episode_source "${EPISODE_SOURCE}")
    fi
    if [[ "${FORCE_REGENERATE_OUTPUTS}" == "1" ]]; then
      PREPARE_ARGS+=(--force_regenerate_outputs)
    fi
    python "${REPO_ROOT}/scripts/prepare_single_arm_multiview_dreamgen_batch.py" "${PREPARE_ARGS[@]}"
    continue
  fi

  bash "${REPO_ROOT}/scripts/run_single_arm_multiview_dreamgen_batch.sh"
done

echo ""
echo "[All done] single_arm multiview"
