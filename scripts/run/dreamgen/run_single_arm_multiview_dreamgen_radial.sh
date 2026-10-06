#!/usr/bin/env bash
# =============================================================================
# run_single_arm_multiview_dreamgen_radial.sh
# DreamGen multiview (single_arm, droid layout) with Radial Attention.
# Exact mirror of the dicache flow; single-GPU; output -> sampling_dataset/radial/.
#
# Env: DATASET_ROOT, INPUT_DIR, OUTPUT_ROOT, CATEGORY, MODEL_SIZE, FPS,
#   MAX_EPISODES, MIN_EPISODE_INDEX, EPISODE_SOURCE, FORCE_REGENERATE_OUTPUTS,
#   RADIAL_{DECAY_FACTOR,BLOCK_SIZE,START_LAYER_IDX,MODEL_TYPE}
# =============================================================================
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "${ROOT}"

VENV="${VENV:-${ROOT}/models/dreamgen/.venv}"
_PYVER="$(basename "$(ls -d ${VENV}/lib/python3.* 2>/dev/null | head -1)")"
export LD_LIBRARY_PATH="${VENV}/lib/${_PYVER}/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}:${ROOT}/models/dreamgen:${PYTHONPATH:-}"

DATASET_ROOT="${DATASET_ROOT:-./datasets/single_arm/multiview}"
INPUT_DIR="${INPUT_DIR:-./sampling_dataset/radial/single_arm/input/multiview/dreamgen}"  # per-method (avoid concurrent-prepare races on shared dense json)
OUTPUT_ROOT="${OUTPUT_ROOT:-./sampling_dataset/radial/single_arm/output/multiview/dreamgen}"
CATEGORY_FILTER="${CATEGORY:-}"
MODEL_SIZE="${MODEL_SIZE:-14B}"
FPS="${FPS:-10}"
FORCE_REGENERATE_OUTPUTS="${FORCE_REGENERATE_OUTPUTS:-0}"
MAX_EPISODES="${MAX_EPISODES:-}"
MIN_EPISODE_INDEX="${MIN_EPISODE_INDEX:-}"
EPISODE_SOURCE="${EPISODE_SOURCE:-}"
NUM_GPUS=1

RADIAL_DECAY_FACTOR="${RADIAL_DECAY_FACTOR:-1.0}"
RADIAL_BLOCK_SIZE="${RADIAL_BLOCK_SIZE:-64}"
RADIAL_START_LAYER_IDX="${RADIAL_START_LAYER_IDX:-4}"
RADIAL_MODEL_TYPE="${RADIAL_MODEL_TYPE:-hunyuan}"

CATEGORIES=()
if [[ -n "${CATEGORY_FILTER}" ]]; then CATEGORIES+=("${CATEGORY_FILTER}"); else CATEGORIES=(makovian non_makovian); fi

echo "============================================================"
echo " DreamGen multiview — single_arm — Radial (single-GPU)"
echo " Model=${MODEL_SIZE} FPS=${FPS} | Radial decay=${RADIAL_DECAY_FACTOR} block=${RADIAL_BLOCK_SIZE} start=${RADIAL_START_LAYER_IDX} type=${RADIAL_MODEL_TYPE}"
echo " Output: ${OUTPUT_ROOT}"
echo "============================================================"

for CAT in "${CATEGORIES[@]}"; do
  DS_PATH="${DATASET_ROOT}/${CAT}"
  [[ -d "${DS_PATH}" ]] || { echo "[WARN] Skip missing: ${DS_PATH}"; continue; }
  [[ -f "${DS_PATH}/meta/episodes.jsonl" ]] || { echo "[WARN] No meta/episodes.jsonl in ${DS_PATH}, skip"; continue; }
  BATCH_JSON="${INPUT_DIR}/${CAT}/batch_input.json"
  OUTPUT_DIR="${OUTPUT_ROOT}/${CAT}"
  echo "========== ${CAT} =========="
  PREPARE_ARGS=(
    --dataset_path "${DS_PATH}" --input_dir "${INPUT_DIR}" --category "${CAT}"
    --batch_json "${BATCH_JSON}" --output_dir "${OUTPUT_DIR}" --skip_existing
  )
  [[ -n "${MAX_EPISODES}" ]] && PREPARE_ARGS+=(--max_episodes "${MAX_EPISODES}")
  [[ -n "${MIN_EPISODE_INDEX}" ]] && PREPARE_ARGS+=(--min_episode_index "${MIN_EPISODE_INDEX}")
  [[ -n "${EPISODE_SOURCE}" ]] && PREPARE_ARGS+=(--episode_source "${EPISODE_SOURCE}")
  [[ "${FORCE_REGENERATE_OUTPUTS}" == "1" ]] && PREPARE_ARGS+=(--force_regenerate_outputs)
  python "${ROOT}/scripts/prepare/prepare_single_arm_multiview_dreamgen_batch.py" "${PREPARE_ARGS[@]}"
  [[ -f "${BATCH_JSON}" ]] || { echo "[WARN] No batch at ${BATCH_JSON}, skip inference"; continue; }

  python -m examples.video2world_gr00t \
    --model_size "${MODEL_SIZE}" --gr00t_variant droid \
    --batch_input_json "${BATCH_JSON}" --prompt_prefix "" \
    --num_gpus "${NUM_GPUS}" --disable_guardrail --fps "${FPS}" \
    --use_radial --radial_decay_factor "${RADIAL_DECAY_FACTOR}" --radial_block_size "${RADIAL_BLOCK_SIZE}" \
    --radial_start_layer_idx "${RADIAL_START_LAYER_IDX}" --radial_model_type "${RADIAL_MODEL_TYPE}"
  echo ""
done

echo "[Done] Radial single_arm multiview — ${CATEGORIES[*]}"
