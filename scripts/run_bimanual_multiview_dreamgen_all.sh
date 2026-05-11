#!/usr/bin/env bash
# =============================================================================
# run_bimanual_multiview_dreamgen_all.sh
# For each task under datasets/bimanual/multiview/{makovian,non_makovian},
# prepare DreamGen batch inputs and run video2world_gr00t (droid).
#
# Usage (from repo root):
#   bash scripts/run_bimanual_multiview_dreamgen_all.sh
#
# Environment:
#   MULTIVIEW_ROOT  default ./datasets/bimanual/multiview
#   CATEGORY        empty = both; or makovian | non_makovian only
#   INPUT_DIR       default ./sampling_dataset/bimanual/input/multiview/dreamgen
#   FOUR_VIEWS      set to 1 for four camera quadrants
#   DYNAMIC_WRIST   left|right when FOUR_VIEWS unset
#   MAX_EPISODES, NUM_GPUS, MODEL_SIZE, MASTER_PORT, FPS
#   SKIP_PREPARE    set to 1 to only run inference (expects batch_input.json already)
#   PREPARE_ONLY    set to 1 to only build grids + batch JSON for every task (no GPU)
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MULTIVIEW_ROOT="${MULTIVIEW_ROOT:-./datasets/bimanual/multiview}"
INPUT_DIR="${INPUT_DIR:-./sampling_dataset/bimanual/input/multiview/dreamgen}"
CATEGORY_FILTER="${CATEGORY:-}"
FOUR_VIEWS="${FOUR_VIEWS:-0}"
DYNAMIC_WRIST="${DYNAMIC_WRIST:-left}"
MODEL_SIZE="${MODEL_SIZE:-14B}"
NUM_GPUS="${NUM_GPUS:-2}"
MASTER_PORT="${MASTER_PORT:-12341}"
FPS="${FPS:-10}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"

CATEGORIES=()
if [[ -n "${CATEGORY_FILTER}" ]]; then
  CATEGORIES+=("${CATEGORY_FILTER}")
else
  CATEGORIES=(makovian non_makovian)
fi

echo "============================================================"
echo " DreamGen multiview — all bimanual tasks"
echo "============================================================"
echo " Data root: ${MULTIVIEW_ROOT}"
echo " Categories: ${CATEGORIES[*]}"
echo " Input dir: ${INPUT_DIR}"
echo " Model: ${MODEL_SIZE}  GPUs: ${NUM_GPUS}  FPS: ${FPS}"
echo " PREPARE_ONLY: ${PREPARE_ONLY}  SKIP_PREPARE: ${SKIP_PREPARE}"
echo "============================================================"
echo ""

source "${REPO_ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${REPO_ROOT}/models/dreamgen${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${PREPARE_ONLY}" != "1" ]]; then
  AVAILABLE_GPUS=$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
  )

  if [[ "${AVAILABLE_GPUS}" -lt 1 ]]; then
    echo "[ERROR] No CUDA GPUs visible to PyTorch." >&2
    exit 1
  fi
  if [[ "${NUM_GPUS}" -gt "${AVAILABLE_GPUS}" ]]; then
    echo "[ERROR] NUM_GPUS=${NUM_GPUS} but only ${AVAILABLE_GPUS} GPU(s) available." >&2
    exit 1
  fi
fi

for CAT in "${CATEGORIES[@]}"; do
  CAT_DIR="${MULTIVIEW_ROOT}/${CAT}"
  if [[ ! -d "${CAT_DIR}" ]]; then
    echo "[WARN] Missing directory, skipping: ${CAT_DIR}"
    continue
  fi

  while IFS= read -r -d '' TASK_DIR; do
    if [[ ! -f "${TASK_DIR}/meta/episodes.jsonl" ]]; then
      echo "[WARN] No meta/episodes.jsonl in ${TASK_DIR}, skip"
      continue
    fi

    TASK_NAME="$(basename "${TASK_DIR}")"
    BATCH_JSON="${INPUT_DIR}/${CAT}/${TASK_NAME}/batch_input.json"

    echo "============================================================"
    echo " Task: ${CAT}/${TASK_NAME}"
    echo "============================================================"

    if [[ "${SKIP_PREPARE}" != "1" ]]; then
      PREPARE_ARGS=(
        --dataset_path "${TASK_DIR}"
        --input_dir    "${INPUT_DIR}"
        --category     "${CAT}"
        --batch_json   "${BATCH_JSON}"
      )
      if [[ -n "${OUTPUT_DIR_ROOT:-}" ]]; then
        PREPARE_ARGS+=(--output_dir "${OUTPUT_DIR_ROOT}/${CAT}/${TASK_NAME}")
      fi
      if [[ "${FOUR_VIEWS}" == "1" ]]; then
        PREPARE_ARGS+=(--four_views)
      fi
      PREPARE_ARGS+=(--dynamic_wrist "${DYNAMIC_WRIST}")
      if [[ -n "${MAX_EPISODES:-}" ]]; then
        PREPARE_ARGS+=(--max_episodes "${MAX_EPISODES}")
      fi
      PREPARE_ARGS+=(--skip_existing)

      python "${REPO_ROOT}/scripts/prepare_bimanual_multiview_dreamgen_batch.py" "${PREPARE_ARGS[@]}"
    fi

    if [[ "${PREPARE_ONLY}" == "1" ]]; then
      echo ""
      continue
    fi

    if [[ ! -f "${BATCH_JSON}" ]]; then
      echo "[WARN] No batch at ${BATCH_JSON}, skip inference"
      continue
    fi

    INFERENCE_ARGS=(
      -m examples.video2world_gr00t
      --model_size       "${MODEL_SIZE}"
      --gr00t_variant    droid
      --batch_input_json "${BATCH_JSON}"
      --prompt_prefix    ""
      --num_gpus         "${NUM_GPUS}"
      --disable_guardrail
      --fps              "${FPS}"
    )

    if [[ "${NUM_GPUS}" -gt 1 ]]; then
      torchrun \
        --nproc_per_node="${NUM_GPUS}" \
        --master_port="${MASTER_PORT}" \
        "${INFERENCE_ARGS[@]}"
    else
      python "${INFERENCE_ARGS[@]}"
    fi

    echo ""
  done < <(find "${CAT_DIR}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
done

echo "[Done] All tasks in categories: ${CATEGORIES[*]}"
