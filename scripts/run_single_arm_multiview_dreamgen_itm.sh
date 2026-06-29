#!/usr/bin/env bash
# =============================================================================
# run_single_arm_multiview_dreamgen_itm.sh
# DreamGen multiview (single_arm, droid layout) with ITM token-merge acceleration.
#
# Mirrors the dense run (run_single_arm_multiview_dreamgen_batch.sh) but routes
# outputs to the itm/ method tree and enables Importance Token Merge.
#
# ITM is SINGLE-GPU only (no context parallelism), so this script forces
# NUM_GPUS=1 and calls `python -m examples.video2world_gr00t` directly.
#
# ITM needs classifier-free guidance — the gr00t pipeline already runs with
# guidance > 0, so no extra flag is required here.
#
# Runs both categories (makovian, non_makovian). Outputs (method = top-level dir):
#   sampling_dataset/itm/single_arm/output/multiview/dreamgen/<cat>/
#
# Usage (repo root):
#   bash scripts/run_single_arm_multiview_dreamgen_itm.sh
#   CUDA_VISIBLE_DEVICES=1 MAX_EPISODES=2 bash scripts/run_single_arm_multiview_dreamgen_itm.sh
#   CATEGORY=makovian bash scripts/run_single_arm_multiview_dreamgen_itm.sh
#
# Environment:
#   DATASET_ROOT   default ./datasets/single_arm/multiview
#   INPUT_DIR      default ./sampling_dataset/dense/single_arm/input/multiview/dreamgen
#   OUTPUT_ROOT    default ./sampling_dataset/itm/single_arm/output/multiview/dreamgen
#   CATEGORY       empty = both; or makovian | non_makovian only
#   MODEL_SIZE, FPS, MAX_EPISODES, MIN_EPISODE_INDEX, EPISODE_SOURCE, FORCE_REGENERATE_OUTPUTS=1
#   ITM_* overrides (see defaults below; 14B is a 36-block model)
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

VENV="${ROOT}/models/dreamgen/.venv"
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}:${ROOT}/models/dreamgen:${PYTHONPATH:-}"

DATASET_ROOT="${DATASET_ROOT:-./datasets/single_arm/multiview}"
INPUT_DIR="${INPUT_DIR:-./sampling_dataset/dense/single_arm/input/multiview/dreamgen}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./sampling_dataset/itm/single_arm/output/multiview/dreamgen}"
CATEGORY_FILTER="${CATEGORY:-}"
MODEL_SIZE="${MODEL_SIZE:-14B}"
FPS="${FPS:-10}"
FORCE_REGENERATE_OUTPUTS="${FORCE_REGENERATE_OUTPUTS:-0}"
MAX_EPISODES="${MAX_EPISODES:-}"
MIN_EPISODE_INDEX="${MIN_EPISODE_INDEX:-}"
EPISODE_SOURCE="${EPISODE_SOURCE:-}"

# ITM is single-GPU only.
NUM_GPUS=1

# ITM defaults (README_prunning.md — 14B is a 36-block model → start_layer_idx 6).
# compress_ratio left to the auto schedule unless ITM_COMPRESS_RATIO is set.
ITM_START_LAYER_IDX="${ITM_START_LAYER_IDX:-6}"
ITM_KEEP_LAST_N_DENSE="${ITM_KEEP_LAST_N_DENSE:-2}"
ITM_PRUNE_FROM_STEP="${ITM_PRUNE_FROM_STEP:-1}"
ITM_MERGE_FROM_STEP="${ITM_MERGE_FROM_STEP:-2}"
ITM_COMPRESS_RATIO="${ITM_COMPRESS_RATIO:-}"

CATEGORIES=()
if [[ -n "${CATEGORY_FILTER}" ]]; then
  CATEGORIES+=("${CATEGORY_FILTER}")
else
  CATEGORIES=(makovian non_makovian)
fi

echo "============================================================"
echo " DreamGen multiview — single_arm — ITM (single-GPU)"
echo "============================================================"
echo " Data root  : ${DATASET_ROOT}"
echo " Categories : ${CATEGORIES[*]}"
echo " Output root: ${OUTPUT_ROOT}"
echo " Model      : ${MODEL_SIZE}  GPUs: ${NUM_GPUS}  FPS: ${FPS}"
echo " ITM        : start_layer=${ITM_START_LAYER_IDX} keep_last=${ITM_KEEP_LAST_N_DENSE} prune_from=${ITM_PRUNE_FROM_STEP} merge_from=${ITM_MERGE_FROM_STEP} compress=${ITM_COMPRESS_RATIO:-auto}"
echo "============================================================"
echo ""

AVAILABLE_GPUS=$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)
if [[ "${AVAILABLE_GPUS}" -lt 1 ]]; then
  echo "[ERROR] No CUDA GPUs visible to PyTorch." >&2
  exit 1
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

  BATCH_JSON="${INPUT_DIR}/${CAT}/batch_input.json"
  OUTPUT_DIR="${OUTPUT_ROOT}/${CAT}"

  echo "========== ${CAT} =========="

  PREPARE_ARGS=(
    --dataset_path "${DS_PATH}"
    --input_dir    "${INPUT_DIR}"
    --category     "${CAT}"
    --batch_json   "${BATCH_JSON}"
    --output_dir   "${OUTPUT_DIR}"
    --skip_existing
  )
  if [[ -n "${MAX_EPISODES}" ]]; then
    PREPARE_ARGS+=(--max_episodes "${MAX_EPISODES}")
  fi
  if [[ -n "${MIN_EPISODE_INDEX}" ]]; then
    PREPARE_ARGS+=(--min_episode_index "${MIN_EPISODE_INDEX}")
  fi
  if [[ -n "${EPISODE_SOURCE}" ]]; then
    PREPARE_ARGS+=(--episode_source "${EPISODE_SOURCE}")
  fi
  if [[ "${FORCE_REGENERATE_OUTPUTS}" == "1" ]]; then
    PREPARE_ARGS+=(--force_regenerate_outputs)
  fi

  python "${ROOT}/scripts/prepare_single_arm_multiview_dreamgen_batch.py" "${PREPARE_ARGS[@]}"

  if [[ ! -f "${BATCH_JSON}" ]]; then
    echo "[WARN] No batch at ${BATCH_JSON}, skip inference"
    continue
  fi

  ITM_FLAGS=(
    --use_itm
    --itm_start_layer_idx   "${ITM_START_LAYER_IDX}"
    --itm_keep_last_n_dense "${ITM_KEEP_LAST_N_DENSE}"
    --itm_prune_from_step   "${ITM_PRUNE_FROM_STEP}"
    --itm_merge_from_step   "${ITM_MERGE_FROM_STEP}"
  )
  if [[ -n "${ITM_COMPRESS_RATIO}" ]]; then
    ITM_FLAGS+=(--itm_compress_ratio "${ITM_COMPRESS_RATIO}")
  fi

  python -m examples.video2world_gr00t \
    --model_size       "${MODEL_SIZE}" \
    --gr00t_variant    droid \
    --batch_input_json "${BATCH_JSON}" \
    --prompt_prefix    "" \
    --num_gpus         "${NUM_GPUS}" \
    --disable_guardrail \
    --fps              "${FPS}" \
    "${ITM_FLAGS[@]}"

  echo ""
done

echo "[Done] ITM single_arm multiview — ${CATEGORIES[*]}"
echo "       Results: ${OUTPUT_ROOT}/<cat>/"
