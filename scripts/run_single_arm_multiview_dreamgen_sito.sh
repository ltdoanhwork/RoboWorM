#!/usr/bin/env bash
# =============================================================================
# run_single_arm_multiview_dreamgen_sito.sh
# DreamGen multiview (single_arm, droid layout) with SiTo self-attention pruning.
#
# Mirrors the dense run (run_single_arm_multiview_dreamgen_batch.sh) but routes
# outputs to the sito/ method tree and enables SiTo token pruning.
#
# SiTo is SINGLE-GPU only (no context parallelism), so this script forces
# NUM_GPUS=1 and calls `python -m examples.video2world_gr00t` directly.
#
# Runs both categories (makovian, non_makovian). Outputs (method = top-level dir):
#   sampling_dataset/sito/single_arm/output/multiview/dreamgen/<cat>/
#
# Usage (repo root):
#   bash scripts/run_single_arm_multiview_dreamgen_sito.sh
#   CUDA_VISIBLE_DEVICES=1 MAX_EPISODES=2 bash scripts/run_single_arm_multiview_dreamgen_sito.sh
#   CATEGORY=makovian bash scripts/run_single_arm_multiview_dreamgen_sito.sh
#
# Environment:
#   DATASET_ROOT   default ./datasets/single_arm/multiview
#   INPUT_DIR      default ./sampling_dataset/dense/single_arm/input/multiview/dreamgen
#   OUTPUT_ROOT    default ./sampling_dataset/sito/single_arm/output/multiview/dreamgen
#   CATEGORY       empty = both; or makovian | non_makovian only
#   MODEL_SIZE, FPS, MAX_EPISODES, MIN_EPISODE_INDEX, EPISODE_SOURCE, FORCE_REGENERATE_OUTPUTS=1
#   SITO_* overrides (see defaults below; 14B is a 36-block model)
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
OUTPUT_ROOT="${OUTPUT_ROOT:-./sampling_dataset/sito/single_arm/output/multiview/dreamgen}"
CATEGORY_FILTER="${CATEGORY:-}"
MODEL_SIZE="${MODEL_SIZE:-14B}"
FPS="${FPS:-10}"
FORCE_REGENERATE_OUTPUTS="${FORCE_REGENERATE_OUTPUTS:-0}"
MAX_EPISODES="${MAX_EPISODES:-}"
MIN_EPISODE_INDEX="${MIN_EPISODE_INDEX:-}"
EPISODE_SOURCE="${EPISODE_SOURCE:-}"

# SiTo is single-GPU only.
NUM_GPUS=1

# SiTo defaults (README_prunning.md — 14B is a 36-block model → start_layer_idx 6).
# prune_ratio left to the auto schedule unless SITO_PRUNE_RATIO is set.
SITO_START_LAYER_IDX="${SITO_START_LAYER_IDX:-6}"
SITO_KEEP_LAST_N_DENSE="${SITO_KEEP_LAST_N_DENSE:-2}"
SITO_PRUNE_RATIO="${SITO_PRUNE_RATIO:-}"

CATEGORIES=()
if [[ -n "${CATEGORY_FILTER}" ]]; then
  CATEGORIES+=("${CATEGORY_FILTER}")
else
  CATEGORIES=(makovian non_makovian)
fi

echo "============================================================"
echo " DreamGen multiview — single_arm — SiTo (single-GPU)"
echo "============================================================"
echo " Data root  : ${DATASET_ROOT}"
echo " Categories : ${CATEGORIES[*]}"
echo " Output root: ${OUTPUT_ROOT}"
echo " Model      : ${MODEL_SIZE}  GPUs: ${NUM_GPUS}  FPS: ${FPS}"
echo " SiTo       : start_layer=${SITO_START_LAYER_IDX} keep_last=${SITO_KEEP_LAST_N_DENSE} prune=${SITO_PRUNE_RATIO:-auto}"
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

  SITO_FLAGS=(
    --use_sito
    --sito_start_layer_idx   "${SITO_START_LAYER_IDX}"
    --sito_keep_last_n_dense "${SITO_KEEP_LAST_N_DENSE}"
  )
  if [[ -n "${SITO_PRUNE_RATIO}" ]]; then
    SITO_FLAGS+=(--sito_prune_ratio "${SITO_PRUNE_RATIO}")
  fi

  python -m examples.video2world_gr00t \
    --model_size       "${MODEL_SIZE}" \
    --gr00t_variant    droid \
    --batch_input_json "${BATCH_JSON}" \
    --prompt_prefix    "" \
    --num_gpus         "${NUM_GPUS}" \
    --disable_guardrail \
    --fps              "${FPS}" \
    "${SITO_FLAGS[@]}"

  echo ""
done

echo "[Done] SiTo single_arm multiview — ${CATEGORIES[*]}"
echo "       Results: ${OUTPUT_ROOT}/<cat>/"
