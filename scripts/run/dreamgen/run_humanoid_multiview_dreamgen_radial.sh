#!/usr/bin/env bash
# DreamGen humanoid MULTIVIEW (2x2 grid) FULL roll-out inference with Radial Attention.
#
# Mirror of the dense rollout, single-GPU, routes output to sampling_dataset/radial.
# Radial needs flashinfer (dreamgen .venv already ships it; else see
# scripts/tools/ops/bootstrap_radial_env.sh).
#
# Output: sampling_dataset/radial/humanoid/output/multiview/dreamgen/<cat>/<iter>/<task>/episode_XXXXXX/
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamgen/run_humanoid_multiview_dreamgen_radial.sh
#
# Env overrides: VENV, ITER_NAME, CHECKPOINTS_DIR, DIT_PATH, GUIDANCE, FPS,
#   NUM_CONDITIONAL_FRAMES, MAX_EPISODES, MAX_SEGMENTS, MULTIVIEW_ROOT, SAVE_ROOT,
#   RADIAL_{DECAY_FACTOR,BLOCK_SIZE,START_LAYER_IDX,MODEL_TYPE}
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

export VENV="${VENV:-${ROOT}/models/dreamgen/.venv}"
_PYVER="$(basename "$(ls -d ${VENV}/lib/python3.* 2>/dev/null | head -1)")"
export LD_LIBRARY_PATH="${VENV}/lib/${_PYVER}/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/${_PYVER}/site-packages/nvidia/cusparse/lib:${VENV}/lib/${_PYVER}/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}:${ROOT}/models/dreamgen:${PYTHONPATH:-}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/checkpoints/finetuned/dreamgen/posttraining/multiview/humanoid_multiview_seg16/checkpoints_40_per}"
if [[ -n "${ITER_NAME:-}" ]]; then
  _LATEST_FNAME="${ITER_NAME%.pt}.pt"
else
  _LATEST_FILE="${CHECKPOINTS_DIR}/latest_checkpoint.txt"
  if [[ -f "${_LATEST_FILE}" ]]; then _LATEST_FNAME="$(cat "${_LATEST_FILE}")"; else _LATEST_FNAME="iter_000100000.pt"; fi
fi
ITER_NAME="${_LATEST_FNAME%.pt}"
DIT_PATH="${DIT_PATH:-${CHECKPOINTS_DIR}/model/${_LATEST_FNAME}}"

NUM_GPUS=1
MODEL_SIZE="${MODEL_SIZE:-2B}"
RESOLUTION="${RESOLUTION:-480}"
FPS="${FPS:-16}"
SAVE_FPS="${SAVE_FPS:-16}"
GUIDANCE="${GUIDANCE:-7}"
NUM_CONDITIONAL_FRAMES="${NUM_CONDITIONAL_FRAMES:-5}"
MAX_EPISODES="${MAX_EPISODES:-}"
MAX_SEGMENTS="${MAX_SEGMENTS:-}"
SEED="${SEED:-0}"
ITER_TAG="${ITER_TAG:-${ITER_NAME}}"

# Radial parameters (seeded from README gr1 defaults; re-tune per model).
RADIAL_DECAY_FACTOR="${RADIAL_DECAY_FACTOR:-1.0}"
RADIAL_BLOCK_SIZE="${RADIAL_BLOCK_SIZE:-64}"
RADIAL_START_LAYER_IDX="${RADIAL_START_LAYER_IDX:-4}"
RADIAL_MODEL_TYPE="${RADIAL_MODEL_TYPE:-hunyuan}"

MULTIVIEW_ROOT="${MULTIVIEW_ROOT:-${ROOT}/datasets/humanoid/multiview}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/radial/humanoid/output/multiview/dreamgen}"

if [[ ! -f "${DIT_PATH}" ]]; then echo "[ERROR] Checkpoint not found: ${DIT_PATH}" >&2; exit 1; fi

DATASET_PATHS=()
for SPLIT in makovian non_makovian; do
  SPLIT_DIR="${MULTIVIEW_ROOT}/${SPLIT}"; [[ -d "${SPLIT_DIR}" ]] || continue
  while IFS= read -r -d '' TASK_DIR; do
    [[ -f "${TASK_DIR}/meta/episodes.jsonl" ]] || continue
    DATASET_PATHS+=("${TASK_DIR}")
  done < <(find "${SPLIT_DIR}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
done
if [[ "${#DATASET_PATHS[@]}" -eq 0 ]]; then echo "[ERROR] No task dirs under ${MULTIVIEW_ROOT}" >&2; exit 1; fi
DATASET_PATH_CSV="$(IFS=,; echo "${DATASET_PATHS[*]}")"

echo "============================================"
echo "DreamGen Humanoid Multiview FULL ROLL-OUT + Radial Attention"
echo "  Checkpoint  : ${DIT_PATH}"
echo "  Venv        : ${VENV}"
echo "  GPUs        : ${NUM_GPUS} (sparse attn = single-GPU)"
echo "  Radial      : decay=${RADIAL_DECAY_FACTOR} block=${RADIAL_BLOCK_SIZE} start_layer=${RADIAL_START_LAYER_IDX} model_type=${RADIAL_MODEL_TYPE}"
echo "  Save root   : ${SAVE_ROOT}"
echo "  Episodes    : ${MAX_EPISODES:-all} (per task)"
echo "============================================"

PY_ARGS=(
  --dataset-path "${DATASET_PATH_CSV}" --save-root "${SAVE_ROOT}" --dit-path "${DIT_PATH}"
  --iter-name "${ITER_TAG}" --model-size "${MODEL_SIZE}" --resolution "${RESOLUTION}"
  --fps "${FPS}" --save-fps "${SAVE_FPS}" --guidance "${GUIDANCE}" --num-gpus "${NUM_GPUS}"
  --num-conditional-frames "${NUM_CONDITIONAL_FRAMES}" --seed "${SEED}" --disable-guardrail
  --use_radial
  --radial_decay_factor "${RADIAL_DECAY_FACTOR}" --radial_block_size "${RADIAL_BLOCK_SIZE}"
  --radial_start_layer_idx "${RADIAL_START_LAYER_IDX}" --radial_model_type "${RADIAL_MODEL_TYPE}"
)
[[ -n "${MAX_EPISODES}" ]] && PY_ARGS+=(--max-episodes "${MAX_EPISODES}")
[[ -n "${MAX_SEGMENTS}" ]] && PY_ARGS+=(--max-segments "${MAX_SEGMENTS}")

python "${ROOT}/scripts/infer/infer_bimanual_multiview_rollout.py" "${PY_ARGS[@]}"

echo ""
echo "All done. Results under: ${SAVE_ROOT}/<category>/${ITER_TAG}/<task>/episode_*/"
