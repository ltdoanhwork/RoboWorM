#!/usr/bin/env bash
# DreamGen bimanual MULTIVIEW (2x2 grid) FULL roll-out inference with Radial
# sparse attention. Mirror of dense (NUM_GPUS=1, output -> sampling_dataset/radial).
# Uses stock .venv. TUNED for multiview: decay 0.5 / start 4 (~1.35x, clean).
# NOTE: the singleview edge (decay 0.2/start 1) CORRUPTS on multiview (longer
# 4-view sequence) -> use decay 0.5 here.
#
# Outputs: sampling_dataset/radial/bimanual/output/multiview/dreamgen/<cat>/<iter>/<task>/episode_XXXXXX/
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamgen/run_bimanual_multiview_dreamgen_radial.sh
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

export VENV="${VENV:-${ROOT}/models/dreamgen/.venv}"
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/python3.12/site-packages/nvidia/cusparse/lib:${VENV}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}:${ROOT}/models/dreamgen:${PYTHONPATH:-}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${ROOT}/.jitcache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${ROOT}/.jitcache/torchinductor}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT}/.jitcache/xdg}"
export TMPDIR="${TMPDIR:-${ROOT}/.jitcache/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT}/.jitcache/mpl}"
mkdir -p "${ROOT}/.jitcache/triton" "${ROOT}/.jitcache/torchinductor" "${ROOT}/.jitcache/xdg" "${ROOT}/.jitcache/tmp" "${ROOT}/.jitcache/mpl"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/checkpoints/finetuned/dreamgen/posttraining/multiview/bimanual_multiview_seg16/checkpoints_40_per}"
if [[ -n "${ITER_NAME:-}" ]]; then
  _LATEST_FNAME="${ITER_NAME%.pt}.pt"
else
  _LATEST_FILE="${CHECKPOINTS_DIR}/latest_checkpoint.txt"
  if [[ -f "${_LATEST_FILE}" ]]; then _LATEST_FNAME="$(cat "${_LATEST_FILE}")"; else _LATEST_FNAME="iter_000120000.pt"; fi
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

# Tuned on bimanual multiview: start 8 / wan -> 1.41x at PSNR 12.11 / SSIM 0.456,
# the least-lossy Radial point. start4/hunyuan is faster (1.59x) but SSIM 0.436 and
# visibly distorts the wrist views; start4/wan is worst (PSNR 11.81). Radial is the
# weakest-quality sparse method here -> kept as a valid benchmark point.
RADIAL_DECAY_FACTOR="${RADIAL_DECAY_FACTOR:-0.5}"
RADIAL_BLOCK_SIZE="${RADIAL_BLOCK_SIZE:-64}"
RADIAL_START_LAYER_IDX="${RADIAL_START_LAYER_IDX:-8}"
RADIAL_MODEL_TYPE="${RADIAL_MODEL_TYPE:-wan}"

MULTIVIEW_ROOT="${MULTIVIEW_ROOT:-${ROOT}/datasets/bimanual/multiview}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/radial/bimanual/output/multiview/dreamgen}"

if [[ ! -f "${DIT_PATH}" ]]; then echo "[ERROR] Checkpoint not found: ${DIT_PATH}" >&2; exit 1; fi

DATASET_PATHS=()
for SPLIT in makovian non_makovian; do
  SPLIT_DIR="${MULTIVIEW_ROOT}/${SPLIT}"; [[ -d "${SPLIT_DIR}" ]] || continue
  while IFS= read -r -d '' TASK_DIR; do
    [[ -f "${TASK_DIR}/meta/episodes.jsonl" ]] || continue
    DATASET_PATHS+=("${TASK_DIR}")
  done < <(find "${SPLIT_DIR}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
done
[[ "${#DATASET_PATHS[@]}" -eq 0 ]] && { echo "[ERROR] No task dirs under ${MULTIVIEW_ROOT}" >&2; exit 1; }
DATASET_PATH_CSV="$(IFS=,; echo "${DATASET_PATHS[*]}")"

echo "============================================"
echo "DreamGen Bimanual Multiview FULL ROLL-OUT + Radial"
echo "  Checkpoint: ${DIT_PATH} | Iter dir: ${ITER_TAG}"
echo "  Model: ${MODEL_SIZE} | Res: ${RESOLUTION} | FPS: ${FPS} | Guidance: ${GUIDANCE} | cond: ${NUM_CONDITIONAL_FRAMES}"
echo "  Radial: decay=${RADIAL_DECAY_FACTOR} block=${RADIAL_BLOCK_SIZE} start_layer=${RADIAL_START_LAYER_IDX} model_type=${RADIAL_MODEL_TYPE}"
echo "  Save root: ${SAVE_ROOT} | Episodes: ${MAX_EPISODES:-all} (per task)"
echo "============================================"

METHOD_FLAGS=(
  --use_radial
  --radial_decay_factor    "${RADIAL_DECAY_FACTOR}"
  --radial_block_size      "${RADIAL_BLOCK_SIZE}"
  --radial_start_layer_idx "${RADIAL_START_LAYER_IDX}"
  --radial_model_type      "${RADIAL_MODEL_TYPE}"
)

PY_ARGS=(
  --dataset-path "${DATASET_PATH_CSV}" --save-root "${SAVE_ROOT}" --dit-path "${DIT_PATH}"
  --iter-name "${ITER_TAG}" --model-size "${MODEL_SIZE}" --resolution "${RESOLUTION}"
  --fps "${FPS}" --save-fps "${SAVE_FPS}" --guidance "${GUIDANCE}" --num-gpus "${NUM_GPUS}"
  --num-conditional-frames "${NUM_CONDITIONAL_FRAMES}" --seed "${SEED}" --disable-guardrail
  "${METHOD_FLAGS[@]}"
)
[[ -n "${MAX_EPISODES}" ]] && PY_ARGS+=(--max-episodes "${MAX_EPISODES}")
[[ -n "${MAX_SEGMENTS}" ]] && PY_ARGS+=(--max-segments "${MAX_SEGMENTS}")

python "${ROOT}/scripts/infer/infer_bimanual_multiview_rollout.py" "${PY_ARGS[@]}"

echo ""
echo "All done. Results under: ${SAVE_ROOT}/<category>/${ITER_TAG}/<task>/episode_*/"
