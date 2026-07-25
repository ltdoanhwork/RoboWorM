#!/usr/bin/env bash
# DreamGen bimanual MULTIVIEW (2x2 grid) FULL roll-out inference with SVG2
# (Sparse VideoGen 2, k-means semantic-aware sparse attention).
#
# Exact mirror of scripts/run_bimanual_multiview_dreamgen_full.sh (dense) -- only
# forces NUM_GPUS=1, routes output to sampling_dataset/svg2, adds SVG2 flags.
#
# NOTE: SVG2 needs flashinfer>=0.3 (VariableBlockSparseAttentionWrapper); stock
# .venv has 0.2.5 which OOMs. Defaults to .venv_svg (flashinfer 0.3.1) + routes
# JIT caches to workspace .jitcache (home quota too small).
#
# Outputs: sampling_dataset/svg2/bimanual/output/multiview/dreamgen/<cat>/<iter>/<task>/episode_XXXXXX/
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_bimanual_multiview_dreamgen_svg2.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export VENV="${VENV:-${ROOT}/models/dreamgen/.venv_svg}"
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/python3.12/site-packages/nvidia/cusparse/lib:${VENV}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}:${ROOT}/models/dreamgen:${PYTHONPATH:-}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${ROOT}/.jitcache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${ROOT}/.jitcache/torchinductor}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${ROOT}/.jitcache/flashinfer}"
export FLASHINFER_JIT_DIR="${FLASHINFER_JIT_DIR:-${ROOT}/.jitcache/flashinfer}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT}/.jitcache/xdg}"
export TMPDIR="${TMPDIR:-${ROOT}/.jitcache/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT}/.jitcache/mpl}"
mkdir -p "${ROOT}/.jitcache/triton" "${ROOT}/.jitcache/torchinductor" "${ROOT}/.jitcache/flashinfer" "${ROOT}/.jitcache/xdg" "${ROOT}/.jitcache/tmp" "${ROOT}/.jitcache/mpl"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/exp_logs/dreamgen_logs/posttraining/multiview/bimanual_multiview_seg16/checkpoints_40_per}"
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

# SVG2 parameters (seeded from singleview; sweep via env).
SVG_START_LAYER_IDX="${SVG_START_LAYER_IDX:-1}"
SVG_DENSE_STEP_FRAC="${SVG_DENSE_STEP_FRAC:-0.3}"
SVG_NUM_Q_CENTROIDS="${SVG_NUM_Q_CENTROIDS:-32}"
SVG_NUM_K_CENTROIDS="${SVG_NUM_K_CENTROIDS:-64}"
SVG_TOP_P_KMEANS="${SVG_TOP_P_KMEANS:-0.9}"
SVG_MIN_KC_RATIO="${SVG_MIN_KC_RATIO:-0.10}"
SVG_KMEANS_ITER_INIT="${SVG_KMEANS_ITER_INIT:-10}"
SVG_KMEANS_ITER_STEP="${SVG_KMEANS_ITER_STEP:-1}"

MULTIVIEW_ROOT="${MULTIVIEW_ROOT:-${ROOT}/datasets/bimanual/multiview}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/svg2/bimanual/output/multiview/dreamgen}"

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
echo "DreamGen Bimanual Multiview FULL ROLL-OUT + SVG2 (.venv_svg/flashinfer0.3)"
echo "  Checkpoint: ${DIT_PATH} | Iter dir: ${ITER_TAG}"
echo "  Model: ${MODEL_SIZE} | Res: ${RESOLUTION} | FPS: ${FPS} | Guidance: ${GUIDANCE} | cond: ${NUM_CONDITIONAL_FRAMES}"
echo "  SVG2: start_layer=${SVG_START_LAYER_IDX} dense_frac=${SVG_DENSE_STEP_FRAC} qC=${SVG_NUM_Q_CENTROIDS} kC=${SVG_NUM_K_CENTROIDS}"
echo "  Save root: ${SAVE_ROOT} | Episodes: ${MAX_EPISODES:-all} (per task)"
echo "============================================"

METHOD_FLAGS=(
  --use_svg --svg_variant svg2
  --svg_start_layer_idx  "${SVG_START_LAYER_IDX}"
  --svg_dense_step_frac  "${SVG_DENSE_STEP_FRAC}"
  --svg_num_q_centroids  "${SVG_NUM_Q_CENTROIDS}"
  --svg_num_k_centroids  "${SVG_NUM_K_CENTROIDS}"
  --svg_top_p_kmeans     "${SVG_TOP_P_KMEANS}"
  --svg_min_kc_ratio     "${SVG_MIN_KC_RATIO}"
  --svg_kmeans_iter_init "${SVG_KMEANS_ITER_INIT}"
  --svg_kmeans_iter_step "${SVG_KMEANS_ITER_STEP}"
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

python "${ROOT}/scripts/infer_bimanual_multiview_rollout.py" "${PY_ARGS[@]}"

echo ""
echo "All done. Results under: ${SAVE_ROOT}/<category>/${ITER_TAG}/<task>/episode_*/"
