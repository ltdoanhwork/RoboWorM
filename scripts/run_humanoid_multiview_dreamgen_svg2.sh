#!/usr/bin/env bash
# DreamGen humanoid MULTIVIEW (2x2 grid) FULL roll-out inference with SVG2
# (Sparse VideoGen v2, k-means semantic-aware sparse attention).
#
# Mirror of the dense rollout, single-GPU, routes output to sampling_dataset/svg2.
# SVG2 may need a separate env (Sparse-VideoGen + flashinfer + cuvs); the dreamgen
# .venv already ships flashinfer + cuvs. Override VENV=models/dreamgen/.venv_svg if
# the default env fails to build SVG2 kernels.
#
# Output: sampling_dataset/svg2/humanoid/output/multiview/dreamgen/<cat>/<iter>/<task>/episode_XXXXXX/
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_humanoid_multiview_dreamgen_svg2.sh
#
# Env overrides: VENV, ITER_NAME, CHECKPOINTS_DIR, DIT_PATH, GUIDANCE, FPS,
#   NUM_CONDITIONAL_FRAMES, MAX_EPISODES, MAX_SEGMENTS, MULTIVIEW_ROOT, SAVE_ROOT,
#   SVG_{START_LAYER_IDX,DENSE_STEP_FRAC,NUM_Q_CENTROIDS,NUM_K_CENTROIDS,TOP_P_KMEANS,MIN_KC_RATIO,KMEANS_ITER_INIT,KMEANS_ITER_STEP}
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# SVG2 needs a separate env (Sparse-VideoGen block-sparse kernels + flashinfer
# 0.3.1). The dreamgen .venv (flashinfer 0.2.5) hits a dense-mask FALLBACK that
# OOMs (57GiB) on the long multiview sequence; .venv_svg (flashinfer 0.3.1) has
# the VariableBlockSparse wrapper and works. Override VENV to change env.
export VENV="${VENV:-${ROOT}/models/dreamgen/.venv_svg}"
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

# SVG2 parameters (verified working on .venv_svg; small centroids for the long
# multiview sequence — large q/k centroids OOM). Re-tune per model.
SVG_START_LAYER_IDX="${SVG_START_LAYER_IDX:-1}"
SVG_DENSE_STEP_FRAC="${SVG_DENSE_STEP_FRAC:-0.3}"
SVG_NUM_Q_CENTROIDS="${SVG_NUM_Q_CENTROIDS:-32}"
SVG_NUM_K_CENTROIDS="${SVG_NUM_K_CENTROIDS:-64}"
SVG_TOP_P_KMEANS="${SVG_TOP_P_KMEANS:-0.7}"  # 0.7 (not 0.9) is the real speed lever; keep centroids >= q32/k64
SVG_MIN_KC_RATIO="${SVG_MIN_KC_RATIO:-0.15}"
SVG_KMEANS_ITER_INIT="${SVG_KMEANS_ITER_INIT:-10}"
SVG_KMEANS_ITER_STEP="${SVG_KMEANS_ITER_STEP:-1}"

MULTIVIEW_ROOT="${MULTIVIEW_ROOT:-${ROOT}/datasets/humanoid/multiview}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/svg2/humanoid/output/multiview/dreamgen}"

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
echo "DreamGen Humanoid Multiview FULL ROLL-OUT + SVG2"
echo "  Checkpoint  : ${DIT_PATH}"
echo "  Venv        : ${VENV}"
echo "  GPUs        : ${NUM_GPUS} (sparse attn = single-GPU)"
echo "  SVG2        : start_layer=${SVG_START_LAYER_IDX} dense_frac=${SVG_DENSE_STEP_FRAC} q_c=${SVG_NUM_Q_CENTROIDS} k_c=${SVG_NUM_K_CENTROIDS} top_p=${SVG_TOP_P_KMEANS}"
echo "  Save root   : ${SAVE_ROOT}"
echo "  Episodes    : ${MAX_EPISODES:-all} (per task)"
echo "============================================"

PY_ARGS=(
  --dataset-path "${DATASET_PATH_CSV}" --save-root "${SAVE_ROOT}" --dit-path "${DIT_PATH}"
  --iter-name "${ITER_TAG}" --model-size "${MODEL_SIZE}" --resolution "${RESOLUTION}"
  --fps "${FPS}" --save-fps "${SAVE_FPS}" --guidance "${GUIDANCE}" --num-gpus "${NUM_GPUS}"
  --num-conditional-frames "${NUM_CONDITIONAL_FRAMES}" --seed "${SEED}" --disable-guardrail
  --use_svg --svg_variant svg2
  --svg_start_layer_idx "${SVG_START_LAYER_IDX}" --svg_dense_step_frac "${SVG_DENSE_STEP_FRAC}"
  --svg_num_q_centroids "${SVG_NUM_Q_CENTROIDS}" --svg_num_k_centroids "${SVG_NUM_K_CENTROIDS}"
  --svg_top_p_kmeans "${SVG_TOP_P_KMEANS}" --svg_min_kc_ratio "${SVG_MIN_KC_RATIO}"
  --svg_kmeans_iter_init "${SVG_KMEANS_ITER_INIT}" --svg_kmeans_iter_step "${SVG_KMEANS_ITER_STEP}"
)
[[ -n "${MAX_EPISODES}" ]] && PY_ARGS+=(--max-episodes "${MAX_EPISODES}")
[[ -n "${MAX_SEGMENTS}" ]] && PY_ARGS+=(--max-segments "${MAX_SEGMENTS}")

python "${ROOT}/scripts/infer_bimanual_multiview_rollout.py" "${PY_ARGS[@]}"

echo ""
echo "All done. Results under: ${SAVE_ROOT}/<category>/${ITER_TAG}/<task>/episode_*/"
