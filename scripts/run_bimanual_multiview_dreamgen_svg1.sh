#!/usr/bin/env bash
# DreamGen bimanual MULTIVIEW (2x2 grid) FULL roll-out inference with SVG1
# (Sparse VideoGen 1: online MSE profiling + spatial/temporal head placement).
#
# Exact mirror of run_bimanual_multiview_dreamgen_full.sh (dense) -- only forces
# NUM_GPUS=1, routes output to sampling_dataset/svg1, adds SVG1 flags.
# Uses stock .venv (SVG1 = flex_attention, not flashinfer VariableBlock).
#
# Outputs: sampling_dataset/svg1/bimanual/output/multiview/dreamgen/<cat>/<iter>/<task>/episode_XXXXXX/
#
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/run_bimanual_multiview_dreamgen_svg1.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
mkdir -p "${ROOT}/.jitcache/triton" "${ROOT}/.jitcache/torchinductor" "${ROOT}/.jitcache/xdg" "${ROOT}/.jitcache/tmp" "${ROOT}/.jitcache/mpl"

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

# SVG1 parameters (seeded from singleview; sweep via env).
# Tuned on bimanual multiview: sparsity 0.25 -> 1.45x at PSNR 12.71 (== dense).
# Unlike single_arm singleview (where 0.25 gave NO speedup), 0.25 already wins here.
# 0.15 is faster (1.60x) but SSIM drops 0.50 -> 0.45; 0.35 is slower (1.33x) with no
# quality gain -> 0.25 is the knee. Small num_sampled_rows/dense_step_frac cut the
# MSE-profiling overhead (paid once on the first segment).
SVG_START_LAYER_IDX="${SVG_START_LAYER_IDX:-4}"
SVG_DENSE_STEP_FRAC="${SVG_DENSE_STEP_FRAC:-0.05}"
SVG_SPARSITY="${SVG_SPARSITY:-0.25}"
SVG_NUM_SAMPLED_ROWS="${SVG_NUM_SAMPLED_ROWS:-16}"
SVG_SAMPLE_MSE_MAX_ROW="${SVG_SAMPLE_MSE_MAX_ROW:-3000}"

MULTIVIEW_ROOT="${MULTIVIEW_ROOT:-${ROOT}/datasets/bimanual/multiview}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/svg1/bimanual/output/multiview/dreamgen}"

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
echo "DreamGen Bimanual Multiview FULL ROLL-OUT + SVG1"
echo "  Checkpoint: ${DIT_PATH} | Iter dir: ${ITER_TAG}"
echo "  Model: ${MODEL_SIZE} | Res: ${RESOLUTION} | FPS: ${FPS} | Guidance: ${GUIDANCE} | cond: ${NUM_CONDITIONAL_FRAMES}"
echo "  SVG1: start_layer=${SVG_START_LAYER_IDX} dense_frac=${SVG_DENSE_STEP_FRAC} sparsity=${SVG_SPARSITY}"
echo "  Save root: ${SAVE_ROOT} | Episodes: ${MAX_EPISODES:-all} (per task)"
echo "============================================"

METHOD_FLAGS=(
  --use_svg --svg_variant svg1
  --svg_start_layer_idx    "${SVG_START_LAYER_IDX}"
  --svg_dense_step_frac    "${SVG_DENSE_STEP_FRAC}"
  --svg_sparsity           "${SVG_SPARSITY}"
  --svg_num_sampled_rows   "${SVG_NUM_SAMPLED_ROWS}"
  --svg_sample_mse_max_row "${SVG_SAMPLE_MSE_MAX_ROW}"
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
