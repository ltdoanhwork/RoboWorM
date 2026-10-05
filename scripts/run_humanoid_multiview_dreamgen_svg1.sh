#!/usr/bin/env bash
# DreamGen humanoid MULTIVIEW (2x2 grid) FULL roll-out inference with SVG1
# (Sparse VideoGen v1 sparse attention).
#
# Exact mirror of scripts/run_humanoid_multiview_dreamgen_full.sh (dense) -- only
# forces NUM_GPUS=1 (sparse attn = single-GPU), routes output to
# sampling_dataset/svg1, and adds the SVG1 flags. Uses the SAME rollout infer
# script (infer_bimanual_multiview_rollout.py, humanoid-aware) so episode dirs +
# metrics match dense exactly.
#
# Output: sampling_dataset/svg1/humanoid/output/multiview/dreamgen/<cat>/<iter>/<task>/episode_XXXXXX/
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_humanoid_multiview_dreamgen_svg1.sh
#
# Env overrides: ITER_NAME, CHECKPOINTS_DIR, DIT_PATH, GUIDANCE, FPS,
#   NUM_CONDITIONAL_FRAMES, MAX_EPISODES, MAX_SEGMENTS, MULTIVIEW_ROOT, SAVE_ROOT,
#   SVG_{START_LAYER_IDX,DENSE_STEP_FRAC,SPARSITY,NUM_SAMPLED_ROWS,SAMPLE_MSE_MAX_ROW}
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export VENV="${ROOT}/models/dreamgen/.venv"
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/python3.12/site-packages/nvidia/cusparse/lib:${VENV}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
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

# SVG1 parameters (seeded from README humanoid singleview defaults; re-tune per model).
SVG_START_LAYER_IDX="${SVG_START_LAYER_IDX:-4}"
SVG_DENSE_STEP_FRAC="${SVG_DENSE_STEP_FRAC:-0.1}"
SVG_SPARSITY="${SVG_SPARSITY:-0.25}"
SVG_NUM_SAMPLED_ROWS="${SVG_NUM_SAMPLED_ROWS:-64}"
SVG_SAMPLE_MSE_MAX_ROW="${SVG_SAMPLE_MSE_MAX_ROW:-3000}"

MULTIVIEW_ROOT="${MULTIVIEW_ROOT:-${ROOT}/datasets/humanoid/multiview}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/svg1/humanoid/output/multiview/dreamgen}"

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
echo "DreamGen Humanoid Multiview FULL ROLL-OUT + SVG1"
echo "  Checkpoint  : ${DIT_PATH}"
echo "  GPUs        : ${NUM_GPUS} (sparse attn = single-GPU)"
echo "  SVG1        : start_layer=${SVG_START_LAYER_IDX} dense_frac=${SVG_DENSE_STEP_FRAC} sparsity=${SVG_SPARSITY} rows=${SVG_NUM_SAMPLED_ROWS}"
echo "  Save root   : ${SAVE_ROOT}"
echo "  Episodes    : ${MAX_EPISODES:-all} (per task)"
echo "============================================"

PY_ARGS=(
  --dataset-path "${DATASET_PATH_CSV}" --save-root "${SAVE_ROOT}" --dit-path "${DIT_PATH}"
  --iter-name "${ITER_TAG}" --model-size "${MODEL_SIZE}" --resolution "${RESOLUTION}"
  --fps "${FPS}" --save-fps "${SAVE_FPS}" --guidance "${GUIDANCE}" --num-gpus "${NUM_GPUS}"
  --num-conditional-frames "${NUM_CONDITIONAL_FRAMES}" --seed "${SEED}" --disable-guardrail
  --use_svg --svg_variant svg1
  --svg_start_layer_idx "${SVG_START_LAYER_IDX}" --svg_dense_step_frac "${SVG_DENSE_STEP_FRAC}"
  --svg_sparsity "${SVG_SPARSITY}" --svg_num_sampled_rows "${SVG_NUM_SAMPLED_ROWS}"
  --svg_sample_mse_max_row "${SVG_SAMPLE_MSE_MAX_ROW}"
)
[[ -n "${MAX_EPISODES}" ]] && PY_ARGS+=(--max-episodes "${MAX_EPISODES}")
[[ -n "${MAX_SEGMENTS}" ]] && PY_ARGS+=(--max-segments "${MAX_SEGMENTS}")

python "${ROOT}/scripts/infer_bimanual_multiview_rollout.py" "${PY_ARGS[@]}"

echo ""
echo "All done. Results under: ${SAVE_ROOT}/<category>/${ITER_TAG}/<task>/episode_*/"
