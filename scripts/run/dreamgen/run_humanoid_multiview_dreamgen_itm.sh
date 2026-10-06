#!/usr/bin/env bash
# DreamGen humanoid MULTIVIEW (2x2 grid) FULL roll-out inference with ITM (token merging).
#
# Exact mirror of scripts/run/dreamgen/run_humanoid_multiview_dreamgen_svg1.sh / _full.sh (dense) --
# single-GPU (pruning has no context parallelism), SAME humanoid-aware rollout infer
# script (infer_bimanual_multiview_rollout.py) so episode dirs + metrics match dense
# exactly, only swapping the sparse flags for ITM and routing output to
# sampling_dataset/itm.
#
# Outputs: sampling_dataset/itm/humanoid/output/multiview/dreamgen/<cat>/<iter>/<task>/episode_XXXXXX/
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamgen/run_humanoid_multiview_dreamgen_itm.sh
#   CUDA_VISIBLE_DEVICES=0 MAX_EPISODES=1 MULTIVIEW_ROOT=/tmp/hmv SAVE_ROOT=/tmp/out \
#     ITM_COMPRESS_RATIO=0.20 bash scripts/run/dreamgen/run_humanoid_multiview_dreamgen_itm.sh
#
# Overrideable env: ITER_NAME, CHECKPOINTS_DIR, DIT_PATH, GUIDANCE, FPS,
#   NUM_CONDITIONAL_FRAMES, MAX_EPISODES, MAX_SEGMENTS, MULTIVIEW_ROOT, SAVE_ROOT,
#   ITM_{COMPRESS_RATIO,START_LAYER_IDX,KEEP_LAST_N_DENSE,PRUNE_FROM_STEP,MERGE_FROM_STEP}
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
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
  if [[ -f "${_LATEST_FILE}" ]]; then
    _LATEST_FNAME="$(cat "${_LATEST_FILE}")"
  else
    _LATEST_FNAME="iter_000100000.pt"
  fi
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

# ITM parameters. DreamGen 2B DiT token merging (see .cursor/rules dreamgen
# pruning insights): ITM tends to be net-negative on small 2B DiT (per-block
# merge/unmerge overhead > attention saved), kept at the least-bad "mid" config
# (start 11, keep_last 4, compress 0.20). Valid benchmark point per accel rule.
ITM_COMPRESS_RATIO="${ITM_COMPRESS_RATIO:-0.20}"
ITM_START_LAYER_IDX="${ITM_START_LAYER_IDX:-11}"
ITM_KEEP_LAST_N_DENSE="${ITM_KEEP_LAST_N_DENSE:-4}"
ITM_PRUNE_FROM_STEP="${ITM_PRUNE_FROM_STEP:-1}"
ITM_MERGE_FROM_STEP="${ITM_MERGE_FROM_STEP:-2}"

MULTIVIEW_ROOT="${MULTIVIEW_ROOT:-${ROOT}/datasets/humanoid/multiview}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/itm/humanoid/output/multiview/dreamgen}"

if [[ ! -f "${DIT_PATH}" ]]; then
  echo "[ERROR] Checkpoint not found: ${DIT_PATH}" >&2
  exit 1
fi

DATASET_PATHS=()
for SPLIT in makovian non_makovian; do
  SPLIT_DIR="${MULTIVIEW_ROOT}/${SPLIT}"
  [[ -d "${SPLIT_DIR}" ]] || continue
  while IFS= read -r -d '' TASK_DIR; do
    [[ -f "${TASK_DIR}/meta/episodes.jsonl" ]] || continue
    DATASET_PATHS+=("${TASK_DIR}")
  done < <(find "${SPLIT_DIR}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
done
if [[ "${#DATASET_PATHS[@]}" -eq 0 ]]; then
  echo "[ERROR] No task dirs with meta/episodes.jsonl under ${MULTIVIEW_ROOT}" >&2
  exit 1
fi
DATASET_PATH_CSV="$(IFS=,; echo "${DATASET_PATHS[*]}")"

echo "============================================"
echo "DreamGen Humanoid Multiview FULL ROLL-OUT + ITM"
echo "  Checkpoint  : ${DIT_PATH}"
echo "  Iter dir    : ${ITER_TAG}"
echo "  Model       : ${MODEL_SIZE} | Res: ${RESOLUTION} | FPS: ${FPS}"
echo "  GPUs        : ${NUM_GPUS} (pruning = single-GPU)"
echo "  Guidance    : ${GUIDANCE} | cond_frames: ${NUM_CONDITIONAL_FRAMES}"
echo "  ITM         : compress=${ITM_COMPRESS_RATIO} start=${ITM_START_LAYER_IDX} keep_last=${ITM_KEEP_LAST_N_DENSE} prune_from=${ITM_PRUNE_FROM_STEP} merge_from=${ITM_MERGE_FROM_STEP}"
echo "  Save root   : ${SAVE_ROOT}"
echo "  Episodes    : ${MAX_EPISODES:-all} (per task)"
echo "============================================"

ITM_FLAGS=(
  --use_itm
  --itm_compress_ratio      "${ITM_COMPRESS_RATIO}"
  --itm_start_layer_idx     "${ITM_START_LAYER_IDX}"
  --itm_keep_last_n_dense   "${ITM_KEEP_LAST_N_DENSE}"
  --itm_prune_from_step     "${ITM_PRUNE_FROM_STEP}"
  --itm_merge_from_step     "${ITM_MERGE_FROM_STEP}"
)

PY_ARGS=(
  --dataset-path "${DATASET_PATH_CSV}"
  --save-root    "${SAVE_ROOT}"
  --dit-path     "${DIT_PATH}"
  --iter-name    "${ITER_TAG}"
  --model-size   "${MODEL_SIZE}"
  --resolution   "${RESOLUTION}"
  --fps          "${FPS}"
  --save-fps     "${SAVE_FPS}"
  --guidance     "${GUIDANCE}"
  --num-gpus     "${NUM_GPUS}"
  --num-conditional-frames "${NUM_CONDITIONAL_FRAMES}"
  --seed         "${SEED}"
  --disable-guardrail
  "${ITM_FLAGS[@]}"
)
[[ -n "${MAX_EPISODES}" ]] && PY_ARGS+=(--max-episodes "${MAX_EPISODES}")
[[ -n "${MAX_SEGMENTS}" ]] && PY_ARGS+=(--max-segments "${MAX_SEGMENTS}")

python "${ROOT}/scripts/infer/infer_bimanual_multiview_rollout.py" "${PY_ARGS[@]}"

echo ""
echo "All done. Results under: ${SAVE_ROOT}/<category>/${ITER_TAG}/<task>/episode_*/"
echo "Summary: ${SAVE_ROOT}/${ITER_TAG}_summary.json"
