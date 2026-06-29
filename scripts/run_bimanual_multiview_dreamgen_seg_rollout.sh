#!/usr/bin/env bash
# DreamGen bimanual MULTIVIEW (2x2 grid) ROLL-OUT inference.
#
# Unlike run_bimanual_multiview_dreamgen_dense.sh (single ~5s clip per episode),
# this rolls out segments autoregressively so the predicted video length is close
# to the ground-truth episode length — same spirit as the single-view
# full-episode script.
#
# Conditioning chain:
#   seg 0 : GT first-frame grid (image)            -> T frames
#   seg k : last frames of seg k-1 (video, 5-frame cond) -> T frames
#   ... until total length >= GT length, then trim to GT length.
#
# Outputs (canonical layout, {model}=dreamgen, distinguished by iter dir):
#   sampling_dataset/dense/bimanual/output/multiview/dreamgen/<cat>/<iter>/<task>/episode_XXXXXX/
#       full_pred.mp4  full_gt.mp4  full_merged.mp4  metrics.json  seg_*.mp4
#
# Usage (from repo root) — NUM_GPUS must equal #GPUs exposed:
#   MAX_EPISODES=2 NUM_GPUS=1 CUDA_VISIBLE_DEVICES=0 bash scripts/run_bimanual_multiview_dreamgen_seg_rollout.sh
#   NUM_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_bimanual_multiview_dreamgen_seg_rollout.sh
#   # pick a specific checkpoint (else auto-detects latest):
#   ITER_NAME=iter_000050000 bash scripts/run_bimanual_multiview_dreamgen_seg_rollout.sh
#
# Overrideable env: ITER_NAME, CHECKPOINTS_DIR, DIT_PATH, NUM_GPUS, GUIDANCE, FPS,
#   NUM_CONDITIONAL_FRAMES, MAX_EPISODES, MAX_SEGMENTS, MULTIVIEW_ROOT, SAVE_ROOT.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- Environment (DreamGen venv) ---
export VENV="${ROOT}/models/dreamgen/.venv"
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/python3.12/site-packages/nvidia/cusparse/lib:${VENV}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/models/dreamgen:${PYTHONPATH:-}"

# --- Checkpoint selection ---
# Pick a checkpoint via ITER_NAME (e.g. ITER_NAME=iter_000050000), otherwise
# fall back to the latest checkpoint recorded in latest_checkpoint.txt.
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/exp_logs/dreamgen_logs/posttraining/multiview/bimanual_multiview_seg/checkpoints_40_per}"
if [[ -n "${ITER_NAME:-}" ]]; then
  # Allow ITER_NAME with or without the .pt suffix.
  _LATEST_FNAME="${ITER_NAME%.pt}.pt"
else
  _LATEST_FILE="${CHECKPOINTS_DIR}/latest_checkpoint.txt"
  if [[ -f "${_LATEST_FILE}" ]]; then
    _LATEST_FNAME="$(cat "${_LATEST_FILE}")"
  else
    _LATEST_FNAME="iter_000010000.pt"
  fi
fi
ITER_NAME="${_LATEST_FNAME%.pt}"
DIT_PATH="${DIT_PATH:-${CHECKPOINTS_DIR}/model/${_LATEST_FNAME}}"

# --- Inference settings ---
NUM_GPUS="${NUM_GPUS:-1}"
MODEL_SIZE="${MODEL_SIZE:-2B}"
RESOLUTION="${RESOLUTION:-480}"
FPS="${FPS:-16}"
SAVE_FPS="${SAVE_FPS:-16}"
GUIDANCE="${GUIDANCE:-7}"
NUM_CONDITIONAL_FRAMES="${NUM_CONDITIONAL_FRAMES:-1}"
MASTER_PORT="${MASTER_PORT:-29504}"
MAX_EPISODES="${MAX_EPISODES:-}"
MAX_SEGMENTS="${MAX_SEGMENTS:-}"
SEED="${SEED:-0}"

# Tag the iter dir by conditioning mode so cond=1 and cond=5 runs of the SAME
# checkpoint don't overwrite each other (still under {model}=dreamgen).
ITER_TAG="${ITER_TAG:-${ITER_NAME}_cond${NUM_CONDITIONAL_FRAMES}}"

# --- Data/output (canonical {model}=dreamgen; iter dir distinguishes ckpt) ---
MULTIVIEW_ROOT="${MULTIVIEW_ROOT:-${ROOT}/datasets/bimanual/multiview}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/dense/bimanual/output/multiview/dreamgen}"

if [[ ! -f "${DIT_PATH}" ]]; then
  echo "[ERROR] Checkpoint not found: ${DIT_PATH}" >&2
  exit 1
fi

# Discover all task dirs (makovian + non_makovian) -> comma-separated list.
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
echo "DreamGen Bimanual Multiview ROLL-OUT"
echo "  Checkpoint  : ${DIT_PATH}"
echo "  Iter name   : ${ITER_NAME}"
echo "  Iter tag    : ${ITER_TAG}  (output dir)"
echo "  Model       : ${MODEL_SIZE} | Res: ${RESOLUTION} | FPS: ${FPS}"
echo "  GPUs        : ${NUM_GPUS}"
echo "  Guidance    : ${GUIDANCE} | cond_frames: ${NUM_CONDITIONAL_FRAMES}"
echo "  Tasks       : ${#DATASET_PATHS[@]}"
echo "  Save root   : ${SAVE_ROOT}"
echo "  Episodes    : ${MAX_EPISODES:-all}"
echo "============================================"

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
)
[[ -n "${MAX_EPISODES}" ]] && PY_ARGS+=(--max-episodes "${MAX_EPISODES}")
[[ -n "${MAX_SEGMENTS}" ]] && PY_ARGS+=(--max-segments "${MAX_SEGMENTS}")

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" \
    "${ROOT}/scripts/infer_bimanual_multiview_rollout.py" "${PY_ARGS[@]}"
else
  python "${ROOT}/scripts/infer_bimanual_multiview_rollout.py" "${PY_ARGS[@]}"
fi

echo ""
echo "All done. Results under: ${SAVE_ROOT}/<category>/${ITER_TAG}/<task>/episode_*/"
