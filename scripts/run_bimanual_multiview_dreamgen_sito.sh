#!/usr/bin/env bash
# DreamGen bimanual MULTIVIEW (2x2 grid) FULL roll-out inference with SiTo (token pruning).
#
# Exact mirror of scripts/run_bimanual_multiview_dreamgen_dicache.sh -- single-GPU,
# same rollout infer script, only swapping the DiCache flags for SiTo and routing
# output to sampling_dataset/sito.
#
# Outputs: sampling_dataset/sito/bimanual/output/multiview/dreamgen/<cat>/<iter>/<task>/episode_XXXXXX/
#
# !!! EXISTING OUTPUT NOTE (2026-08-11) !!!
# The 300 episodes already on disk under sampling_dataset/sito/bimanual/output/
# multiview/dreamgen/.../iter_000120000/ were generated on Jul 27 with the OLD
# default SITO_PRUNE_RATIO=0.30. The tuned default is now 0.20 (0.30 visually
# smears the Pred on this 2x2-grid multiview — see the dreamgen-bimanual-pruning
# -insights rule). Those 300 are KEPT as-is (not regenerated); the script SKIPS
# existing files, so a plain re-run will NOT overwrite them. To regenerate at 0.20,
# delete the iter_000120000 dirs first (or run with a fresh SAVE_ROOT).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_bimanual_multiview_dreamgen_dicache.sh
#
# Overrideable env: ITER_NAME, CHECKPOINTS_DIR, DIT_PATH, GUIDANCE, FPS,
#   NUM_CONDITIONAL_FRAMES, MAX_EPISODES, MAX_SEGMENTS, MULTIVIEW_ROOT, SAVE_ROOT,
#   DICACHE_{NUM_STEPS,REL_L1_THRESH,RET_RATIO,PROBE_DEPTH}
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export VENV="${ROOT}/models/dreamgen/.venv"
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/python3.12/site-packages/nvidia/cusparse/lib:${VENV}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}:${ROOT}/models/dreamgen:${PYTHONPATH:-}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/checkpoints/finetuned/dreamgen/posttraining/multiview/bimanual_multiview_seg16/checkpoints_40_per}"
if [[ -n "${ITER_NAME:-}" ]]; then
  _LATEST_FNAME="${ITER_NAME%.pt}.pt"
else
  _LATEST_FILE="${CHECKPOINTS_DIR}/latest_checkpoint.txt"
  if [[ -f "${_LATEST_FILE}" ]]; then
    _LATEST_FNAME="$(cat "${_LATEST_FILE}")"
  else
    _LATEST_FNAME="iter_000120000.pt"
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

# SiTo parameters. TUNED on bimanual multiview 2x2 grid (2 ep, ckpt-120000) vs
# dense 119.0s/seg:
#   p0.20 -> 99.9s (1.19x)  Pred mostly coherent, mild speckle  <-- DEFAULT (knee)
#   p0.30 -> 92.2s (1.29x)  PSNR 13.35 (>dense!) but full_merged Pred CORRUPTS
#                           (arm dissolves into speckle, box melts) — PSNR misleads
# Multiview 2x2 + autoregressive rollout degrades EARLY (same as humanoid multiview);
# the blue-mat static bg inflates PSNR while the Pred views smear. Judge on
# full_merged, not PSNR. group_mode temporal, noise 0, start 6, keep 2.
SITO_PRUNE_RATIO="${SITO_PRUNE_RATIO:-0.20}"
SITO_START_LAYER_IDX="${SITO_START_LAYER_IDX:-6}"
SITO_KEEP_LAST_N_DENSE="${SITO_KEEP_LAST_N_DENSE:-2}"
SITO_GROUP_MODE="${SITO_GROUP_MODE:-temporal}"
SITO_NOISE_ALPHA="${SITO_NOISE_ALPHA:-0.0}"

MULTIVIEW_ROOT="${MULTIVIEW_ROOT:-${ROOT}/datasets/bimanual/multiview}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/sito/bimanual/output/multiview/dreamgen}"

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
echo "DreamGen Bimanual Multiview FULL ROLL-OUT + SiTo"
echo "  Checkpoint  : ${DIT_PATH}"
echo "  Iter dir    : ${ITER_TAG}"
echo "  Model       : ${MODEL_SIZE} | Res: ${RESOLUTION} | FPS: ${FPS}"
echo "  GPUs        : ${NUM_GPUS} (pruning = single-GPU)"
echo "  Guidance    : ${GUIDANCE} | cond_frames: ${NUM_CONDITIONAL_FRAMES}"
echo "  SiTo        : prune=${SITO_PRUNE_RATIO} start=${SITO_START_LAYER_IDX} keep_last=${SITO_KEEP_LAST_N_DENSE} group=${SITO_GROUP_MODE} noise=${SITO_NOISE_ALPHA}"
echo "  Save root   : ${SAVE_ROOT}"
echo "  Episodes    : ${MAX_EPISODES:-all} (per task)"
echo "============================================"

DICACHE_FLAGS=(
  --use_sito
  --sito_prune_ratio        "${SITO_PRUNE_RATIO}"
  --sito_start_layer_idx    "${SITO_START_LAYER_IDX}"
  --sito_keep_last_n_dense  "${SITO_KEEP_LAST_N_DENSE}"
  --sito_group_mode         "${SITO_GROUP_MODE}"
  --sito_noise_alpha        "${SITO_NOISE_ALPHA}"
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
  "${DICACHE_FLAGS[@]}"
)
[[ -n "${MAX_EPISODES}" ]] && PY_ARGS+=(--max-episodes "${MAX_EPISODES}")
[[ -n "${MAX_SEGMENTS}" ]] && PY_ARGS+=(--max-segments "${MAX_SEGMENTS}")

python "${ROOT}/scripts/infer_bimanual_multiview_rollout.py" "${PY_ARGS[@]}"

echo ""
echo "All done. Results under: ${SAVE_ROOT}/<category>/${ITER_TAG}/<task>/episode_*/"
echo "Summary: ${SAVE_ROOT}/${ITER_TAG}_summary.json"
