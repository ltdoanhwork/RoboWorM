#!/usr/bin/env bash
# DreamGen bimanual MULTIVIEW (2x2 grid) FULL roll-out inference with ScalingCache.
#
# Exact mirror of scripts/run/dreamgen/run_bimanual_multiview_dreamgen_bsc.sh in logic, checkpoint,
# prompt, rollout convention and paths -- only:
#   * routes output to sampling_dataset/scalingcache/... (method = top-level dir),
#   * adds the ScalingCache flags,
#   * puts AM_DiT/src on PYTHONPATH, which is how `scl.backend` is importable.
#
# ScalingCache is a BASELINE: runtime AM_DiT/methods/cache_strategy/ScalingCache/dit.py (reused
# verbatim), DreamGen adapter AM_DiT/methods/cache_strategy/ScalingCache/dreamgen.py.
# Skip accounting: $CACHE_STATS_JSON (same schema as WorldCache/DiCache/FasterCache).
# Knob units are DENOISE steps (not net forwards, unlike the ours-method launcher).
#
# Usage (repo root):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamgen/run_bimanual_multiview_dreamgen_scalingcache.sh
#
# Overrideable env: ITER_NAME, CHECKPOINTS_DIR, DIT_PATH, GUIDANCE, FPS,
#   NUM_CONDITIONAL_FRAMES, MAX_EPISODES, MAX_SEGMENTS, MULTIVIEW_ROOT, SAVE_ROOT,
#   AM_DIT_ROOT, CACHE_STATS_JSON, SC_{ER,FIRST,LAST,NUM_STEPS,WARM_CAL}
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

export VENV="${ROOT}/models/dreamgen/.venv"
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/python3.12/site-packages/nvidia/cusparse/lib:${VENV}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"

# AM_DiT/src is what makes `from scl.backend import apply_scalingcache` resolve. AM_DiT's ROOT is
# deliberately NOT added: its methods/ is a regular package and would shadow this repo's
# namespace package. The shim appends what it needs to __path__ instead (AM_DiT/src/README.md).
AM_DIT_ROOT="${AM_DIT_ROOT:-$(cd "${ROOT}/../../AM_DiT" 2>/dev/null && pwd || true)}"
if [[ -z "${AM_DIT_ROOT}" || ! -f "${AM_DIT_ROOT}/src/scl/backend.py" ]]; then
  echo "[ERROR] AM_DiT not found. Set AM_DIT_ROOT to the AM_DiT checkout (needs src/scl/backend.py)." >&2
  exit 1
fi
# Repo root must precede models/dreamgen so `methods.cache_strategy` is importable.
export PYTHONPATH="${ROOT}:${ROOT}/models/dreamgen:${AM_DIT_ROOT}/src:${PYTHONPATH:-}"

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

# Cache backends are single-GPU only.
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

# ScalingCache parameters. S_f (first_enhance) is the paper's one tuned knob (10/12/14
# at T~50); error_rate scales delta_s and is OUR extra knob (paper = 1.0).
# SC_WARM_CAL=1 (default) seeds delta_s per Algorithm 1 -- the unseeded threshold was
# a measured bug on Ctrl-World (AM_DiT docs/INSIGHTS.md SEC 61). 0 only to reproduce it.
SC_NUM_STEPS="${SC_NUM_STEPS:-35}"
SC_ER="${SC_ER:-1.0}"
SC_FIRST="${SC_FIRST:-10}"
SC_LAST="${SC_LAST:-2}"
SC_WARM_CAL="${SC_WARM_CAL:-1}"

MULTIVIEW_ROOT="${MULTIVIEW_ROOT:-${ROOT}/datasets/bimanual/multiview}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/scalingcache/bimanual/output/multiview/dreamgen}"

# Skip accounting (the rollout harness reports nothing about cache behaviour, so a
# dead cache and a healthy one look identical without it).
export CACHE_STATS_JSON="${CACHE_STATS_JSON:-${SAVE_ROOT}/${ITER_TAG}_cache_stats.json}"

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
echo "DreamGen Bimanual Multiview FULL ROLL-OUT + ScalingCache"
echo "  Checkpoint  : ${DIT_PATH}"
echo "  Iter dir    : ${ITER_TAG}"
echo "  Model       : ${MODEL_SIZE} | Res: ${RESOLUTION} | FPS: ${FPS}"
echo "  GPUs        : ${NUM_GPUS} (cache = single-GPU)"
echo "  Guidance    : ${GUIDANCE} | cond_frames: ${NUM_CONDITIONAL_FRAMES}"
echo "  AM_DiT      : ${AM_DIT_ROOT}"
echo "  ScalingCache: error_rate=${SC_ER} S_f=${SC_FIRST} last=${SC_LAST} warm_cal=${SC_WARM_CAL} steps=${SC_NUM_STEPS} (denoise steps)"
echo "  Save root   : ${SAVE_ROOT}"
echo "  Skip json   : ${CACHE_STATS_JSON}"
echo "  Episodes    : ${MAX_EPISODES:-all} (per task)"
echo "============================================"

SCALINGCACHE_FLAGS=(
  --scalingcache-enabled
  --scalingcache-num-steps     "${SC_NUM_STEPS}"
  --scalingcache-error-rate    "${SC_ER}"
  --scalingcache-first-enhance "${SC_FIRST}"
  --scalingcache-last-enhance  "${SC_LAST}"
)
if [[ "${SC_WARM_CAL}" == "0" ]]; then SCALINGCACHE_FLAGS+=(--scalingcache-no-warm-calibrate); fi

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
  "${SCALINGCACHE_FLAGS[@]}"
)
[[ -n "${MAX_EPISODES}" ]] && PY_ARGS+=(--max-episodes "${MAX_EPISODES}")
[[ -n "${MAX_SEGMENTS}" ]] && PY_ARGS+=(--max-segments "${MAX_SEGMENTS}")

python "${ROOT}/scripts/infer/infer_bimanual_multiview_rollout.py" "${PY_ARGS[@]}"

echo ""
echo "All done. Results under: ${SAVE_ROOT}/<category>/${ITER_TAG}/<task>/episode_*/"
echo "Summary: ${SAVE_ROOT}/${ITER_TAG}_summary.json"
