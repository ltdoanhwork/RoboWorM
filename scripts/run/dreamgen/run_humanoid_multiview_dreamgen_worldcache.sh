#!/usr/bin/env bash
# DreamGen humanoid MULTIVIEW (2x2 grid) FULL roll-out inference with WorldCache.
#
# Exact mirror of scripts/run/dreamgen/run_humanoid_multiview_dreamgen_full.sh (dense) -- only
# forces NUM_GPUS=1 (cache = single-GPU), routes output to sampling_dataset/worldcache,
# and adds the WorldCache flags.
#
# Outputs: sampling_dataset/worldcache/humanoid/output/multiview/dreamgen/<cat>/<iter>/<task>/episode_XXXXXX/
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamgen/run_humanoid_multiview_dreamgen_worldcache.sh
#
# Overrideable env: ITER_NAME, CHECKPOINTS_DIR, DIT_PATH, GUIDANCE, FPS,
#   NUM_CONDITIONAL_FRAMES, MAX_EPISODES, MAX_SEGMENTS, MULTIVIEW_ROOT, SAVE_ROOT,
#   WORLDCACHE_{NUM_STEPS,REL_L1_THRESH,RET_RATIO,PROBE_DEPTH,MOTION_SENSITIVITY,DYNAMIC_DECAY},
#   EPISODE_IDS (comma-separated trajectory ids; for targeted qualitative/timing runs)
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
EPISODE_IDS="${EPISODE_IDS:-}"
MAX_SEGMENTS="${MAX_SEGMENTS:-}"
SEED="${SEED:-0}"
ITER_TAG="${ITER_TAG:-${ITER_NAME}}"

# WorldCache parameters. TUNED for humanoid DreamGen multiview (2B seg16),
# measured on 1 GPU (2 episodes, per-segment gen time; dense=116s/seg):
#   thresh 0.20 / ret 0.2 / probe 2 -> 55.8s/seg (2.08x) at PSNR 12.54 (dense
#   12.76), visuals ~= dense. probe_depth 2 (not 4) is faster AND higher quality
#   here. Ladder: t0.15=69s(1.67x, PSNR 12.60); t0.20=56s(2.08x). WorldCache is
#   solid but DiCache is faster at equal/better quality (2.28x). DEFAULT t0.20.
WORLDCACHE_NUM_STEPS="${WORLDCACHE_NUM_STEPS:-35}"
WORLDCACHE_REL_L1_THRESH="${WORLDCACHE_REL_L1_THRESH:-0.20}"
WORLDCACHE_RET_RATIO="${WORLDCACHE_RET_RATIO:-0.2}"
WORLDCACHE_PROBE_DEPTH="${WORLDCACHE_PROBE_DEPTH:-2}"
WORLDCACHE_MOTION_SENSITIVITY="${WORLDCACHE_MOTION_SENSITIVITY:-5.0}"
WORLDCACHE_DYNAMIC_DECAY="${WORLDCACHE_DYNAMIC_DECAY:-0}"

MULTIVIEW_ROOT="${MULTIVIEW_ROOT:-${ROOT}/datasets/humanoid/multiview}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/worldcache/humanoid/output/multiview/dreamgen}"

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
echo "DreamGen Humanoid Multiview FULL ROLL-OUT + WorldCache"
echo "  Checkpoint  : ${DIT_PATH}"
echo "  Iter dir    : ${ITER_TAG}"
echo "  Model       : ${MODEL_SIZE} | Res: ${RESOLUTION} | FPS: ${FPS}"
echo "  GPUs        : ${NUM_GPUS} (cache = single-GPU)"
echo "  Guidance    : ${GUIDANCE} | cond_frames: ${NUM_CONDITIONAL_FRAMES}"
echo "  WorldCache  : steps=${WORLDCACHE_NUM_STEPS} thresh=${WORLDCACHE_REL_L1_THRESH} ret=${WORLDCACHE_RET_RATIO} probe=${WORLDCACHE_PROBE_DEPTH} motion=${WORLDCACHE_MOTION_SENSITIVITY} decay=${WORLDCACHE_DYNAMIC_DECAY}"
echo "  Save root   : ${SAVE_ROOT}"
echo "  Episodes    : ${MAX_EPISODES:-all} (per task)"
echo "============================================"

WORLDCACHE_FLAGS=(
  --worldcache-enabled
  --worldcache-num-steps          "${WORLDCACHE_NUM_STEPS}"
  --worldcache-rel-l1-thresh      "${WORLDCACHE_REL_L1_THRESH}"
  --worldcache-ret-ratio          "${WORLDCACHE_RET_RATIO}"
  --worldcache-probe-depth        "${WORLDCACHE_PROBE_DEPTH}"
  --worldcache-motion-sensitivity "${WORLDCACHE_MOTION_SENSITIVITY}"
)
if [[ "${WORLDCACHE_DYNAMIC_DECAY}" == "1" ]]; then
  WORLDCACHE_FLAGS+=(--worldcache-dynamic-decay)
fi

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
  "${WORLDCACHE_FLAGS[@]}"
)
[[ -n "${MAX_EPISODES}" ]] && PY_ARGS+=(--max-episodes "${MAX_EPISODES}")
[[ -n "${EPISODE_IDS}" ]] && PY_ARGS+=(--episode-ids "${EPISODE_IDS}")
[[ -n "${MAX_SEGMENTS}" ]] && PY_ARGS+=(--max-segments "${MAX_SEGMENTS}")

python "${ROOT}/scripts/infer/infer_bimanual_multiview_rollout.py" "${PY_ARGS[@]}"

echo ""
echo "All done. Results under: ${SAVE_ROOT}/<category>/${ITER_TAG}/<task>/episode_*/"
echo "Summary: ${SAVE_ROOT}/${ITER_TAG}_summary.json"
