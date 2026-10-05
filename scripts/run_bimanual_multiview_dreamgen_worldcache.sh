#!/usr/bin/env bash
# DreamGen bimanual MULTIVIEW (2x2 grid) FULL roll-out inference with WorldCache.
#
# Exact mirror of scripts/run_bimanual_multiview_dreamgen_full.sh (dense) in logic,
# checkpoint, prompt, rollout convention and paths -- only:
#   * forces NUM_GPUS=1 (cache backends are single-GPU / context_parallel_size=1),
#   * routes output to sampling_dataset/worldcache/... (method = top-level dir),
#   * adds the WorldCache flags.
#
# Outputs (canonical layout, {model}=dreamgen):
#   sampling_dataset/worldcache/bimanual/output/multiview/dreamgen/<cat>/<iter>/<task>/episode_XXXXXX/
#
# Usage (repo root):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_bimanual_multiview_dreamgen_worldcache.sh
#
# Overrideable env: ITER_NAME, CHECKPOINTS_DIR, DIT_PATH, GUIDANCE, FPS,
#   NUM_CONDITIONAL_FRAMES, MAX_EPISODES, MAX_SEGMENTS, MULTIVIEW_ROOT, SAVE_ROOT,
#   WORLDCACHE_{NUM_STEPS,REL_L1_THRESH,RET_RATIO,PROBE_DEPTH,MOTION_SENSITIVITY,DYNAMIC_DECAY}
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export VENV="${ROOT}/models/dreamgen/.venv"
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/python3.12/site-packages/nvidia/cusparse/lib:${VENV}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
# Repo root must precede models/dreamgen so `methods.cache_strategy` is importable.
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

# WorldCache parameters. TUNED optimum for bimanual DreamGen multiview (2B seg16):
# thresh=0.20 / ret=0.2 / probe_depth=2 -> 217s vs dense 338s (~1.56x) at
# PSNR 13.87 (dense 13.69 -> SLIGHTLY BETTER), clean visuals. Key finding:
# probe_depth 2 (not 4) is both faster AND higher quality here, and this DiT
# tolerates thresh 0.20 (like DiCache). Speed ladder: probe4 t0.12=289s(1.17x)
# -> t0.16=277s -> t0.20/ret0.3=244s -> probe2 t0.20/ret0.2=217s. Now competitive
# with DiCache (1.66x). Lowering ret (warm-up) helps once thresh is high enough.
#
# PUBLISHED-METHOD KNOBS (arXiv:2603.22286 "WorldCache: Content-Aware Caching for
# Accelerated Video World Models"). The paper's method is FOUR modules, and three
# of them were implemented in methods/cache_strategy/WorldCache/runtime.py but had
# no way in from this launcher, so every WorldCache row this repo has ever produced
# on DreamGen is the paper's `+ CFC` ABLATION ROW, not WorldCache:
#
#   paper Table 4 (Cosmos-Predict2.5-2B, I2W)   speedup
#     + CFC                                      1.52x   <- what this launcher ran
#     + CFC + SWD                                1.67x
#     + CFC + SWD + OFA                          1.49x
#     + CFC + SWD + OFA + ATS  (= WorldCache)    2.30x
#
# and the paper calls ATS "the primary source of WorldCache's speedup gains".
# Comparing anything against the CFC-only row is comparing against a strawman
# (AM_DiT/docs/INSIGHTS.md sections 41, 55). Defaults are left unchanged so existing
# results stay reproducible; the published configuration is
#   WORLDCACHE_REL_L1_THRESH=0.08 WORLDCACHE_MOTION_SENSITIVITY=2.0 \
#   WORLDCACHE_SALIENCY_ENABLED=1 WORLDCACHE_SALIENCY_WEIGHT=0.12 \
#   WORLDCACHE_OSI_ENABLED=1 WORLDCACHE_DYNAMIC_DECAY=1
# (tau0=0.08, alpha=2.0, beta_s=0.12, beta_d=4.0 -- paper Sec 4.1).
WORLDCACHE_NUM_STEPS="${WORLDCACHE_NUM_STEPS:-35}"
WORLDCACHE_REL_L1_THRESH="${WORLDCACHE_REL_L1_THRESH:-0.20}"
WORLDCACHE_RET_RATIO="${WORLDCACHE_RET_RATIO:-0.2}"
WORLDCACHE_PROBE_DEPTH="${WORLDCACHE_PROBE_DEPTH:-2}"
WORLDCACHE_MOTION_SENSITIVITY="${WORLDCACHE_MOTION_SENSITIVITY:-5.0}"
WORLDCACHE_DYNAMIC_DECAY="${WORLDCACHE_DYNAMIC_DECAY:-0}"
WORLDCACHE_SALIENCY_ENABLED="${WORLDCACHE_SALIENCY_ENABLED:-0}"
WORLDCACHE_SALIENCY_WEIGHT="${WORLDCACHE_SALIENCY_WEIGHT:-5.0}"
WORLDCACHE_OSI_ENABLED="${WORLDCACHE_OSI_ENABLED:-0}"
WORLDCACHE_FLOW_ENABLED="${WORLDCACHE_FLOW_ENABLED:-0}"
WORLDCACHE_FLOW_SCALE="${WORLDCACHE_FLOW_SCALE:-0.5}"

MULTIVIEW_ROOT="${MULTIVIEW_ROOT:-${ROOT}/datasets/bimanual/multiview}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/worldcache/bimanual/output/multiview/dreamgen}"

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
echo "DreamGen Bimanual Multiview FULL ROLL-OUT + WorldCache"
echo "  Checkpoint  : ${DIT_PATH}"
echo "  Iter dir    : ${ITER_TAG}"
echo "  Model       : ${MODEL_SIZE} | Res: ${RESOLUTION} | FPS: ${FPS}"
echo "  GPUs        : ${NUM_GPUS} (cache = single-GPU)"
echo "  Guidance    : ${GUIDANCE} | cond_frames: ${NUM_CONDITIONAL_FRAMES}"
echo "  WorldCache  : steps=${WORLDCACHE_NUM_STEPS} thresh=${WORLDCACHE_REL_L1_THRESH} ret=${WORLDCACHE_RET_RATIO} probe=${WORLDCACHE_PROBE_DEPTH} motion=${WORLDCACHE_MOTION_SENSITIVITY} decay=${WORLDCACHE_DYNAMIC_DECAY}"
echo "  modules     : CFC=on(always) SWD=${WORLDCACHE_SALIENCY_ENABLED}(beta=${WORLDCACHE_SALIENCY_WEIGHT}) OFA/OSI=${WORLDCACHE_OSI_ENABLED} ATS=${WORLDCACHE_DYNAMIC_DECAY} warp=${WORLDCACHE_FLOW_ENABLED}"
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
if [[ "${WORLDCACHE_SALIENCY_ENABLED}" == "1" ]]; then
  WORLDCACHE_FLAGS+=(--worldcache-saliency-enabled
                     --worldcache-saliency-weight "${WORLDCACHE_SALIENCY_WEIGHT}")
fi
if [[ "${WORLDCACHE_OSI_ENABLED}" == "1" ]]; then
  WORLDCACHE_FLAGS+=(--worldcache-osi-enabled)
fi
if [[ "${WORLDCACHE_FLOW_ENABLED}" == "1" ]]; then
  WORLDCACHE_FLAGS+=(--worldcache-flow-enabled
                     --worldcache-flow-scale "${WORLDCACHE_FLOW_SCALE}")
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
[[ -n "${MAX_SEGMENTS}" ]] && PY_ARGS+=(--max-segments "${MAX_SEGMENTS}")

python "${ROOT}/scripts/infer_bimanual_multiview_rollout.py" "${PY_ARGS[@]}"

echo ""
echo "All done. Results under: ${SAVE_ROOT}/<category>/${ITER_TAG}/<task>/episode_*/"
echo "Summary: ${SAVE_ROOT}/${ITER_TAG}_summary.json"
