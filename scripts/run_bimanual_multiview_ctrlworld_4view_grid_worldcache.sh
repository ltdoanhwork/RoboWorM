#!/bin/bash
# WorldCache on the bimanual 4-VIEW 2x2 GRID Ctrl-World model.
#
# Mirrors run_bimanual_multiview_ctrlworld_4view_grid.sh (the dense 4-view flow)
# EXACTLY -- same infer_bimanual_multiview_ctrlworld.py, same ckpt / dataset /
# --config bimanual_grid / DOWN_SAMPLE=6 rollout -- and only:
#   * swaps the leading sampling_dataset/dense -> sampling_dataset/worldcache
#   * adds the WorldCache flags
# (acceleration-benchmark + sampling-dataset-layout rules)
#
# Output:
#   sampling_dataset/worldcache/bimanual/output/multiview/ctrlworld_4view_grid/
#       {makovian,non_makovian}/episode_*/{full_pred,pred_all_views,gt_all_views}.mp4 + metrics.json
#
# Resumable: episodes whose full_pred.mp4 exists are skipped.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_bimanual_multiview_ctrlworld_4view_grid_worldcache.sh
#   NUM_SHARDS=4 bash scripts/run_bimanual_multiview_ctrlworld_4view_grid_worldcache.sh --launch-shards
#
# Env overrides:
#   CKPT (default checkpoint-110000.pt), SPLIT (default all), NUM_EPISODES,
#   NUM_INFERENCE_STEPS, GUIDANCE, DOWN_SAMPLE (default 6 = 5fps),
#   OUTPUT_ROOT / INPUT_ROOT, NUM_SHARDS / SHARD_ID, GPUS (for --launch-shards),
#   WORLDCACHE_{REL_L1_THRESH,RET_RATIO,PROBE_DEPTH,MOTION_SENSITIVITY}

set -e

ROOT="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics"
cd "${ROOT}"

DATASET_NAME="${DATASET_NAME:-bimanual_multiview_4view_grid}"
# Pin the dataset dir to the 4-view tree. The base script honours DATASET_DIR
# over DATASET_NAME, so a stale DATASET_DIR inherited from the caller's env
# (e.g. a deleted scratch sweep dir) would silently process 0 episodes.
DATASET_DIR="${DATASET_DIR:-${ROOT}/models/Ctrl-World/dataset_example/${DATASET_NAME}}"
export DATASET_DIR
CKPT="${CKPT:-${ROOT}/checkpoints/finetuned/ctrlworld/bimanual_multiview_4view_grid/checkpoint-110000.pt}"
DOWN_SAMPLE="${DOWN_SAMPLE:-6}"
SPLIT="${SPLIT:-all}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/sampling_dataset/worldcache/bimanual/output/multiview/ctrlworld_4view_grid}"
INPUT_ROOT="${INPUT_ROOT:-${ROOT}/sampling_dataset/worldcache/bimanual/input/multiview/ctrlworld_4view_grid}"

# WorldCache parameters -- TUNED on the 4-VIEW GRID model (see
# .cursor/rules/ctrlworld-bimanual-multiview-4view-grid-caching-insights.mdc).
# thresh=0.04 / ret_ratio=0.0 / probe_depth=1 -> 1.81x at PSNR 21.67
# (dense 22.12, -0.45), visually identical to dense on all 4 views.
# The big levers on the 4-view grid are ret_ratio (0.4 -> 0.0: 1.14x -> 1.51x)
# and probe_depth (2 -> 1: 1.51x -> 1.81x), NOT the threshold: thresh>=0.06
# collapses quality (PSNR 20.3 -> 16.9 at 0.08) for little extra speed.
WORLDCACHE_REL_L1_THRESH="${WORLDCACHE_REL_L1_THRESH:-0.04}"
WORLDCACHE_RET_RATIO="${WORLDCACHE_RET_RATIO:-0.0}"
WORLDCACHE_PROBE_DEPTH="${WORLDCACHE_PROBE_DEPTH:-1}"
WORLDCACHE_MOTION_SENSITIVITY="${WORLDCACHE_MOTION_SENSITIVITY:-2.5}"

if [ ! -f "${CKPT}" ]; then
  echo "ERROR: checkpoint not found: ${CKPT}" >&2
  exit 1
fi

if [ ! -d "${DATASET_DIR}/annotation" ]; then
  echo "ERROR: dataset annotation dir not found: ${DATASET_DIR}/annotation" >&2
  echo "       (unset a stale DATASET_DIR in your environment)" >&2
  exit 1
fi

# ---- multi-GPU shard launcher ----------------------------------------------
if [ "${1:-}" = "--launch-shards" ]; then
  NUM_SHARDS="${NUM_SHARDS:-4}"
  GPUS="${GPUS:-0,1,2,3}"
  IFS=',' read -ra GPU_ARR <<< "${GPUS}"
  mkdir -p "${ROOT}/logs/exp_logs/tmux_logs"
  echo "Launching ${NUM_SHARDS} shards over GPUs ${GPUS}"
  for ((i = 0; i < NUM_SHARDS; i++)); do
    gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"
    log="${ROOT}/logs/exp_logs/tmux_logs/cw_grid_4view_worldcache_shard${i}.log"
    tmux new-session -d -s "cw_g4wc_s${i}" \
      "CUDA_VISIBLE_DEVICES=${gpu} NUM_SHARDS=${NUM_SHARDS} SHARD_ID=${i} \
       CKPT='${CKPT}' SPLIT='${SPLIT}' DOWN_SAMPLE='${DOWN_SAMPLE}' \
       OUTPUT_ROOT='${OUTPUT_ROOT}' INPUT_ROOT='${INPUT_ROOT}' \
       WORLDCACHE_REL_L1_THRESH='${WORLDCACHE_REL_L1_THRESH}' \
       WORLDCACHE_RET_RATIO='${WORLDCACHE_RET_RATIO}' \
       bash scripts/run_bimanual_multiview_ctrlworld_4view_grid_worldcache.sh 2>&1 | tee '${log}'; exec bash"
    echo "  shard ${i} -> GPU ${gpu}  (tmux cw_g4wc_s${i}, log ${log})"
  done
  echo ""
  echo "When all shards finish, merge the summaries:"
  echo "  python scripts/merge_ctrlworld_shard_summaries.py --output-root ${OUTPUT_ROOT}"
  exit 0
fi

# ---- single process (optionally one shard) ---------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CONFIG=bimanual_grid \
DATASET_NAME="${DATASET_NAME}" \
CKPT="${CKPT}" \
DOWN_SAMPLE="${DOWN_SAMPLE}" \
SPLIT="${SPLIT}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
INPUT_ROOT="${INPUT_ROOT}" \
NUM_EPISODES="${NUM_EPISODES:-}" \
NUM_EPISODES_PER_CATEGORY="${NUM_EPISODES_PER_CATEGORY:-}" \
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-}" \
GUIDANCE="${GUIDANCE:-}" \
WORLDCACHE_REL_L1_THRESH="${WORLDCACHE_REL_L1_THRESH}" \
WORLDCACHE_RET_RATIO="${WORLDCACHE_RET_RATIO}" \
WORLDCACHE_PROBE_DEPTH="${WORLDCACHE_PROBE_DEPTH}" \
WORLDCACHE_MOTION_SENSITIVITY="${WORLDCACHE_MOTION_SENSITIVITY}" \
  bash scripts/run_bimanual_multiview_ctrlworld_worldcache.sh

if [ "${NUM_SHARDS:-1}" -le 1 ]; then
  python scripts/merge_ctrlworld_shard_summaries.py --output-root "${OUTPUT_ROOT}" || true
fi
