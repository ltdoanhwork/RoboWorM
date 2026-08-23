#!/bin/bash
# DiCache on the bimanual 4-VIEW 2x2 GRID Ctrl-World model.
#
# Mirrors run_bimanual_multiview_ctrlworld_4view_grid.sh (the dense 4-view flow)
# EXACTLY -- same infer_bimanual_multiview_ctrlworld.py, same ckpt / dataset /
# --config bimanual_grid / DOWN_SAMPLE=6 rollout -- and only:
#   * swaps the leading sampling_dataset/dense -> sampling_dataset/dicache
#   * adds the DiCache flags
# (acceleration-benchmark + sampling-dataset-layout rules)
#
# Output:
#   sampling_dataset/dicache/bimanual/output/multiview/ctrlworld_4view_grid/
#       {makovian,non_makovian}/episode_*/{full_pred,pred_all_views,gt_all_views}.mp4 + metrics.json
#
# Resumable: episodes whose full_pred.mp4 exists are skipped.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_bimanual_multiview_ctrlworld_4view_grid_dicache.sh
#   NUM_SHARDS=4 bash scripts/run_bimanual_multiview_ctrlworld_4view_grid_dicache.sh --launch-shards
#
# Env overrides:
#   CKPT (default checkpoint-110000.pt), SPLIT (default all), NUM_EPISODES,
#   NUM_INFERENCE_STEPS, GUIDANCE, DOWN_SAMPLE (default 6 = 5fps),
#   OUTPUT_ROOT / INPUT_ROOT, NUM_SHARDS / SHARD_ID, GPUS (for --launch-shards),
#   DICACHE_{REL_L1_THRESH,RET_RATIO,PROBE_DEPTH}

set -e

ROOT="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics"
cd "${ROOT}"

DATASET_NAME="${DATASET_NAME:-bimanual_multiview_4view_grid}"
# Pin the dataset dir to the 4-view tree. The base script honours DATASET_DIR
# over DATASET_NAME, so a stale DATASET_DIR inherited from the caller's env
# (e.g. a deleted scratch sweep dir) would silently process 0 episodes.
DATASET_DIR="${DATASET_DIR:-${ROOT}/models/Ctrl-World/dataset_example/${DATASET_NAME}}"
export DATASET_DIR
CKPT="${CKPT:-${ROOT}/exp_logs/ctrlworld_logs/bimanual_multiview_4view_grid/checkpoint-110000.pt}"
DOWN_SAMPLE="${DOWN_SAMPLE:-6}"
SPLIT="${SPLIT:-all}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/sampling_dataset/dicache/bimanual/output/multiview/ctrlworld_4view_grid}"
INPUT_ROOT="${INPUT_ROOT:-${ROOT}/sampling_dataset/dicache/bimanual/input/multiview/ctrlworld_4view_grid}"

# DiCache parameters -- TUNED on the 4-VIEW GRID model (see
# .cursor/rules/ctrlworld-bimanual-multiview-4view-grid-caching-insights.mdc).
# thresh=0.03 / ret_ratio=0.0 / probe_depth=1 -> 1.72x at PSNR 21.58
# (dense 22.12, -0.55), visually identical to dense on all 4 views.
# Same levers as WorldCache (ret_ratio + probe_depth do the work). DiCache
# needs the LOWER threshold 0.03: at 0.04 quality drops faster than WorldCache.
DICACHE_REL_L1_THRESH="${DICACHE_REL_L1_THRESH:-0.03}"
DICACHE_RET_RATIO="${DICACHE_RET_RATIO:-0.0}"
DICACHE_PROBE_DEPTH="${DICACHE_PROBE_DEPTH:-1}"

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
  mkdir -p "${ROOT}/exp_logs/tmux_logs"
  echo "Launching ${NUM_SHARDS} shards over GPUs ${GPUS}"
  for ((i = 0; i < NUM_SHARDS; i++)); do
    gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"
    log="${ROOT}/exp_logs/tmux_logs/cw_grid_4view_dicache_shard${i}.log"
    tmux new-session -d -s "cw_g4dc_s${i}" \
      "CUDA_VISIBLE_DEVICES=${gpu} NUM_SHARDS=${NUM_SHARDS} SHARD_ID=${i} \
       CKPT='${CKPT}' SPLIT='${SPLIT}' DOWN_SAMPLE='${DOWN_SAMPLE}' \
       OUTPUT_ROOT='${OUTPUT_ROOT}' INPUT_ROOT='${INPUT_ROOT}' \
       DICACHE_REL_L1_THRESH='${DICACHE_REL_L1_THRESH}' \
       DICACHE_RET_RATIO='${DICACHE_RET_RATIO}' \
       bash scripts/run_bimanual_multiview_ctrlworld_4view_grid_dicache.sh 2>&1 | tee '${log}'; exec bash"
    echo "  shard ${i} -> GPU ${gpu}  (tmux cw_g4dc_s${i}, log ${log})"
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
DICACHE_REL_L1_THRESH="${DICACHE_REL_L1_THRESH}" \
DICACHE_RET_RATIO="${DICACHE_RET_RATIO}" \
DICACHE_PROBE_DEPTH="${DICACHE_PROBE_DEPTH}" \
  bash scripts/run_bimanual_multiview_ctrlworld_dicache.sh

if [ "${NUM_SHARDS:-1}" -le 1 ]; then
  python scripts/merge_ctrlworld_shard_summaries.py --output-root "${OUTPUT_ROOT}" || true
fi
