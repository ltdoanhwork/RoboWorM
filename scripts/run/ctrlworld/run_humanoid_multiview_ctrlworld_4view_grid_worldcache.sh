#!/bin/bash
# WorldCache on the humanoid 4-VIEW 2x2 GRID Ctrl-World model.
#
# Mirrors run_humanoid_multiview_ctrlworld_4view_grid.sh (the dense 4-view flow)
# EXACTLY -- same infer_bimanual_multiview_ctrlworld.py, same ckpt / dataset /
# --config humanoid_grid / DOWN_SAMPLE=6 rollout -- and only:
#   * swaps the leading sampling_dataset/dense -> sampling_dataset/worldcache
#   * adds the WorldCache flags
# (acceleration-benchmark + sampling-dataset-layout rules)
#
# Output:
#   sampling_dataset/worldcache/humanoid/output/multiview/ctrlworld_4view_grid/
#       {makovian,non_makovian}/episode_*/{full_pred,pred_all_views,gt_all_views}.mp4 + metrics.json
#
# Resumable: episodes whose full_pred.mp4 exists are skipped.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run/ctrlworld/run_humanoid_multiview_ctrlworld_4view_grid_worldcache.sh
#   NUM_SHARDS=4 bash scripts/run/ctrlworld/run_humanoid_multiview_ctrlworld_4view_grid_worldcache.sh --launch-shards
#
# Env overrides:
#   CKPT (default checkpoint-140000.pt = cumulative step 190000), SPLIT (default all),
#   NUM_EPISODES, NUM_INFERENCE_STEPS, GUIDANCE, DOWN_SAMPLE (default 6 = 5fps),
#   OUTPUT_ROOT / INPUT_ROOT, NUM_SHARDS / SHARD_ID, GPUS (for --launch-shards),
#   WORLDCACHE_{REL_L1_THRESH,RET_RATIO,PROBE_DEPTH,MOTION_SENSITIVITY}

set -e

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "${ROOT}"

DATASET_NAME="${DATASET_NAME:-humanoid_multiview_4view_grid}"
# Pin the dataset dir to the 4-view tree. The base script honours DATASET_DIR over
# DATASET_NAME, so a stale DATASET_DIR inherited from the caller's env (e.g. a
# deleted scratch sweep dir) would silently process 0 episodes.
DATASET_DIR="${DATASET_DIR:-${ROOT}/models/Ctrl-World/dataset_example/${DATASET_NAME}}"
export DATASET_DIR
CKPT="${CKPT:-${ROOT}/checkpoints/finetuned/ctrlworld/humanoid_multiview_4view_grid/checkpoint-140000.pt}"
DOWN_SAMPLE="${DOWN_SAMPLE:-6}"
SPLIT="${SPLIT:-all}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/sampling_dataset/worldcache/humanoid/output/multiview/ctrlworld_4view_grid}"
INPUT_ROOT="${INPUT_ROOT:-${ROOT}/sampling_dataset/worldcache/humanoid/input/multiview/ctrlworld_4view_grid}"

# WorldCache parameters -- TUNED on the humanoid 4-VIEW GRID model (see
# .cursor/rules/ctrlworld-humanoid-multiview-4view-grid-caching-insights.mdc).
# thresh=0.03 / ret_ratio=0.0 / probe_depth=1 -> 1.62x (2 val eps) / 1.65x (4 val
# eps) at PSNR -0.03 vs dense, visually identical on all 4 views.
# Same lesson as the bimanual 4-view grid: ret_ratio (0.4 -> 0.0) and probe_depth
# (2 -> 1) are the speed levers, NOT the threshold. On THIS model probe_depth=2
# is actively harmful (PSNR 20.56 -> 17.71: the deeper probe drifts less in
# relative terms, so the same threshold over-skips). Keep thresh <= 0.03; 0.035
# already costs -0.17 PSNR and 0.04 costs -0.46 for ~+0.1x.
WORLDCACHE_REL_L1_THRESH="${WORLDCACHE_REL_L1_THRESH:-0.03}"
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
    log="${ROOT}/logs/exp_logs/tmux_logs/cw_hum_grid_4view_worldcache_shard${i}.log"
    tmux new-session -d -s "cw_h4wc_s${i}" \
      "CUDA_VISIBLE_DEVICES=${gpu} NUM_SHARDS=${NUM_SHARDS} SHARD_ID=${i} \
       CKPT='${CKPT}' SPLIT='${SPLIT}' DOWN_SAMPLE='${DOWN_SAMPLE}' \
       OUTPUT_ROOT='${OUTPUT_ROOT}' INPUT_ROOT='${INPUT_ROOT}' \
       WORLDCACHE_REL_L1_THRESH='${WORLDCACHE_REL_L1_THRESH}' \
       WORLDCACHE_RET_RATIO='${WORLDCACHE_RET_RATIO}' \
       WORLDCACHE_PROBE_DEPTH='${WORLDCACHE_PROBE_DEPTH}' \
       bash scripts/run/ctrlworld/run_humanoid_multiview_ctrlworld_4view_grid_worldcache.sh 2>&1 | tee '${log}'; exec bash"
    echo "  shard ${i} -> GPU ${gpu}  (tmux cw_h4wc_s${i}, log ${log})"
  done
  echo ""
  echo "When all shards finish, merge the summaries:"
  echo "  python scripts/tools/ops/merge_ctrlworld_shard_summaries.py --output-root ${OUTPUT_ROOT}"
  exit 0
fi

# ---- single process (optionally one shard) ---------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CONFIG=humanoid_grid \
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
  bash scripts/run/ctrlworld/run_humanoid_multiview_ctrlworld_worldcache.sh

if [ "${NUM_SHARDS:-1}" -le 1 ]; then
  python scripts/tools/ops/merge_ctrlworld_shard_summaries.py --output-root "${OUTPUT_ROOT}" || true
fi
