#!/bin/bash
# ITM (spatio-temporal prune-once-hold, similarity_recover) on the humanoid
# 4-VIEW 2x2 GRID Ctrl-World model.
#
# Mirrors run_humanoid_multiview_ctrlworld_4view_grid.sh (the dense 4-view flow)
# EXACTLY -- same infer_bimanual_multiview_ctrlworld.py, same ckpt / dataset /
# --config humanoid_grid / DOWN_SAMPLE=6 rollout -- and only:
#   * swaps the leading sampling_dataset/dense -> sampling_dataset/itm
#   * adds the ITM ST-hold flags
# (acceleration-benchmark + sampling-dataset-layout rules)
#
# Output:
#   sampling_dataset/itm/humanoid/output/multiview/ctrlworld_4view_grid/
#       {makovian,non_makovian}/episode_*/{full_pred,pred_all_views,gt_all_views}.mp4 + metrics.json
#
# Resumable: episodes whose full_pred.mp4 exists are skipped.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run/ctrlworld/run_humanoid_multiview_ctrlworld_4view_grid_itm.sh
#   NUM_SHARDS=4 bash scripts/run/ctrlworld/run_humanoid_multiview_ctrlworld_4view_grid_itm.sh --launch-shards
#
# Env overrides:
#   CKPT (default checkpoint-140000.pt = cumulative step 190000), SPLIT (default all),
#   NUM_EPISODES, NUM_INFERENCE_STEPS, GUIDANCE, DOWN_SAMPLE (default 6 = 5fps),
#   OUTPUT_ROOT / INPUT_ROOT, NUM_SHARDS / SHARD_ID, GPUS (for --launch-shards),
#   ITM_ST_KEEP_RATIO, ITM_MAX_DOWNSAMPLE_RATIO

set -e

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "${ROOT}"

DATASET_NAME="${DATASET_NAME:-humanoid_multiview_4view_grid}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/models/Ctrl-World/dataset_example/${DATASET_NAME}}"
export DATASET_DIR
CKPT="${CKPT:-${ROOT}/checkpoints/finetuned/ctrlworld/humanoid_multiview_4view_grid/checkpoint-140000.pt}"
DOWN_SAMPLE="${DOWN_SAMPLE:-6}"
SPLIT="${SPLIT:-all}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/sampling_dataset/itm/humanoid/output/multiview/ctrlworld_4view_grid}"
INPUT_ROOT="${INPUT_ROOT:-${ROOT}/sampling_dataset/itm/humanoid/input/multiview/ctrlworld_4view_grid}"

# ITM ST-hold parameters -- TUNED on the humanoid 4-VIEW GRID model (see
# .cursor/rules/ctrlworld-humanoid-multiview-4view-grid-pruning-insights.mdc).
# ITM ST-hold = SiTo ST-hold + similarity_recover (pruned tokens copy the most
# SIMILAR kept token, not the nearest index), which preserves appearance better at
# the same keep ratio. With max_downsample_ratio=4 (all 3 stages) it gives the
# same ~1.11x as SiTo but marginally higher quality. DEFAULT = k0.85/d4
# (PSNR -0.33, LPIPS 0.25 = best pruning quality at 1.11x). k0.9/d1 = quality-safe
# (~1.0x); <=k0.8 corrupts.
ITM_ST_KEEP_RATIO="${ITM_ST_KEEP_RATIO:-0.85}"
ITM_MAX_DOWNSAMPLE_RATIO="${ITM_MAX_DOWNSAMPLE_RATIO:-4}"

if [ ! -f "${CKPT}" ]; then
  echo "ERROR: checkpoint not found: ${CKPT}" >&2
  exit 1
fi
if [ ! -d "${DATASET_DIR}/annotation" ]; then
  echo "ERROR: dataset annotation dir not found: ${DATASET_DIR}/annotation" >&2
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
    log="${ROOT}/logs/exp_logs/tmux_logs/cw_hum_grid_4view_itm_shard${i}.log"
    tmux new-session -d -s "cw_h4itm_s${i}" \
      "CUDA_VISIBLE_DEVICES=${gpu} NUM_SHARDS=${NUM_SHARDS} SHARD_ID=${i} \
       CKPT='${CKPT}' SPLIT='${SPLIT}' DOWN_SAMPLE='${DOWN_SAMPLE}' \
       OUTPUT_ROOT='${OUTPUT_ROOT}' INPUT_ROOT='${INPUT_ROOT}' \
       ITM_ST_KEEP_RATIO='${ITM_ST_KEEP_RATIO}' \
       ITM_MAX_DOWNSAMPLE_RATIO='${ITM_MAX_DOWNSAMPLE_RATIO}' \
       bash scripts/run/ctrlworld/run_humanoid_multiview_ctrlworld_4view_grid_itm.sh 2>&1 | tee '${log}'; exec bash"
    echo "  shard ${i} -> GPU ${gpu}  (tmux cw_h4itm_s${i}, log ${log})"
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
ITM_ST_KEEP_RATIO="${ITM_ST_KEEP_RATIO}" \
ITM_MAX_DOWNSAMPLE_RATIO="${ITM_MAX_DOWNSAMPLE_RATIO}" \
  bash scripts/run/ctrlworld/run_humanoid_multiview_ctrlworld_itm.sh

if [ "${NUM_SHARDS:-1}" -le 1 ]; then
  python scripts/tools/ops/merge_ctrlworld_shard_summaries.py --output-root "${OUTPUT_ROOT}" || true
fi
