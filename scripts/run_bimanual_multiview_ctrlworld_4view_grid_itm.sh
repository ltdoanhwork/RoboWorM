#!/bin/bash
# ITM (spatio-temporal prune-once-hold, similarity_recover) on the bimanual
# 4-VIEW 2x2 GRID Ctrl-World model.
#
# Mirrors run_bimanual_multiview_ctrlworld_4view_grid.sh (the dense 4-view flow)
# EXACTLY -- same infer_bimanual_multiview_ctrlworld.py, same ckpt / dataset /
# --config bimanual_grid / DOWN_SAMPLE=6 rollout -- and only:
#   * swaps the leading sampling_dataset/dense -> sampling_dataset/itm
#   * adds the ITM ST-hold flags
# Delegates to the shared (embodiment-agnostic) base script
# run_humanoid_multiview_ctrlworld_itm.sh with CONFIG=bimanual_grid + bimanual
# ckpt/dataset.
# (acceleration-benchmark + sampling-dataset-layout rules)
#
# Output:
#   sampling_dataset/itm/bimanual/output/multiview/ctrlworld_4view_grid/
#       {makovian,non_makovian}/episode_*/{full_pred,pred_all_views,gt_all_views}.mp4 + metrics.json
#
# Resumable: episodes whose full_pred.mp4 exists are skipped.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_bimanual_multiview_ctrlworld_4view_grid_itm.sh
#   NUM_SHARDS=4 bash scripts/run_bimanual_multiview_ctrlworld_4view_grid_itm.sh --launch-shards
#
# Env overrides:
#   CKPT (default checkpoint-110000.pt), SPLIT (default all),
#   NUM_EPISODES, NUM_INFERENCE_STEPS, GUIDANCE, DOWN_SAMPLE (default 6 = 5fps),
#   OUTPUT_ROOT / INPUT_ROOT, NUM_SHARDS / SHARD_ID, GPUS (for --launch-shards),
#   ITM_ST_KEEP_RATIO, ITM_MAX_DOWNSAMPLE_RATIO

set -e

ROOT="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics"
cd "${ROOT}"

DATASET_NAME="${DATASET_NAME:-bimanual_multiview_4view_grid}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/models/Ctrl-World/dataset_example/${DATASET_NAME}}"
export DATASET_DIR
CKPT="${CKPT:-${ROOT}/checkpoints/finetuned/ctrlworld/bimanual_multiview_4view_grid/checkpoint-110000.pt}"
DOWN_SAMPLE="${DOWN_SAMPLE:-6}"
SPLIT="${SPLIT:-all}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/sampling_dataset/itm/bimanual/output/multiview/ctrlworld_4view_grid}"
INPUT_ROOT="${INPUT_ROOT:-${ROOT}/sampling_dataset/itm/bimanual/input/multiview/ctrlworld_4view_grid}"

# ITM ST-hold parameters -- TUNED on this bimanual 4-VIEW GRID model (2 val eps,
# ckpt-110000, single-GPU vs dense 225.7s/ep mean, PSNR 22.12):
#   k0.9/d1  -> 1.02x, PSNR 21.92 (-0.20)  near-best / quality-safe (~no speedup)
#   k0.85/d4 -> 1.04x, PSNR 20.68 (-1.44)  best-quality knee      <-- DEFAULT
#   k0.7/d4  -> 1.13x, PSNR 18.73 (-3.39)  drop (corrupt)
# ITM's similarity_recover keeps quality noticeably better than SiTo at the same
# ratio (k0.85/d4 PSNR 20.68 vs SiTo k0.8/d4 19.68) but yields less speedup (1.04x
# vs 1.08x). Pick ITM k0.85/d4 for the quality-leaning pruning point, SiTo k0.8/d4
# for a bit more speed. k0.9/d1 is the ==dense safe point (no speedup).
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
    log="${ROOT}/logs/exp_logs/tmux_logs/cw_bim_grid_4view_itm_shard${i}.log"
    tmux new-session -d -s "cw_b4itm_s${i}" \
      "CUDA_VISIBLE_DEVICES=${gpu} NUM_SHARDS=${NUM_SHARDS} SHARD_ID=${i} \
       CKPT='${CKPT}' SPLIT='${SPLIT}' DOWN_SAMPLE='${DOWN_SAMPLE}' \
       OUTPUT_ROOT='${OUTPUT_ROOT}' INPUT_ROOT='${INPUT_ROOT}' \
       ITM_ST_KEEP_RATIO='${ITM_ST_KEEP_RATIO}' \
       ITM_MAX_DOWNSAMPLE_RATIO='${ITM_MAX_DOWNSAMPLE_RATIO}' \
       bash scripts/run_bimanual_multiview_ctrlworld_4view_grid_itm.sh 2>&1 | tee '${log}'; exec bash"
    echo "  shard ${i} -> GPU ${gpu}  (tmux cw_b4itm_s${i}, log ${log})"
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
ITM_ST_KEEP_RATIO="${ITM_ST_KEEP_RATIO}" \
ITM_MAX_DOWNSAMPLE_RATIO="${ITM_MAX_DOWNSAMPLE_RATIO}" \
  bash scripts/run_humanoid_multiview_ctrlworld_itm.sh

if [ "${NUM_SHARDS:-1}" -le 1 ]; then
  python scripts/merge_ctrlworld_shard_summaries.py --output-root "${OUTPUT_ROOT}" || true
fi
