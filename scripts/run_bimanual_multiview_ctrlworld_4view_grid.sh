#!/bin/bash
# Full-episode Ctrl-World inference for the bimanual 4-VIEW 2x2 GRID model.
#
# Mirrors run_bimanual_multiview_ctrlworld.sh (the dense 3-view flow) EXACTLY —
# same infer_bimanual_multiview_ctrlworld.py, same autoregressive rollout logic
# (pred_step=num_frames=5, overlap 1 frame, history_idx [0,0,-8,-6,-4,-2],
# interact_num derived per-episode from video_length), same replay of GT 14-D
# qpos as the action condition, same metrics. The ONLY differences are:
#   --config bimanual_grid   (4 views in a 2x2 latent canvas 48x80, vs 3x1 72x40)
#   the 4-view dataset + the 4-view finetuned checkpoint
#
# Output (sampling-dataset-layout compliant, canonical folder):
#   sampling_dataset/dense/bimanual/output/multiview/ctrlworld_4view_grid/
#       {makovian,non_makovian}/episode_*/{full_pred,pred_all_views,gt_all_views}.mp4 + metrics.json
#
# Resumable: episodes whose full_pred.mp4 exists are skipped.
#
# Usage:
#   # single GPU, all episodes
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_bimanual_multiview_ctrlworld_4view_grid.sh
#
#   # 4-GPU parallel (shards the episode list, then merge summaries)
#   NUM_SHARDS=4 bash scripts/run_bimanual_multiview_ctrlworld_4view_grid.sh --launch-shards
#
# Env overrides:
#   CKPT (default checkpoint-110000.pt), SPLIT (default all), NUM_EPISODES,
#   NUM_INFERENCE_STEPS, GUIDANCE, DOWN_SAMPLE (default 6 = 5fps, matches training),
#   OUTPUT_ROOT / INPUT_ROOT, NUM_SHARDS / SHARD_ID, GPUS (for --launch-shards)

set -e

ROOT="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics"
cd "${ROOT}"

DATASET_NAME="${DATASET_NAME:-bimanual_multiview_4view_grid}"
CKPT="${CKPT:-${ROOT}/exp_logs/ctrlworld_logs/bimanual_multiview_4view_grid/checkpoint-110000.pt}"
DOWN_SAMPLE="${DOWN_SAMPLE:-6}"
SPLIT="${SPLIT:-all}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/sampling_dataset/dense/bimanual/output/multiview/ctrlworld_4view_grid}"
INPUT_ROOT="${INPUT_ROOT:-${ROOT}/sampling_dataset/dense/bimanual/input/multiview/ctrlworld_4view_grid}"

if [ ! -f "${CKPT}" ]; then
  echo "ERROR: checkpoint not found: ${CKPT}" >&2
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
    log="${ROOT}/exp_logs/tmux_logs/cw_grid_4view_full_shard${i}.log"
    tmux new-session -d -s "cw_g4_s${i}" \
      "CUDA_VISIBLE_DEVICES=${gpu} NUM_SHARDS=${NUM_SHARDS} SHARD_ID=${i} \
       CKPT='${CKPT}' SPLIT='${SPLIT}' DOWN_SAMPLE='${DOWN_SAMPLE}' \
       OUTPUT_ROOT='${OUTPUT_ROOT}' INPUT_ROOT='${INPUT_ROOT}' \
       bash scripts/run_bimanual_multiview_ctrlworld_4view_grid.sh 2>&1 | tee '${log}'; exec bash"
    echo "  shard ${i} -> GPU ${gpu}  (tmux cw_g4_s${i}, log ${log})"
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
  bash scripts/run_bimanual_multiview_ctrlworld.sh

if [ "${NUM_SHARDS:-1}" -le 1 ]; then
  python scripts/merge_ctrlworld_shard_summaries.py --output-root "${OUTPUT_ROOT}" || true
fi
