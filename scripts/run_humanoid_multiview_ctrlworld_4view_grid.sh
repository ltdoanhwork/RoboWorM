#!/bin/bash
# Full-episode Ctrl-World inference for the humanoid 4-VIEW 2x2 GRID model.
#
# Mirrors run_humanoid_multiview_ctrlworld.sh (the dense 3-view flow) EXACTLY —
# same infer_bimanual_multiview_ctrlworld.py, same autoregressive rollout logic
# (pred_step=num_frames=5, overlap 1 frame, interact_num derived per-episode from
# video_length), same replay of the GT 36-D state as the action condition, same
# metrics. The ONLY differences are:
#   --config humanoid_grid   (4 views in a 2x2 latent canvas 48x80, vs 3x1 72x40)
#   the 4-view dataset + the 4-view finetuned checkpoint
#
# View names come from each episode's annotation `view_keys`, because humanoid
# tasks use two different camera naming groups (cam_high_rgb/cam_third_view vs
# cam_head_rgb/cam_front_rgb).
#
# Output (sampling-dataset-layout compliant, canonical folder):
#   sampling_dataset/dense/humanoid/output/multiview/ctrlworld_4view_grid/
#       {makovian,non_makovian}/episode_*/{full_pred,pred_all_views,gt_all_views}.mp4 + metrics.json
#
# Resumable: episodes whose full_pred.mp4 exists are skipped.
#
# Usage:
#   # single GPU, all episodes
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_humanoid_multiview_ctrlworld_4view_grid.sh
#
#   # quick smoke test (2 val episodes to a scratch dir)
#   CUDA_VISIBLE_DEVICES=0 NUM_EPISODES=2 SPLIT=val \
#     OUTPUT_ROOT=/tmp/hum4v/output INPUT_ROOT=/tmp/hum4v/input \
#     bash scripts/run_humanoid_multiview_ctrlworld_4view_grid.sh
#
#   # multi-GPU shards, then merge
#   NUM_SHARDS=4 bash scripts/run_humanoid_multiview_ctrlworld_4view_grid.sh --launch-shards
#
# Env overrides:
#   CKPT (default checkpoint-140000.pt = cumulative step 190000 — see NOTE below),
#   SPLIT (default all), NUM_EPISODES, NUM_INFERENCE_STEPS, GUIDANCE,
#   DOWN_SAMPLE (default 6 = 5fps, matches training), OUTPUT_ROOT / INPUT_ROOT,
#   NUM_SHARDS / SHARD_ID, GPUS (for --launch-shards)
#
# NOTE on checkpoint numbering: train_wm.py has no resume — every restart resets
# global_step to 0, so checkpoint filenames count PER TRAINING SEGMENT, not
# cumulatively. This model was interrupted twice by disk-quota failures, giving 3
# segments:
#   segment 0: warm from 3-view humanoid ckpt-200000, own steps 0->20000, last
#              valid = its checkpoint-10000  => cum 10000
#   segment 1: warm from that,                own steps 0->50000, last valid
#              = its checkpoint-40000         => cum 50000  (= warmstart_cum50000.pt)
#   segment 2: warm from that,                own steps 0->150000, last valid
#              = its checkpoint-140000        => cum 190000
# So for the CURRENT files: cumulative step = <filename number> + 50000.

set -e

ROOT="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics"
cd "${ROOT}"

DATASET_NAME="${DATASET_NAME:-humanoid_multiview_4view_grid}"
# checkpoint-140000.pt of the 3rd training segment = cumulative step 190000
CKPT="${CKPT:-${ROOT}/exp_logs/ctrlworld_logs/humanoid_multiview_4view_grid/checkpoint-140000.pt}"
DOWN_SAMPLE="${DOWN_SAMPLE:-6}"
SPLIT="${SPLIT:-all}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/sampling_dataset/dense/humanoid/output/multiview/ctrlworld_4view_grid}"
INPUT_ROOT="${INPUT_ROOT:-${ROOT}/sampling_dataset/dense/humanoid/input/multiview/ctrlworld_4view_grid}"

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
    log="${ROOT}/exp_logs/tmux_logs/cw_hum_grid_4view_full_shard${i}.log"
    tmux new-session -d -s "cw_h4_s${i}" \
      "CUDA_VISIBLE_DEVICES=${gpu} NUM_SHARDS=${NUM_SHARDS} SHARD_ID=${i} \
       CKPT='${CKPT}' SPLIT='${SPLIT}' DOWN_SAMPLE='${DOWN_SAMPLE}' \
       OUTPUT_ROOT='${OUTPUT_ROOT}' INPUT_ROOT='${INPUT_ROOT}' \
       bash scripts/run_humanoid_multiview_ctrlworld_4view_grid.sh 2>&1 | tee '${log}'; exec bash"
    echo "  shard ${i} -> GPU ${gpu}  (tmux cw_h4_s${i}, log ${log})"
  done
  echo ""
  echo "When all shards finish, merge the summaries:"
  echo "  python scripts/merge_ctrlworld_shard_summaries.py --output-root ${OUTPUT_ROOT}"
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
  bash scripts/run_humanoid_multiview_ctrlworld.sh

if [ "${NUM_SHARDS:-1}" -le 1 ]; then
  python scripts/merge_ctrlworld_shard_summaries.py --output-root "${OUTPUT_ROOT}" || true
fi
