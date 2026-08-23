#!/bin/bash
# Radial Attention on the humanoid 4-VIEW 2x2 GRID Ctrl-World model.
# Mirrors run_humanoid_multiview_ctrlworld_4view_grid.sh (dense), delegating to the
# base radial script with CONFIG=humanoid_grid. See the 4view-grid sparse-attn rule.
#
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/run_humanoid_multiview_ctrlworld_4view_grid_radial.sh
#        NUM_SHARDS=4 bash scripts/run_humanoid_multiview_ctrlworld_4view_grid_radial.sh --launch-shards
# Env: CKPT, SPLIT, NUM_EPISODES, DOWN_SAMPLE, OUTPUT_ROOT/INPUT_ROOT, NUM_SHARDS/SHARD_ID,
#      GPUS, RADIAL_{DECAY,BLOCK_SIZE,FIRST_LAYERS_FP,MODEL_TYPE}

set -e
ROOT="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics"
cd "${ROOT}"

DATASET_NAME="${DATASET_NAME:-humanoid_multiview_4view_grid}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/models/Ctrl-World/dataset_example/${DATASET_NAME}}"; export DATASET_DIR
CKPT="${CKPT:-${ROOT}/exp_logs/ctrlworld_logs/humanoid_multiview_4view_grid/checkpoint-140000.pt}"
DOWN_SAMPLE="${DOWN_SAMPLE:-6}"; SPLIT="${SPLIT:-all}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/sampling_dataset/radial/humanoid/output/multiview/ctrlworld_4view_grid}"
INPUT_ROOT="${INPUT_ROOT:-${ROOT}/sampling_dataset/radial/humanoid/input/multiview/ctrlworld_4view_grid}"
RADIAL_DECAY="${RADIAL_DECAY:-1.0}"; RADIAL_BLOCK_SIZE="${RADIAL_BLOCK_SIZE:-64}"
RADIAL_FIRST_LAYERS_FP="${RADIAL_FIRST_LAYERS_FP:-2}"; RADIAL_MODEL_TYPE="${RADIAL_MODEL_TYPE:-hunyuan}"

[ -f "${CKPT}" ] || { echo "ERROR: checkpoint not found: ${CKPT}" >&2; exit 1; }
[ -d "${DATASET_DIR}/annotation" ] || { echo "ERROR: annotation dir not found: ${DATASET_DIR}/annotation" >&2; exit 1; }

if [ "${1:-}" = "--launch-shards" ]; then
  NUM_SHARDS="${NUM_SHARDS:-4}"; GPUS="${GPUS:-0,1,2,3}"; IFS=',' read -ra GPU_ARR <<< "${GPUS}"
  mkdir -p "${ROOT}/exp_logs/tmux_logs"
  for ((i = 0; i < NUM_SHARDS; i++)); do
    gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"; log="${ROOT}/exp_logs/tmux_logs/cw_hum_grid_4view_radial_shard${i}.log"
    tmux new-session -d -s "cw_h4rad_s${i}" \
      "CUDA_VISIBLE_DEVICES=${gpu} NUM_SHARDS=${NUM_SHARDS} SHARD_ID=${i} CKPT='${CKPT}' SPLIT='${SPLIT}' \
       DOWN_SAMPLE='${DOWN_SAMPLE}' OUTPUT_ROOT='${OUTPUT_ROOT}' INPUT_ROOT='${INPUT_ROOT}' \
       RADIAL_DECAY='${RADIAL_DECAY}' RADIAL_BLOCK_SIZE='${RADIAL_BLOCK_SIZE}' \
       RADIAL_FIRST_LAYERS_FP='${RADIAL_FIRST_LAYERS_FP}' RADIAL_MODEL_TYPE='${RADIAL_MODEL_TYPE}' \
       bash scripts/run_humanoid_multiview_ctrlworld_4view_grid_radial.sh 2>&1 | tee '${log}'; exec bash"
    echo "  shard ${i} -> GPU ${gpu}  (tmux cw_h4rad_s${i}, log ${log})"
  done
  echo "Merge after: python scripts/merge_ctrlworld_shard_summaries.py --output-root ${OUTPUT_ROOT}"; exit 0
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CONFIG=humanoid_grid DATASET_NAME="${DATASET_NAME}" CKPT="${CKPT}" DOWN_SAMPLE="${DOWN_SAMPLE}" SPLIT="${SPLIT}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" INPUT_ROOT="${INPUT_ROOT}" \
RADIAL_DECAY="${RADIAL_DECAY}" RADIAL_BLOCK_SIZE="${RADIAL_BLOCK_SIZE}" \
RADIAL_FIRST_LAYERS_FP="${RADIAL_FIRST_LAYERS_FP}" RADIAL_MODEL_TYPE="${RADIAL_MODEL_TYPE}" \
  bash scripts/run_humanoid_multiview_ctrlworld_radial.sh

[ "${NUM_SHARDS:-1}" -le 1 ] && python scripts/merge_ctrlworld_shard_summaries.py --output-root "${OUTPUT_ROOT}" || true
