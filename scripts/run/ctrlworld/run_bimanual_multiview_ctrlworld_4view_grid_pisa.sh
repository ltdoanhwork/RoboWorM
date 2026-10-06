#!/bin/bash
# PISA sparse attention on the bimanual 4-VIEW 2x2 GRID Ctrl-World model.
# Mirrors run_bimanual_multiview_ctrlworld_4view_grid.sh (dense), delegating to the
# shared base pisa script with CONFIG=bimanual_grid. Resumable. See the 4view-grid
# sparse-attention rule.
#
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/run/ctrlworld/run_bimanual_multiview_ctrlworld_4view_grid_pisa.sh
#        NUM_SHARDS=4 bash scripts/run/ctrlworld/run_bimanual_multiview_ctrlworld_4view_grid_pisa.sh --launch-shards
# Env: CKPT, SPLIT, NUM_EPISODES, DOWN_SAMPLE, OUTPUT_ROOT/INPUT_ROOT, NUM_SHARDS/SHARD_ID,
#      GPUS, PISA_{DENSITY,BLOCK_SIZE,START_LAYER}

set -e
ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "${ROOT}"

DATASET_NAME="${DATASET_NAME:-bimanual_multiview_4view_grid}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/models/Ctrl-World/dataset_example/${DATASET_NAME}}"; export DATASET_DIR
CKPT="${CKPT:-${ROOT}/checkpoints/finetuned/ctrlworld/bimanual_multiview_4view_grid/checkpoint-110000.pt}"
DOWN_SAMPLE="${DOWN_SAMPLE:-6}"; SPLIT="${SPLIT:-all}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/sampling_dataset/pisa/bimanual/output/multiview/ctrlworld_4view_grid}"
INPUT_ROOT="${INPUT_ROOT:-${ROOT}/sampling_dataset/pisa/bimanual/input/multiview/ctrlworld_4view_grid}"
PISA_DENSITY="${PISA_DENSITY:-0.2}"; PISA_BLOCK_SIZE="${PISA_BLOCK_SIZE:-64}"; PISA_START_LAYER="${PISA_START_LAYER:-0}"

[ -f "${CKPT}" ] || { echo "ERROR: checkpoint not found: ${CKPT}" >&2; exit 1; }
[ -d "${DATASET_DIR}/annotation" ] || { echo "ERROR: annotation dir not found: ${DATASET_DIR}/annotation" >&2; exit 1; }

if [ "${1:-}" = "--launch-shards" ]; then
  NUM_SHARDS="${NUM_SHARDS:-4}"; GPUS="${GPUS:-0,1,2,3}"; IFS=',' read -ra GPU_ARR <<< "${GPUS}"
  mkdir -p "${ROOT}/logs/exp_logs/tmux_logs"
  for ((i = 0; i < NUM_SHARDS; i++)); do
    gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"; log="${ROOT}/logs/exp_logs/tmux_logs/cw_bim_grid_4view_pisa_shard${i}.log"
    tmux new-session -d -s "cw_b4pisa_s${i}" \
      "CUDA_VISIBLE_DEVICES=${gpu} NUM_SHARDS=${NUM_SHARDS} SHARD_ID=${i} CKPT='${CKPT}' SPLIT='${SPLIT}' \
       DOWN_SAMPLE='${DOWN_SAMPLE}' OUTPUT_ROOT='${OUTPUT_ROOT}' INPUT_ROOT='${INPUT_ROOT}' \
       PISA_DENSITY='${PISA_DENSITY}' PISA_BLOCK_SIZE='${PISA_BLOCK_SIZE}' PISA_START_LAYER='${PISA_START_LAYER}' \
       bash scripts/run/ctrlworld/run_bimanual_multiview_ctrlworld_4view_grid_pisa.sh 2>&1 | tee '${log}'; exec bash"
    echo "  shard ${i} -> GPU ${gpu}  (tmux cw_b4pisa_s${i}, log ${log})"
  done
  echo "Merge after: python scripts/tools/ops/merge_ctrlworld_shard_summaries.py --output-root ${OUTPUT_ROOT}"; exit 0
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CONFIG=bimanual_grid DATASET_NAME="${DATASET_NAME}" CKPT="${CKPT}" DOWN_SAMPLE="${DOWN_SAMPLE}" SPLIT="${SPLIT}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" INPUT_ROOT="${INPUT_ROOT}" \
NUM_EPISODES="${NUM_EPISODES:-}" \
NUM_EPISODES_PER_CATEGORY="${NUM_EPISODES_PER_CATEGORY:-}" \
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-}" \
GUIDANCE="${GUIDANCE:-}" \
PISA_DENSITY="${PISA_DENSITY}" PISA_BLOCK_SIZE="${PISA_BLOCK_SIZE}" PISA_START_LAYER="${PISA_START_LAYER}" \
  bash scripts/run/ctrlworld/run_humanoid_multiview_ctrlworld_pisa.sh

[ "${NUM_SHARDS:-1}" -le 1 ] && python scripts/tools/ops/merge_ctrlworld_shard_summaries.py --output-root "${OUTPUT_ROOT}" || true
