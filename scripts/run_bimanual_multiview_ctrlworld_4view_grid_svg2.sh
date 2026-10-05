#!/bin/bash
# SVG2 (SAP) sparse attention on the bimanual 4-VIEW 2x2 GRID Ctrl-World model.
# Mirrors run_bimanual_multiview_ctrlworld_4view_grid.sh (dense), delegating to the
# shared base svg2 script with CONFIG=bimanual_grid. See the 4view-grid sparse-attn rule.
# NOTE: env DOES have flashinfer (0.3.1) + flex_attention (torch 2.7.1) — verified
# 2026-08-17. SVG2 genuinely runs the flashinfer block-sparse path (log prints
# "Installed SVG FlashInfer gated monkey patch"), NOT an SDPA fallback. Kept as a
# valid benchmark point (it is still net-negative here because T=3840 is too short
# for block-sparse to win — see the 4view-grid sparse-attn rule).
#
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/run_bimanual_multiview_ctrlworld_4view_grid_svg2.sh
#        NUM_SHARDS=4 bash scripts/run_bimanual_multiview_ctrlworld_4view_grid_svg2.sh --launch-shards
# Env: CKPT, SPLIT, NUM_EPISODES, DOWN_SAMPLE, OUTPUT_ROOT/INPUT_ROOT, NUM_SHARDS/SHARD_ID,
#      GPUS, SVG_{NUM_Q_CENTROIDS,NUM_K_CENTROIDS,TOP_P_KMEANS,FIRST_LAYERS_FP}

set -e
ROOT="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics"
cd "${ROOT}"

DATASET_NAME="${DATASET_NAME:-bimanual_multiview_4view_grid}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/models/Ctrl-World/dataset_example/${DATASET_NAME}}"; export DATASET_DIR
CKPT="${CKPT:-${ROOT}/exp_logs/ctrlworld_logs/bimanual_multiview_4view_grid/checkpoint-110000.pt}"
DOWN_SAMPLE="${DOWN_SAMPLE:-6}"; SPLIT="${SPLIT:-all}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/sampling_dataset/svg2/bimanual/output/multiview/ctrlworld_4view_grid}"
INPUT_ROOT="${INPUT_ROOT:-${ROOT}/sampling_dataset/svg2/bimanual/input/multiview/ctrlworld_4view_grid}"
SVG_NUM_Q_CENTROIDS="${SVG_NUM_Q_CENTROIDS:-32}"; SVG_NUM_K_CENTROIDS="${SVG_NUM_K_CENTROIDS:-64}"
SVG_TOP_P_KMEANS="${SVG_TOP_P_KMEANS:-0.7}"; SVG_FIRST_LAYERS_FP="${SVG_FIRST_LAYERS_FP:-2}"

[ -f "${CKPT}" ] || { echo "ERROR: checkpoint not found: ${CKPT}" >&2; exit 1; }
[ -d "${DATASET_DIR}/annotation" ] || { echo "ERROR: annotation dir not found: ${DATASET_DIR}/annotation" >&2; exit 1; }

if [ "${1:-}" = "--launch-shards" ]; then
  NUM_SHARDS="${NUM_SHARDS:-4}"; GPUS="${GPUS:-0,1,2,3}"; IFS=',' read -ra GPU_ARR <<< "${GPUS}"
  mkdir -p "${ROOT}/exp_logs/tmux_logs"
  for ((i = 0; i < NUM_SHARDS; i++)); do
    gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"; log="${ROOT}/exp_logs/tmux_logs/cw_bim_grid_4view_svg2_shard${i}.log"
    tmux new-session -d -s "cw_b4svg2_s${i}" \
      "CUDA_VISIBLE_DEVICES=${gpu} NUM_SHARDS=${NUM_SHARDS} SHARD_ID=${i} CKPT='${CKPT}' SPLIT='${SPLIT}' \
       DOWN_SAMPLE='${DOWN_SAMPLE}' OUTPUT_ROOT='${OUTPUT_ROOT}' INPUT_ROOT='${INPUT_ROOT}' \
       SVG_NUM_Q_CENTROIDS='${SVG_NUM_Q_CENTROIDS}' SVG_NUM_K_CENTROIDS='${SVG_NUM_K_CENTROIDS}' \
       SVG_TOP_P_KMEANS='${SVG_TOP_P_KMEANS}' SVG_FIRST_LAYERS_FP='${SVG_FIRST_LAYERS_FP}' \
       bash scripts/run_bimanual_multiview_ctrlworld_4view_grid_svg2.sh 2>&1 | tee '${log}'; exec bash"
    echo "  shard ${i} -> GPU ${gpu}  (tmux cw_b4svg2_s${i}, log ${log})"
  done
  echo "Merge after: python scripts/merge_ctrlworld_shard_summaries.py --output-root ${OUTPUT_ROOT}"; exit 0
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CONFIG=bimanual_grid DATASET_NAME="${DATASET_NAME}" CKPT="${CKPT}" DOWN_SAMPLE="${DOWN_SAMPLE}" SPLIT="${SPLIT}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" INPUT_ROOT="${INPUT_ROOT}" \
NUM_EPISODES="${NUM_EPISODES:-}" \
NUM_EPISODES_PER_CATEGORY="${NUM_EPISODES_PER_CATEGORY:-}" \
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-}" \
GUIDANCE="${GUIDANCE:-}" \
SVG_NUM_Q_CENTROIDS="${SVG_NUM_Q_CENTROIDS}" SVG_NUM_K_CENTROIDS="${SVG_NUM_K_CENTROIDS}" \
SVG_TOP_P_KMEANS="${SVG_TOP_P_KMEANS}" SVG_FIRST_LAYERS_FP="${SVG_FIRST_LAYERS_FP}" \
  bash scripts/run_humanoid_multiview_ctrlworld_svg2.sh

[ "${NUM_SHARDS:-1}" -le 1 ] && python scripts/merge_ctrlworld_shard_summaries.py --output-root "${OUTPUT_ROOT}" || true
