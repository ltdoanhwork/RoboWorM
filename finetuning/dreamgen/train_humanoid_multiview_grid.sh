#!/bin/bash
# End-to-end: prepare grid dataset → compute T5 embeddings → launch training
# for humanoid multiview (2x2 grid, 4 cameras).
#
# Usage:
#   NPROC=8 bash finetuning/dreamgen/train_humanoid_multiview_grid.sh
#   NPROC=4 bash finetuning/dreamgen/train_humanoid_multiview_grid.sh
set -ex

# --- Config ---
NNODES=${NNODES:-1}
NPROC=${NPROC:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-12341}
NODE_RANK=${NODE_RANK:-0}

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DREAMGEN_ROOT="${PROJECT_ROOT}/models/dreamgen"

# --- Environment ---
source "${DREAMGEN_ROOT}/.venv/bin/activate"
export PYTHONPATH="${DREAMGEN_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
export COSMOS_PREDICT2_ARGS="--checkpoints ${PROJECT_ROOT}/checkpoints"
export IMAGINAIRE_OUTPUT_ROOT="${PROJECT_ROOT}/exp_logs/dreamgen_logs"
export OMP_NUM_THREADS=8

export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FI_EFA_USE_DEVICE_RDMA=1
export RDMAV_FORK_SAFE=1
export TORCH_DIST_INIT_BARRIER=1
export CUDA_MODULE_LOADING=LAZY
export LD_PRELOAD=""

DATASET_PATH="${PROJECT_ROOT}/datasets/humanoid_multiview_grid_dreamgen"
EXPERIMENT="predict2_video2world_training_2b_humanoid_multiview"

echo "============================================"
echo "Humanoid Multiview Grid Training Pipeline"
echo "  Project:    ${PROJECT_ROOT}"
echo "  Dataset:    ${DATASET_PATH}"
echo "  Experiment: ${EXPERIMENT}"
echo "  GPUs:       ${NPROC} x ${NNODES} nodes"
echo "============================================"

# --- Step 1: Prepare grid dataset ---
echo ""
echo "=== [STEP 1/3] Preparing 2x2 grid dataset ==="
python "${PROJECT_ROOT}/finetuning/dreamgen/scripts/prepare_humanoid_multiview_grid_dreamgen.py" \
  --source "${PROJECT_ROOT}/datasets/humanoid/multiview" \
  --target "${DATASET_PATH}"

n_videos=$(ls "${DATASET_PATH}/videos/" 2>/dev/null | wc -l)
n_metas=$(ls "${DATASET_PATH}/metas/" 2>/dev/null | wc -l)
echo "  Grid videos: ${n_videos}, Metas: ${n_metas}"

if [ "${n_videos}" -eq 0 ]; then
  echo "ERROR: No grid videos created. Check source dataset." >&2
  exit 1
fi

# --- Step 2: Compute T5 embeddings ---
echo ""
echo "=== [STEP 2/3] Computing T5-XXL embeddings ==="
cd "${DREAMGEN_ROOT}"
python -m scripts.get_t5_embeddings \
  --dataset_path "${DATASET_PATH}"
echo "  T5 embeddings saved to: ${DATASET_PATH}/t5_xxl/"

# --- Step 3: Launch training ---
echo ""
echo "=== [STEP 3/3] Launching training: ${EXPERIMENT} ==="
echo "  ${NPROC} GPUs, ${NNODES} node(s)"

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC}" \
    --master_port="${MASTER_PORT}" \
    --master_addr="${MASTER_ADDR}" \
    --node_rank="${NODE_RANK}" \
    -m scripts.train \
    --config=cosmos_predict2/configs/base/config.py -- \
    experiment="${EXPERIMENT}"
