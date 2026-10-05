#!/bin/bash
# End-to-end: prepare grid datasets (humanoid + bimanual) → compute T5 embeddings → launch training
# for all multiview (2x2 grid, 4 cameras per embodiment).
#
# Trains on ConcatDataset of humanoid_multiview_grid_dreamgen + bimanual_multiview_grid_dreamgen.
#
# Usage:
#   NPROC=4 bash finetuning/dreamgen/train_all_multiview_grid.sh
#   NPROC=8 bash finetuning/dreamgen/train_all_multiview_grid.sh
set -ex

# --- Config ---
NNODES=${NNODES:-1}
NPROC=${NPROC:-4}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-12341}
NODE_RANK=${NODE_RANK:-0}

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DREAMGEN_ROOT="${PROJECT_ROOT}/models/dreamgen"

# --- Environment ---
source "${DREAMGEN_ROOT}/.venv/bin/activate"
export PYTHONPATH="${DREAMGEN_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
export COSMOS_PREDICT2_ARGS="--checkpoints ${PROJECT_ROOT}/checkpoints/base"
export IMAGINAIRE_OUTPUT_ROOT="${PROJECT_ROOT}/checkpoints/finetuned/dreamgen"
export OMP_NUM_THREADS=8

export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FI_EFA_USE_DEVICE_RDMA=1
export RDMAV_FORK_SAFE=1
export TORCH_DIST_INIT_BARRIER=1
export CUDA_MODULE_LOADING=LAZY
export LD_PRELOAD=""

HUMANOID_DATASET_PATH="${PROJECT_ROOT}/datasets/humanoid_multiview_grid_dreamgen"
BIMANUAL_DATASET_PATH="${PROJECT_ROOT}/datasets/bimanual_multiview_grid_dreamgen"
EXPERIMENT="predict2_video2world_training_2b_all_multiview"

echo "============================================"
echo "All Multiview Grid Training Pipeline"
echo "  Project:    ${PROJECT_ROOT}"
echo "  Humanoid:   ${HUMANOID_DATASET_PATH}"
echo "  Bimanual:   ${BIMANUAL_DATASET_PATH}"
echo "  Experiment: ${EXPERIMENT}"
echo "  GPUs:       ${NPROC} x ${NNODES} nodes"
echo "============================================"

# --- Step 1a: Prepare humanoid grid dataset ---
echo ""
echo "=== [STEP 1/4] Preparing humanoid 2x2 grid dataset ==="
python "${PROJECT_ROOT}/finetuning/dreamgen/scripts/prepare_humanoid_multiview_grid_dreamgen.py" \
  --source "${PROJECT_ROOT}/datasets/humanoid/multiview" \
  --target "${HUMANOID_DATASET_PATH}"

n_videos=$(ls "${HUMANOID_DATASET_PATH}/videos/" 2>/dev/null | wc -l)
echo "  Humanoid grid videos: ${n_videos}"

# --- Step 1b: Prepare bimanual grid dataset ---
echo ""
echo "=== [STEP 2/4] Preparing bimanual 2x2 grid dataset ==="
python "${PROJECT_ROOT}/finetuning/dreamgen/scripts/prepare_bimanual_multiview_grid_dreamgen.py" \
  --source "${PROJECT_ROOT}/datasets/bimanual/multiview" \
  --target "${BIMANUAL_DATASET_PATH}"

n_videos_b=$(ls "${BIMANUAL_DATASET_PATH}/videos/" 2>/dev/null | wc -l)
echo "  Bimanual grid videos: ${n_videos_b}"

if [ "${n_videos}" -eq 0 ] && [ "${n_videos_b}" -eq 0 ]; then
  echo "ERROR: No grid videos created for either dataset. Check sources." >&2
  exit 1
fi

# --- Step 2: Compute T5 embeddings for both datasets ---
echo ""
echo "=== [STEP 3/4] Computing T5-XXL embeddings ==="
cd "${DREAMGEN_ROOT}"

echo "  Computing T5 for humanoid..."
python -m scripts.get_t5_embeddings \
  --dataset_path "${HUMANOID_DATASET_PATH}"
echo "  T5 embeddings saved to: ${HUMANOID_DATASET_PATH}/t5_xxl/"

echo "  Computing T5 for bimanual..."
python -m scripts.get_t5_embeddings \
  --dataset_path "${BIMANUAL_DATASET_PATH}"
echo "  T5 embeddings saved to: ${BIMANUAL_DATASET_PATH}/t5_xxl/"

# --- Step 3: Launch training ---
echo ""
echo "=== [STEP 4/4] Launching training: ${EXPERIMENT} ==="
echo "  ${NPROC} GPUs, ${NNODES} node(s)"
echo "  Dataset: humanoid_multiview_grid + bimanual_multiview_grid (ConcatDataset)"

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC}" \
    --master_port="${MASTER_PORT}" \
    --master_addr="${MASTER_ADDR}" \
    --node_rank="${NODE_RANK}" \
    -m scripts.train \
    --config=cosmos_predict2/configs/base/config.py -- \
    experiment="${EXPERIMENT}"
