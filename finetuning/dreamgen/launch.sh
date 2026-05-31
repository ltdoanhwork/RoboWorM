#!/bin/bash
# DreamGen (Cosmos-Predict2 Video2World) post-training launcher.
#
# Usage:
#   NPROC=8 bash finetuning/dreamgen/launch.sh <experiment_name>
#
# Single-GPU debug:
#   NPROC=1 bash finetuning/dreamgen/launch.sh <experiment_name>
#
# Multi-node (set NNODES, MASTER_ADDR, NODE_RANK externally):
#   NNODES=4 NPROC=8 MASTER_ADDR=<head_ip> NODE_RANK=0 \
#     bash finetuning/dreamgen/launch.sh <experiment_name>
#
# Available single-view experiments:
#   predict2_video2world_training_2b_bimanual_singleview
#   predict2_video2world_training_2b_single_arm_singleview
#   predict2_video2world_training_2b_humanoid_singleview
#   predict2_video2world_training_2b_all_singleview        ← all 3 combined (ConcatDataset)
#
# Available multi-view experiments (4 cameras as independent samples):
#   predict2_video2world_training_2b_humanoid_multiview
#   predict2_video2world_training_2b_bimanual_multiview
#   predict2_video2world_training_2b_all_multiview         ← humanoid + bimanual combined
#
# Prerequisites:
#   1. Run data preparation scripts in finetuning/dreamgen/scripts/
#   2. Run bash finetuning/dreamgen/scripts/compute_t5_embeddings.sh
#   3. Ensure DreamGen venv installed at models/dreamgen/.venv/

set -ex

NNODES=${NNODES:-1}
NPROC=${NPROC:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-12341}
NODE_RANK=${NODE_RANK:-0}

export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FI_EFA_USE_DEVICE_RDMA=1
export RDMAV_FORK_SAFE=1
export TORCH_DIST_INIT_BARRIER=1

# CUDA 12 fix: avoid loading system CUDA libraries over PyTorch bundled ones
export CUDA_MODULE_LOADING=LAZY
export LD_PRELOAD=""

echo "Running on $NNODES nodes with $NPROC processes per node. Node rank: $NODE_RANK."

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PROJECT_ROOT
DREAMGEN_ROOT="${PROJECT_ROOT}/models/dreamgen"

export PYTHONPATH="${DREAMGEN_ROOT}:${PROJECT_ROOT}:${PYTHONPATH}"
export OMP_NUM_THREADS=8
export IMAGINAIRE_OUTPUT_ROOT="${PROJECT_ROOT}/exp_logs/dreamgen_logs"

# Point constants.py to local checkpoints dir (contains nvidia/Cosmos-Predict2-*, google-t5/t5-11b, etc.)
export COSMOS_PREDICT2_ARGS="--checkpoints ${PROJECT_ROOT}/checkpoints"

source "${DREAMGEN_ROOT}/.venv/bin/activate"

experiment_name=$1
if [ -z "$experiment_name" ]; then
    echo "ERROR: experiment name required."
    echo "Usage: bash $0 <experiment_name>"
    echo ""
    echo "Available singleview experiments:"
    echo "  predict2_video2world_training_2b_bimanual_singleview"
    echo "  predict2_video2world_training_2b_single_arm_singleview"
    echo "  predict2_video2world_training_2b_humanoid_singleview"
    echo ""
    echo "Available singleview 16fps experiments (fps-normalized):"
    echo "  predict2_video2world_training_2b_single_arm_singleview_16fps"
    echo "  predict2_video2world_training_2b_single_arm_static_singleview_16fps"
    echo "  predict2_video2world_training_2b_bimanual_singleview_16fps"
    echo "  predict2_video2world_training_2b_arm_bimanual_singleview_16fps"
    echo ""
    echo "Available multiview experiments:"
    echo "  predict2_video2world_training_2b_humanoid_multiview"
    echo "  predict2_video2world_training_2b_bimanual_multiview"
    echo "  predict2_video2world_training_2b_all_multiview"
    exit 1
fi

cd "${DREAMGEN_ROOT}"

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC}" \
    --master_port="${MASTER_PORT}" \
    --master_addr="${MASTER_ADDR}" \
    --node_rank="${NODE_RANK}" \
    -m scripts.train \
    --config=cosmos_predict2/configs/base/config.py -- \
    experiment="${experiment_name}"
