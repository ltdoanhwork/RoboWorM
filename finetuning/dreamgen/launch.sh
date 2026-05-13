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
DREAMGEN_ROOT="${PROJECT_ROOT}/models/dreamgen"

export PYTHONPATH="${DREAMGEN_ROOT}:${PROJECT_ROOT}:${PYTHONPATH}"
export OMP_NUM_THREADS=8
export IMAGINAIRE_OUTPUT_ROOT="${PROJECT_ROOT}/exp_logs/dreamgen_logs"

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
