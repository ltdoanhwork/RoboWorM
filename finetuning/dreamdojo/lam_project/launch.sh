set -ex

NNODES=${NNODES:-1}
NPROC=${NPROC:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-12341}
NODE_RANK=${NODE_RANK:-0}
SEED=${SEED:-42}

CONFIG_INDEX=$1

export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FI_EFA_USE_DEVICE_RDMA=1
export RDMAV_FORK_SAFE=1
export TORCH_DIST_INIT_BARRIER=1

# CUDA 12 fix: Force PyTorch to use its bundled CUDA libraries
export CUDA_MODULE_LOADING=LAZY
export LD_PRELOAD=""  # Clear any preloaded libraries

echo "Running on $NNODES nodes with $NPROC processes per node. This node rank is $NODE_RANK."

export PYTHONPATH=/mnt/amlfs-01/shared/shenyuang/DreamDojo/external/lam_project:$PYTHONPATH
export OMP_NUM_THREADS=8

source /mnt/amlfs-01/shared/shenyuang/DreamDojo/.venv/bin/activate

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC \
    --master_port=$MASTER_PORT --master_addr $MASTER_ADDR \
    --node_rank=$NODE_RANK \
    main.py fit \
    --config config/lam_${CONFIG_INDEX}.yaml \
    2>&1 | tee output_train.log