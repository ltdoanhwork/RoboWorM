set -e
set -u
set -o pipefail


COSMOS_PREDICT_PATH="/mnt/data/sftp/data/vla_intern/workspace/binh/dreamgen/dreamgen"
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# DATASET_PATH="/work/HHRI-AI/POC/public/pretraining_data/Robotic-Data/Video-Simulation-Data/robocasa/robocasa_30_demos_lerobot_5_chosen_tasks_v3-training-format"
DATASET_PATH="/mnt/data/sftp/data/vla_intern/workspace/binh/dreamgen/dreamgen/datasets/realworld-for-dg-training-v1"

# EXPERIMENT="debug"F
EXPERIMENT="predict2_video2world_training_2b_groot_gr1_480"
# NUM_GPUS=1 #4
NUM_GPUS=4
MASTER_PORT=12341

unset NCCL_SOCKET_IFNAME
unset NCCL_IB_HCA


# conda activate dg2
export PATH="/home/binhng/conda_setup/miniconda3/envs/dg2/bin:$PATH"

# export PYTHONPATH="${COSMOS_PREDICT_PATH}:${PYTHONPATH:-}"
export PYTHONPATH="${PYTHONPATH:-}:${COSMOS_PREDICT_PATH}"

export CUDA_HOME=/mnt/data/sftp/data/vla_intern/workspace/binh/dreamgen/cuda_setup/cuda_12.4
# export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# export NCCL_SOCKET_IFNAME="ib0"
# export NCCL_IB_DISABLE=0
# export NCCL_DEBUG=INFO
# export NCCL_P2P_DISABLE=1 \
# export NCCL_SHM_DISABLE=1 \
# export NCCL_NET_DISABLE=1 


# export NCCL_IB_DISABLE=1

# Keep everything on localhost

unset NCCL_SOCKET_IFNAME
unset NCCL_IB_HCA
export NCCL_DEBUG=INFO

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12341
export NCCL_SOCKET_IFNAME=eth0 
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# Optional: get clearer logs
# export NCCL_DEBUG=INFO

# export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# export NCCL_SOCKET_IFNAME="ib0"
# export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO

export CUDA_VISIBLE_DEVICES=0,1,2,3
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# export WANDB_API_KEY="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"
export WANDB_API_KEY="4d7d3d768aa4a5e26e287875fad303a3c4586fc6"
# export WANDB_ENTITY="dreamgen"
# export WANDB_PROJECT="Robotics_VLA"
export WANDB_ENTITY="Robotics_VLA"
export WANDB_PROJECT="dreamgen"

export BATCH_SIZE=1
BATCH_SIZE=1 #${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-8}
MAX_STEPS=${MAX_STEPS:-""}

#############################################################################################################
# =============================================================================
# VALIDATION
# =============================================================================

echo "============================================"
echo "Cosmos Predict2 Training Configuration"
echo "============================================"
echo "Cosmos Predict Path: $COSMOS_PREDICT_PATH"
echo "Dataset Path: $DATASET_PATH"
echo "Experiment: $EXPERIMENT"
echo "Number of GPUs: $NUM_GPUS"
echo "W&B Project: $WANDB_PROJECT"
echo "W&B Entity: $WANDB_ENTITY"
echo "============================================"

echo "Verifying paths and requirements..."

if [ ! -d "$COSMOS_PREDICT_PATH" ]; then
    echo "ERROR: Cosmos Predict directory does not exist: $COSMOS_PREDICT_PATH"
    exit 1
fi

if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset directory does not exist: $DATASET_PATH"
    exit 1
fi

METAS_DIR="${DATASET_PATH}/metas"
VIDEOS_DIR="${DATASET_PATH}/videos"
T5_DIR="${DATASET_PATH}/t5_xxl"

if [ ! -d "$METAS_DIR" ]; then
    echo "ERROR: Metas directory does not exist: $METAS_DIR"
    exit 1
fi

if [ ! -d "$VIDEOS_DIR" ]; then
    echo "ERROR: Videos directory does not exist: $VIDEOS_DIR"
    exit 1
fi

if [ ! -d "$T5_DIR" ]; then
    echo "ERROR: T5 embeddings directory does not exist: $T5_DIR"
    echo "Please run the T5 embedding extraction script first!"
    exit 1
fi

# Count dataset files
txt_files=$(find "$METAS_DIR" -name "*.txt" | wc -l)
mp4_files=$(find "$VIDEOS_DIR" -name "*.mp4" | wc -l)
pickle_files=$(find "$T5_DIR" -name "*.pickle" | wc -l)

echo "Dataset validation:"
echo "  - Text files: $txt_files"
echo "  - Video files: $mp4_files"
echo "  - T5 embeddings: $pickle_files"

if [ $pickle_files -eq 0 ]; then
    echo "ERROR: No T5 embedding files found. Please run embedding extraction first!"
    exit 1
fi

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. CUDA not available?"
    exit 1
fi

available_gpus=$(nvidia-smi --list-gpus | wc -l)
echo "Available GPUs: $available_gpus"

if [ $available_gpus -lt $NUM_GPUS ]; then
    echo "ERROR: Requested $NUM_GPUS GPUs but only $available_gpus available"
    exit 1
fi

echo "All validations passed!"
echo ""
########################################################################################

# =============================================================================
# TRAINING SETUP
# =============================================================================

# Change to the Cosmos Predict directory
echo "Changing to Cosmos Predict directory..."
cd "$COSMOS_PREDICT_PATH"

FILTERED_DATASET_PATH="${DATASET_PATH}_filtered"

if [ -d "$FILTERED_DATASET_PATH" ]; then
    echo "✅ Filtered dataset already exists at: $FILTERED_DATASET_PATH"
    video_count=$(find "$FILTERED_DATASET_PATH/videos" -type f -name "*.mp4" 2>/dev/null | wc -l)
    if [ "$video_count" -gt 0 ]; then
        echo "Found $video_count videos. Skipping the filtering process."
        echo "💡 To force re-filtering, delete the directory: rm -rf \"$FILTERED_DATASET_PATH\""
    else
        echo "⚠️ WARNING: Filtered directory exists but is empty. Proceeding to re-filter."
    fi
else
    echo "Filtered dataset not found. Starting video filtering (need 93+ frames)..."
    pip install --user ffmpegcv opencv-python-headless 2>/dev/null || echo "Dependencies already installed"

    FILTERED_METAS_DIR="${FILTERED_DATASET_PATH}/metas"
    FILTERED_VIDEOS_DIR="${FILTERED_DATASET_PATH}/videos"
    FILTERED_T5_DIR="${FILTERED_DATASET_PATH}/t5_xxl"
    mkdir -p "$FILTERED_METAS_DIR" "$FILTERED_VIDEOS_DIR" "$FILTERED_T5_DIR"

    python3 -c "
import os, cv2, shutil
from pathlib import Path

def get_frame_count(video_path):
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): return 0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return count
    except Exception:
        return 0

videos_dir = Path('$VIDEOS_DIR')
metas_dir = Path('$METAS_DIR')
t5_dir = Path('$T5_DIR')
filtered_videos_dir = Path('$FILTERED_VIDEOS_DIR')
filtered_metas_dir = Path('$FILTERED_METAS_DIR')
filtered_t5_dir = Path('$FILTERED_T5_DIR')
MIN_FRAMES = 100
valid_count, total_count = 0, 0

print(f'Starting video filtering (minimum {MIN_FRAMES} frames required)...')
for video_file in sorted(videos_dir.glob('*.mp4')):
    total_count += 1
    if get_frame_count(video_file) >= MIN_FRAMES:
        base_name = video_file.stem
        meta_file = metas_dir / f'{base_name}.txt'
        t5_file = t5_dir / f'{base_name}.pickle'
        if meta_file.exists() and t5_file.exists():
            shutil.copy2(video_file, filtered_videos_dir)
            shutil.copy2(meta_file, filtered_metas_dir)
            shutil.copy2(t5_file, filtered_t5_dir)
            valid_count += 1
    if total_count % 100 == 0:
        print(f'Processed {total_count} videos, kept {valid_count}')

print(f'\nFiltering complete: {valid_count}/{total_count} videos kept')
if valid_count == 0:
    print('ERROR: No valid videos found after filtering!')
    import sys; sys.exit(1)
"
fi


# Update dataset path to use filtered dataset
DATASET_PATH="$FILTERED_DATASET_PATH"
echo "Using dataset path for training: $DATASET_PATH"

# Symlink dataset to expected path
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# EXPECTED_DATASET_PATH="datasets/benchmark_train/gr1"
# # EXPECTED_DATASET_PATH="datasets/robcoasa_draft_filtered"
# echo "Creating dataset symlink for config compatibility..."
# mkdir -p "$(dirname "$EXPECTED_DATASET_PATH")"
# rm -rf "$EXPECTED_DATASET_PATH"
# ln -s "$(readlink -f "$DATASET_PATH")" "$EXPECTED_DATASET_PATH"
# echo "Created symlink: $EXPECTED_DATASET_PATH -> $(readlink -f "$DATASET_PATH")"


# -------------------- Symlink to expected path --------------------
EXPECTED_DATASET_PATH="datasets/benchmark_train/gr1"
echo "Creating dataset symlink for config compatibility..."
mkdir -p "$(dirname "$EXPECTED_DATASET_PATH")"
rm -rf "$EXPECTED_DATASET_PATH"
ln -s "$(readlink -f "$DATASET_PATH")" "$EXPECTED_DATASET_PATH"

# Verify symlink
if [[ -L "$EXPECTED_DATASET_PATH" ]]; then
  echo "Created symlink: $EXPECTED_DATASET_PATH -> $(readlink -f "$DATASET_PATH")"
else
  echo "ERROR: Failed to create symlink at $EXPECTED_DATASET_PATH" >&2
  exit 1
fi




# Create output and log directories
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/mnt/data/sftp/data/vla_intern/workspace/binh/dreamgen/dreamgen/outputs/training_${EXPERIMENT}_${TIMESTAMP}"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
echo "Training output directory: $OUTPUT_DIR"


# =============================================================================
# BUILD & EXECUTE TRAINING COMMAND
# =============================================================================
HYDRA_RUN_DIR="$OUTPUT_DIR/hydra_runs"
mkdir -p "$HYDRA_RUN_DIR"

TORCHRUN_CMD="torchrun --standalone --nproc_per_node=$NUM_GPUS --nnodes=1"
# TORCHRUN_CMD="python -m torch.distributed.run --standalone --nproc_per_node=$NUM_GPUS --nnodes=1"

# TRAIN_CMD="-m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=${EXPERIMENT}"
TRAIN_CMD="-m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=${EXPERIMENT}"
if [ -n "$MAX_STEPS" ]; then
    TRAIN_CMD="$TRAIN_CMD max_steps=${MAX_STEPS}"
fi
TRAIN_CMD="$TRAIN_CMD hydra.run.dir=${HYDRA_RUN_DIR} hydra.job.chdir=True"
FULL_CMD="$TORCHRUN_CMD $TRAIN_CMD"
pwd
echo "============================================"
echo "Starting Training"
echo "Command: $FULL_CMD"
echo "Logs will be saved to: $LOG_DIR"
echo "============================================"

# Save command to log file
echo "Command: $FULL_CMD" > "$LOG_DIR/training_command.log"
env | grep -E "(CUDA|NCCL|PYTHON|WANDB)" >> "$LOG_DIR/training_command.log"

# Execute training with logging
exec &> >(tee -a "$LOG_DIR/training_output.log")
echo "Training started at: $(date)"
$FULL_CMD

if [ $? -eq 0 ]; then
    echo -e "\n============================================\nTraining completed successfully!\nOutput directory: $OUTPUT_DIR\n============================================"
else
    echo -e "\n============================================\nTraining failed! Check logs in: $LOG_DIR\n============================================"
    exit 1
fi