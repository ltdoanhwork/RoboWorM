#!/bin/bash

# Cosmos Predict2 Training Script - Single Node (8 GPUs)
# This script runs post-training for Cosmos-Predict2-2B-Video2World model

# Set strict error handling
set -e
set -u
set -o pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

# Define paths
COSMOS_PREDICT_PATH="/work/HHRI-AI/POC/public/pretraining_weights/NVIDIA/Cosmos-Predict/cosmos-predict2"
DATASET_PATH="/work/HHRI-AI/POC/public/pretraining_data/Robotic-Data/Video-Simulation-Data/robocasa/robocasa_30_demos_lerobot_5_chosen_tasks_v3-training-format"

# Training configuration
EXPERIMENT="predict2_video2world_training_2b_groot_gr1_480"
NUM_GPUS=4
MASTER_PORT=12341

# =============================================================================
# PREREQUISITES & ENVIRONMENT
# =============================================================================

# --- CORRECTION 3: Install wandb reliably at the start ---
echo "Ensuring required packages are installed..."
pip install --user wandb

# Export PYTHONPATH
export PYTHONPATH="${COSMOS_PREDICT_PATH}:${PYTHONPATH:-}"

# Set CUDA settings
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_SOCKET_IFNAME="ib0"
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO

# =============================================================================
# W&B CONFIGURATION (Weights & Biases)
# =============================================================================

# --- CORRECTION 1 & 2: Use placeholders for security and correctness ---
# 🚨 IMPORTANT: Replace placeholders with your actual W&B information.
# It's recommended to set WANDB_API_KEY in your ~/.bashrc for better security.
export WANDB_API_KEY="64b5263e8e66be917f988a4fc03084b014f95c8c"

# The ENTITY is your W&B username or team name.
export WANDB_ENTITY="tvnhiemhcmus"

# The PROJECT is the name of the project dashboard where runs will be logged.
export WANDB_PROJECT="Robotic-Cosmos-Predict-Project"
# =============================================================================

# Optional: Override default settings
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-8}
MAX_STEPS=${MAX_STEPS:-""}

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

# ... (The rest of your script from here is correct and does not need changes) ...
# ... (Validation, Video Filtering, Symlinking, and Execution sections) ...

# Verify paths exist
echo "Verifying paths and requirements..."

if [ ! -d "$COSMOS_PREDICT_PATH" ]; then
    echo "ERROR: Cosmos Predict directory does not exist: $COSMOS_PREDICT_PATH"
    exit 1
fi

if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset directory does not exist: $DATASET_PATH"
    exit 1
fi

# Check for required directories in dataset
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

# =============================================================================
# TRAINING SETUP
# =============================================================================

# Change to the Cosmos Predict directory
echo "Changing to Cosmos Predict directory..."
cd "$COSMOS_PREDICT_PATH"

# =============================================================================
# VIDEO FILTERING - Conditionally filter to avoid re-processing
# =============================================================================

# Define the path for the filtered dataset
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
# EXPECTED_DATASET_PATH="datasets/benchmark_train/gr1"
EXPECTED_DATASET_PATH="datasets/benchmark_train/gr1"
echo "Creating dataset symlink for config compatibility..."
mkdir -p "$(dirname "$EXPECTED_DATASET_PATH")"
rm -rf "$EXPECTED_DATASET_PATH"
ln -s "$(readlink -f "$DATASET_PATH")" "$EXPECTED_DATASET_PATH"
echo "Created symlink: $EXPECTED_DATASET_PATH -> $(readlink -f "$DATASET_PATH")"

# Create output and log directories
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/work/HHRI-AI/POC/public/pretraining_weights/NVIDIA/Cosmos-Predict/cosmos-predict2/outputs/training_${EXPERIMENT}_${TIMESTAMP}"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
echo "Training output directory: $OUTPUT_DIR"

# =============================================================================
# BUILD & EXECUTE TRAINING COMMAND
# =============================================================================
HYDRA_RUN_DIR="$OUTPUT_DIR/hydra_runs"
mkdir -p "$HYDRA_RUN_DIR"

TORCHRUN_CMD="torchrun --standalone --nproc_per_node=$NUM_GPUS --nnodes=1"
TRAIN_CMD="-m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=${EXPERIMENT}"
if [ -n "$MAX_STEPS" ]; then
    TRAIN_CMD="$TRAIN_CMD max_steps=${MAX_STEPS}"
fi
TRAIN_CMD="$TRAIN_CMD hydra.run.dir=${HYDRA_RUN_DIR} hydra.job.chdir=True"
FULL_CMD="$TORCHRUN_CMD $TRAIN_CMD"

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