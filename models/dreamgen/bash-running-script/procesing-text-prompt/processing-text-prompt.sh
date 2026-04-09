#!/bin/bash

# T5 Embeddings Extraction Script - Simple Multi-GPU Version
# This script extracts T5-XXL embeddings from the Groot dataset and saves them in pickle format

# Set strict error handling
set -e
set -u
set -o pipefail

# Define paths
COSMOS_PREDICT_PATH="/work/HHRI-AI/POC/public/pretraining_weights/NVIDIA/Cosmos-Predict/cosmos-predict2"
SCRIPT_PATH="${COSMOS_PREDICT_PATH}/scripts/get_t5_embeddings_from_groot_dataset_v1.py"
DATASET_PATH="/work/HHRI-AI/POC/public/pretraining_data/Robotic-Data/Video-Simulation-Data/robocasa/robocasa_100_demos_lerobot_5_chosen_tasks_v2-training-format"

# Export PYTHONPATH (handle case where PYTHONPATH might not be set)
export PYTHONPATH="${COSMOS_PREDICT_PATH}:${PYTHONPATH:-}"

# Print configuration for verification
echo "============================================"
echo "T5 Embeddings Extraction Configuration"
echo "============================================"
echo "Cosmos Predict Path: $COSMOS_PREDICT_PATH"
echo "Script Path: $SCRIPT_PATH"
echo "Dataset Path: $DATASET_PATH"
echo "PYTHONPATH: $PYTHONPATH"
echo "============================================"

# Verify paths exist
echo "Verifying paths..."

if [ ! -d "$COSMOS_PREDICT_PATH" ]; then
    echo "ERROR: Cosmos Predict directory does not exist: $COSMOS_PREDICT_PATH"
    exit 1
fi

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: Python script does not exist: $SCRIPT_PATH"
    exit 1
fi

if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset directory does not exist: $DATASET_PATH"
    exit 1
fi

echo "All paths verified successfully!"
echo ""

# Change to the Cosmos Predict directory
echo "Changing to Cosmos Predict directory..."
cd "$COSMOS_PREDICT_PATH"

# Verify that metas and videos directories exist
METAS_DIR="${DATASET_PATH}/metas"
VIDEOS_DIR="${DATASET_PATH}/videos"

echo "Checking for required subdirectories..."
if [ ! -d "$METAS_DIR" ]; then
    echo "ERROR: Metas directory does not exist: $METAS_DIR"
    exit 1
fi

if [ ! -d "$VIDEOS_DIR" ]; then
    echo "ERROR: Videos directory does not exist: $VIDEOS_DIR"
    exit 1
fi

# Count files for information
txt_files=$(find "$METAS_DIR" -name "*.txt" | wc -l)
mp4_files=$(find "$VIDEOS_DIR" -name "*.mp4" | wc -l)
echo "Found $txt_files text files and $mp4_files video files"
echo ""

# Create temporary directories for multi-GPU processing
NUM_GPUS=4

echo "Creating temporary dataset splits for $NUM_GPUS GPUs..."

temp_base_dir=$(mktemp -d)
echo "Using temporary directory: $temp_base_dir"

# Get all text files and sort them
txt_files_list=$(find "$METAS_DIR" -name "*.txt" | sort)
total_files=$(echo "$txt_files_list" | wc -l)
files_per_gpu=$((($total_files + $NUM_GPUS - 1) / $NUM_GPUS))

echo "Distributing $total_files files across $NUM_GPUS GPUs ($files_per_gpu files per GPU)"

# Create separate dataset directories for each GPU
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    gpu_dataset_dir="$temp_base_dir/gpu_${gpu_id}_dataset"
    gpu_metas_dir="$gpu_dataset_dir/metas"
    gpu_videos_dir="$gpu_dataset_dir/videos"
    gpu_t5_dir="$gpu_dataset_dir/t5_xxl"
    
    mkdir -p "$gpu_metas_dir"
    mkdir -p "$gpu_videos_dir"
    mkdir -p "$gpu_t5_dir"
done

# Distribute files among GPU directories
gpu_id=0
file_count=0

for txt_file in $txt_files_list; do
    base_name=$(basename "$txt_file" .txt)
    video_file="${VIDEOS_DIR}/${base_name}.mp4"
    
    if [ -f "$video_file" ]; then
        gpu_dataset_dir="$temp_base_dir/gpu_${gpu_id}_dataset"
        
        # Copy text file
        cp "$txt_file" "$gpu_dataset_dir/metas/"
        
        # Create symlink to video file (faster than copying)
        ln -s "$video_file" "$gpu_dataset_dir/videos/"
        
        file_count=$((file_count + 1))
        
        # Move to next GPU if current one has enough files
        if [ $file_count -eq $files_per_gpu ] && [ $gpu_id -lt $((NUM_GPUS - 1)) ]; then
            echo "GPU $gpu_id: assigned $file_count files"
            gpu_id=$((gpu_id + 1))
            file_count=0
        fi
    fi
done

# Show final assignment for last GPU
if [ $file_count -gt 0 ]; then
    echo "GPU $gpu_id: assigned $file_count files"
fi

echo ""
echo "Starting multi-GPU processing..."

# Launch processes on each GPU
pids=()
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    gpu_dataset_dir="$temp_base_dir/gpu_${gpu_id}_dataset"
    gpu_files_count=$(find "$gpu_dataset_dir/metas" -name "*.txt" 2>/dev/null | wc -l)
    
    if [ $gpu_files_count -gt 0 ]; then
        echo "Starting GPU $gpu_id with $gpu_files_count files..."
        
        # Run on specific GPU in background
        CUDA_VISIBLE_DEVICES=$gpu_id python -m scripts.get_t5_embeddings_from_groot_dataset_v1 \
            --dataset_path "$gpu_dataset_dir" \
            > "$temp_base_dir/gpu_${gpu_id}.log" 2>&1 &
        
        pids+=($!)
    else
        echo "No files assigned to GPU $gpu_id"
    fi
done

echo ""
echo "All processes started. Waiting for completion..."
echo "You can monitor progress in separate terminals with:"
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    if [ -f "$temp_base_dir/gpu_${gpu_id}.log" ]; then
        echo "  tail -f $temp_base_dir/gpu_${gpu_id}.log"
    fi
done
echo ""

# Wait for all processes to complete
failed_processes=0
for i in "${!pids[@]}"; do
    pid=${pids[$i]}
    if wait $pid; then
        echo "GPU $i process completed successfully (PID: $pid)"
    else
        echo "GPU $i process failed (PID: $pid)"
        failed_processes=$((failed_processes + 1))
    fi
done

# Copy results back to main dataset directory
echo ""
echo "Copying results back to main dataset..."
main_t5_dir="${DATASET_PATH}/t5_xxl"
mkdir -p "$main_t5_dir"

for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    gpu_t5_dir="$temp_base_dir/gpu_${gpu_id}_dataset/t5_xxl"
    if [ -d "$gpu_t5_dir" ]; then
        cp "$gpu_t5_dir"/*.pickle "$main_t5_dir/" 2>/dev/null || true
        copied_files=$(find "$gpu_t5_dir" -name "*.pickle" | wc -l)
        echo "Copied $copied_files pickle files from GPU $gpu_id"
    fi
done

# Show final logs
echo ""
echo "=== FINAL LOGS ==="
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    log_file="$temp_base_dir/gpu_${gpu_id}.log"
    if [ -f "$log_file" ]; then
        echo "--- GPU $gpu_id Log ---"
        tail -5 "$log_file"
        echo ""
    fi
done

# Clean up
rm -rf "$temp_base_dir"

if [ $failed_processes -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "All GPU processes completed successfully!"
    
    # Count final results
    final_pickle_count=$(find "$main_t5_dir" -name "*.pickle" | wc -l)
    echo "Total T5 embeddings created: $final_pickle_count"
    echo "============================================"
else
    echo ""
    echo "============================================"
    echo "ERROR: $failed_processes GPU processes failed!"
    echo "============================================"
    exit 1
fi