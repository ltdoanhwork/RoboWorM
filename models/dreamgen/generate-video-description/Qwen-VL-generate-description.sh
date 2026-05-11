#!/bin/bash

# =============================================================================
# Qwen2.5-VL Inference Script - PERMISSION FIX
# =============================================================================

# -----------------
# CONFIGURATION
# -----------------
SCRIPT_DIR="/work/HHRI-AI/POC/public/DreamGen/VLA-Humanoid/generate-video-description/"
PYTHON_SCRIPT="$SCRIPT_DIR/inference-multi-gpus-qwen-vlm.py"
MODEL_PATH="/work/HHRI-AI/POC/public/pretraining_weights/Vision-Language-Models/Alibaba/Qwen-VLM/Qwen2.5-VL-32B-Instruct"
GPU_MEMORY=0.9

# 1. INPUT: The Read-Only Video Source
BASE_VIDEO_ROOT="/work/HHRI-AI/POC/public/pretraining_data/Robotic-Data/Video-Simulation-Data/robocasa/robocasa_100_demos_lerobot_5_chosen_tasks_v2/videos/chunk-000"

# 2. OUTPUT: CHANGE THIS to a folder YOU have permission to write to!
#    Example: "/work/HHRI-AI/POC/users/YOUR_USERNAME/experiments/Qwen_Results"
OUTPUT_ROOT="/work/HHRI-AI/POC/public/pretraining_data/Robotic-Data/Video-Simulation-Data/robocasa/robocasa_100_demos_lerobot_5_chosen_tasks_v2/results" 

# Combine them (Script will create this folder)
MAIN_OUTPUT_DIR="${OUTPUT_ROOT}/chunk-000_captions"

VIDEO_PROMPT="Can you describe what would be the best description instruction for this video related to robotic tasks?"
TARGET_FOLDERS=("left_image" "right_image" "wrist_image")

# ... (Rest of script remains exactly the same) ...

# -----------------
# 🐍 PYTHON ENVIRONMENT SETUP
# -----------------
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate your_environment_name

# =============================================================================
# 🛡️ SAFETY & CLEANUP
# =============================================================================
PIDs=()

cleanup() {
    echo ""
    echo "🛑  CAUGHT CTRL+C! Killing processes..."
    for pid in "${PIDs[@]}"; do
        kill -9 "$pid" 2>/dev/null
    done
    pkill -9 -f "inference-multi-gpus-qwen-vlm.py" 2>/dev/null
    exit 1
}
trap cleanup SIGINT SIGTERM

# =============================================================================
# PROCESSING FUNCTION
# =============================================================================
process_specific_folder() {
    local SUBFOLDER_NAME=$1
    local INPUT_PATH="${BASE_VIDEO_ROOT}/${SUBFOLDER_NAME}"
    local TARGET_OUTPUT_DIR="${MAIN_OUTPUT_DIR}/${SUBFOLDER_NAME}"
    
    PIDs=()

    echo "#################################################################"
    echo "🚀 STARTING BATCH FOR: $SUBFOLDER_NAME"
    echo "📂 Input:  $INPUT_PATH"
    echo "📂 Output: $TARGET_OUTPUT_DIR"
    echo "#################################################################"

    if [ ! -d "$INPUT_PATH" ]; then
        echo "⚠️  Folder $INPUT_PATH does not exist. Skipping."
        return
    fi

    # Check if we can write to output dir
    if ! mkdir -p "$TARGET_OUTPUT_DIR" 2>/dev/null; then
        echo "❌ CRITICAL ERROR: Permission denied creating output directory:"
        echo "   $TARGET_OUTPUT_DIR"
        echo "   Please change OUTPUT_ROOT in the script to a folder you own."
        exit 1
    fi

    # Hardware Config
    TOTAL_GPUS_TO_USE=4
    TENSOR_PARALLEL_PER_JOB=2 
    
    NUM_JOBS=$((TOTAL_GPUS_TO_USE / TENSOR_PARALLEL_PER_JOB))
    
    # 1. FIND VIDEOS
    ALL_FILES_LIST="${TARGET_OUTPUT_DIR}/all_videos_${SUBFOLDER_NAME}.txt"
    find "$INPUT_PATH" -maxdepth 1 -type f -name "*.mp4" | sort > "$ALL_FILES_LIST"
    
    # Check if find worked (if permission denied on INPUT, this will fail too)
    if [ ! -f "$ALL_FILES_LIST" ]; then
         echo "❌ Error: Could not generate video list. Check read permissions on input folder."
         return
    fi
    
    TOTAL_FILES=$(wc -l < "$ALL_FILES_LIST")
    if [ "$TOTAL_FILES" -eq 0 ]; then
        echo "❌ No .mp4 files found in $SUBFOLDER_NAME"
        return
    fi
    echo "✅ Found $TOTAL_FILES videos."

    # 2. SPLIT FILES
    split -d -a 1 -n l/$NUM_JOBS "$ALL_FILES_LIST" "${TARGET_OUTPUT_DIR}/job_batch_"

    # 3. LAUNCH JOBS
    for ((i=0; i<NUM_JOBS; i++)); do
        BATCH_FILE="${TARGET_OUTPUT_DIR}/job_batch_$i"
        if [ ! -f "$BATCH_FILE" ]; then continue; fi

        GPU_START=$((i * TENSOR_PARALLEL_PER_JOB))
        GPU_END=$((GPU_START + TENSOR_PARALLEL_PER_JOB - 1))
        GPU_SEQ=$(seq -s, $GPU_START $GPU_END)
        
        JOB_OUTPUT_DIR="${TARGET_OUTPUT_DIR}/job_${i}"
        JOB_LOG="${JOB_OUTPUT_DIR}/inference.log"
        mkdir -p "$JOB_OUTPUT_DIR"
        
        echo "--> [$SUBFOLDER_NAME] Launching Job $i on GPUs [$GPU_SEQ]"
        echo "    Logging to: $JOB_LOG"

        (
            export CUDA_DEVICE_ORDER=PCI_BUS_ID
            export CUDA_VISIBLE_DEVICES=$GPU_SEQ
            
            python3 "$PYTHON_SCRIPT" \
                --file-list "$BATCH_FILE" \
                --file-types "video" \
                --text "$VIDEO_PROMPT" \
                --model "$MODEL_PATH" \
                --gpu-count "$TENSOR_PARALLEL_PER_JOB" \
                --tensor-parallel-size "$TENSOR_PARALLEL_PER_JOB" \
                --gpu-memory-util "$GPU_MEMORY" \
                --output-dir "$JOB_OUTPUT_DIR" \
                --batch-size 1 > "$JOB_LOG" 2>&1
        ) &
        
        PIDs+=($!)
        
        # --- DEBUGGING BLOCK ---
        sleep 3
        if grep -q "Error" "$JOB_LOG"; then
             echo "⚠️  POTENTIAL ERROR DETECTED IN LOG:"
             grep "Error" "$JOB_LOG" | head -n 3
        fi
        # -----------------------
    done

    echo "⏳ Waiting for jobs to complete..."
    wait "${PIDS[@]}"
    
    # 4. VERIFY AND MERGE
    GENERATED_COUNT=$(find "${TARGET_OUTPUT_DIR}/job_"* -name "*_desc.txt" | wc -l)
    
    if [ "$GENERATED_COUNT" -eq 0 ]; then
        echo "❌ ERROR: Jobs finished but ZERO description files were found."
        echo "   Please check the logs in: $TARGET_OUTPUT_DIR"
    else 
        echo "📦 Merging $GENERATED_COUNT results..."
        find "${TARGET_OUTPUT_DIR}/job_"* -name "*_desc.txt" -exec mv {} "$TARGET_OUTPUT_DIR/" \; 2>/dev/null
        rm -f "${TARGET_OUTPUT_DIR}/job_batch_"*
        rm -f "$ALL_FILES_LIST"
        echo "✅ Success for $SUBFOLDER_NAME"
    fi
    echo ""
}

# =============================================================================
# EXECUTION
# =============================================================================
pkill -f "inference-multi-gpus-qwen-vlm.py" 2>/dev/null

for FOLDER in "${TARGET_FOLDERS[@]}"; do
    process_specific_folder "$FOLDER"
done