#!/bin/bash

# ==============================================================================
#      Automated Video Generation - Final Multi-GPU Batch Script
# ==============================================================================
# This script builds a manifest of all jobs and then calls the inference
# script a single time to process the entire batch, avoiding repeated model loads.
# ==============================================================================
export PYTHONPATH="/work/HHRI-AI/POC/public/pretraining_weights/NVIDIA/Cosmos-Predict/cosmos-predict2:$PYTHONPATH"

# --- GPU Configuration ---
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_GPUS=8

# --- Path Configuration ---
DATASET_DIR="/work/HHRI-AI/POC/public/pretraining_data/Robotic-Data/Nvidia-Cosmos-Reasoning-SFT/Cosmos-Reason1-SFT-Dataset/test-sampling-data"
PROMPT_DIR="/work/HHRI-AI/POC/public/pretraining_data/Robotic-Data/Nvidia-Cosmos-Reasoning-SFT/Cosmos-Reason1-SFT-Dataset/test-sampling-data/Qwen2.5-VL-32B-Instruct-output"
COSMOS_DIR="/work/HHRI-AI/POC/public/pretraining_weights/NVIDIA/Cosmos-Predict/cosmos-predict2"
OUTPUT_VIDEO_DIR="$COSMOS_DIR/output"
MANIFEST_FILE="/tmp/video_generation_manifest.json"

# --- Pre-run Checks ---
if [ ! -d "$DATASET_DIR" ]; then
  echo "❌ Error: Dataset directory not found at '$DATASET_DIR'"
  exit 1
fi
if ! command -v jq &> /dev/null; then
    echo "❌ Error: 'jq' is not installed. Please install it (e.g., 'sudo apt-get install jq')."
    exit 1
fi

# --- 1. Prepare Environment & Create Input Manifest ---
echo "🚀 Preparing to generate input manifest..."
mkdir -p "$OUTPUT_VIDEO_DIR"
echo "[]" > "$MANIFEST_FILE"
echo "✅ Output video directory is ready at '$OUTPUT_VIDEO_DIR'."

# --- 2. Loop Through Data to Build the Manifest ---
shopt -s nullglob
task_count=0

for screenshot_dir in "$DATASET_DIR"/*_screenshots; do
  video_id=$(basename "$screenshot_dir" _screenshots)
  
  image_file=$(find "$screenshot_dir" -name "*frame_0.jpg" | head -n 1)
  prompt_file="$PROMPT_DIR/${video_id}_result.txt"

  if [ -z "$image_file" ] || [ ! -f "$prompt_file" ]; then
    echo "⚠️ Warning: Missing image or prompt for $video_id. Skipping."
    continue
  fi

  prompt_text=$(sed '1,/Generated Response:/d' "$prompt_file" | tr -d '\n\r')
  if [ -z "$prompt_text" ]; then
    echo "⚠️ Warning: Could not extract prompt for $video_id. Skipping."
    continue
  fi

  output_path="output/2B_${video_id}.mp4"

  # --- Add job to manifest with CORRECTED JSON keys ---
  jq \
    --arg image "$image_file" \
    --arg prompt "$prompt_text" \
    --arg save_path "$output_path" \
    '. += [{input_video: $image, prompt: $prompt, output_video: $save_path}]' \
    "$MANIFEST_FILE" > "${MANIFEST_FILE}.tmp" && mv "${MANIFEST_FILE}.tmp" "$MANIFEST_FILE"

  echo "  📝 Added task for Video ID: $video_id"
  task_count=$((task_count + 1))
done

echo "--------------------------------------------------------"
echo "✅ Manifest created with $task_count tasks."

# --- 3. Execute the Python Script ONCE with the Manifest ---
if [ "$task_count" -gt 0 ]; then
  echo "🚀 Starting single batch-run... Model will load once."
  cd "$COSMOS_DIR"
  
  # <<< FINAL COMMAND: Using the correct --batch_input_json flag >>>
  #    --offload_guardrail \
  #    --disable_guardrail \
  #    --offload_prompt_refiner
  ##  --dit_path /work/HHRI-AI/POC/public/pretraining_weights/NVIDIA/Cosmos-Predict/Cosmos-Predict2-14B-Sample-GR00T-Dreams-GR1 \
  # --dit_path /work/HHRI-AI/POC/public/pretraining_weights/NVIDIA/Cosmos-Predict/Cosmos-Predict2-14B-Sample-GR00T-Dreams-GR1/model-480p-16fps.pt \
  
  torchrun --nproc_per_node=$NUM_GPUS examples/video2world.py \
    --num_gpus $NUM_GPUS \
    --model_size 2B \
     --batch_input_json "$MANIFEST_FILE" \
    --num_conditional_frames 1 \
    --disable_guardrail \

else
  echo "🤷 No valid tasks found to process. Exiting."
fi

rm "$MANIFEST_FILE"
echo "--------------------------------------------------------"
echo "🎉 All videos have been processed! Check the output folder."