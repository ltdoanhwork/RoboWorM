#!/bin/bash

# ==============================================================================
#      Automated Video Generation - Libero Data Batch Script
# ==============================================================================
# This script processes video episodes from a flatter data structure.
# ==============================================================================
export PYTHONPATH="/mnt/data/sftp/data/vla_intern/workspace/binh/dreamgen/dreamgen:$PYTHONPATH"

# --- GPU Configuration ---
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_GPUS=4

# screen shots/ 
#   episode_000/frame0_screenshots.jpg frame0_screenshots.jpg 
  
# text dir/
#   episode_000_results.txt

# --- Path Configuration (UPDATED) ---
DATASET_DIR="/mnt/data/sftp/data/vla_intern/workspace/binh/dreamgen/data/robocasa_100demos_5tasks_pred/robocasa-100demos-5tasks-for-dg-inference/left_view/Screenshots"
PROMPT_DIR="/mnt/data/sftp/data/vla_intern/workspace/binh/dreamgen/data/robocasa_100demos_5tasks_pred/robocasa-100demos-5tasks-for-dg-inference/left_view/Qwen2.5-VL-32B-Instruct"
COSMOS_CODE_DIR="/mnt/data/sftp/data/vla_intern/workspace/binh/dreamgen/dreamgen" 
COSMOS_OUTPUT_DIR="/mnt/data/sftp/data/vla_intern/workspace/binh/dreamgen/data/robocasa_100demos_5tasks_pred/robocasa-100demos-5tasks-for-dg-inference/left_view/generated_videos" 
MANIFEST_FILE="/mnt/data/sftp/data/vla_intern/workspace/binh/dreamgen/data/robocasa_100demos_5tasks_pred/robocasa-100demos-5tasks-for-dg-inference/left_view/video_generation_manifest.json"

# --- Naming Convention Configuration ---
# 1. We assume folders end in "_screenshots"
CAMERA_VIEW_SUFFIX="_screenshots" 

# 2. We assume prompt files are exactly [VideoName].txt (e.g., episode_000000.txt)
PROMPT_FILE_SUFFIX="_result.txt" 

# --- Pre-run Checks ---
if [ ! -d "$DATASET_DIR" ]; then
  echo "❌ Error: Dataset directory not found at '$DATASET_DIR'"
  exit 1
fi
if [ ! -d "$PROMPT_DIR" ]; then
  echo "❌ Error: Prompt directory not found at '$PROMPT_DIR'"
  exit 1
fi
if ! command -v jq &> /dev/null; then
    echo "❌ Error: 'jq' is not installed. Please install it (e.g., 'sudo apt-get install jq')."
    exit 1
fi

# --- 1. Prepare Environment & Create Input Manifest ---
echo "🚀 Preparing environment and generating input manifest..."
mkdir -p "$COSMOS_OUTPUT_DIR"
echo "[]" > "$MANIFEST_FILE"
echo "✅ Output video directory is ready at '$COSMOS_OUTPUT_DIR'."

# --- 2. Loop Through Episodes ---
shopt -s nullglob
task_count=0

# Loop through all screenshot directories (e.g., episode_000000_screenshots)
for episode_screenshot_dir in "$DATASET_DIR"/episode*"$CAMERA_VIEW_SUFFIX"; do
  
  # 1. Extract the base episode name (Removes "_screenshots")
  # Example: "episode_000000_screenshots" -> "episode_000000"
  episode_name=$(basename "$episode_screenshot_dir" | sed "s/$CAMERA_VIEW_SUFFIX//")
  
  echo "📂 Processing $episode_name..."
  
  # 2. Construct Prompt Filename
  # Since you said the prompt file IS the video name, we just add .txt
  # Example: "episode_000000" -> "episode_000000.txt"
  prompt_file="${PROMPT_DIR}/${episode_name}${PROMPT_FILE_SUFFIX}"
  
  screenshot_dir="$episode_screenshot_dir"
  
  # Find the first frame (frame_0)
  image_file=$(find "$screenshot_dir" -name "*frame_0.jpg" -o -name "*frame_001*.jpg" 2>/dev/null | head -n 1)
  
  # Check if both prompt and image exist
  if [ ! -f "$prompt_file" ]; then
    echo "  ⚠️ Missing prompt file: $prompt_file. Skipping."
    continue
  fi
  
  if [ -z "$image_file" ]; then
    echo "  ⚠️ Missing frame_0 image in $screenshot_dir. Skipping."
    continue
  fi
  
  # Extract prompt text
  # This command reads the file. If "Generated Response:" exists, it removes lines before it.
  # If the file is just clean text, it cleans up whitespace.
  prompt_text=$(sed '1,/Generated Response:/d' "$prompt_file" | tr -d '\n\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  
  # FALLBACK: If the sed command emptied the string (because "Generated Response:" wasn't found but sed behaved strictly),
  # read the whole file. Use this if your text files are just pure prompts.
  if [ -z "$prompt_text" ]; then
     prompt_text=$(<"$prompt_file")
     prompt_text=$(echo "$prompt_text" | tr -d '\n\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  fi

  if [ -z "$prompt_text" ]; then
    echo "  ⚠️ Empty prompt content for $episode_name. Skipping."
    continue
  fi
  
  # Create unique output filename
  output_path="${COSMOS_OUTPUT_DIR}/${episode_name}_predicted_3chunks_new.mp4"
  # --- NEW: CHECK FOR EXISTING FILE ---
  if [ -f "$output_path" ]; then
    echo "  ⏭️  Video already exists: $episode_name. Skipping..."
    continue
  fi
  # Add job to manifest
  jq \
    --arg image "$image_file" \
    --arg prompt "$prompt_text" \
    --arg save_path "$output_path" \
    '. += [{input_video: $image, prompt: $prompt, output_video: $save_path}]' \
    "$MANIFEST_FILE" > "${MANIFEST_FILE}.tmp" && mv "${MANIFEST_FILE}.tmp" "$MANIFEST_FILE"
  
  echo "  ✅ Added task for: $episode_name"
  task_count=$((task_count + 1))
done

echo "--------------------------------------------------------"
echo "✅ Manifest created with $task_count tasks."
# Optional: Preview tasks
# jq -r '.[] | .output_video' "$MANIFEST_FILE" | head -n 5

# --- 3. Execute the Python Script ONCE with the Manifest ---
if [ "$task_count" -gt 0 ]; then
  echo ""
  echo "🚀 Starting batch processing... Model will load once."
  cd "$COSMOS_CODE_DIR"
  
  torchrun --nproc_per_node=$NUM_GPUS examples/video2world_lvg.py \
    --num_gpus $NUM_GPUS \
    --model_size 2B \
    --dit_path /mnt/data/sftp/data/vla_intern/workspace/binh/dreamgen/dreamgen/checkpoints/posttraining/video2world/robocasa_100demos_5tasks_v1/checkpoints_40_per/model/iter_000045000.pt \
    --batch_input_json "$MANIFEST_FILE" \
    --num_conditional_frames 1 \
    --num_chunks 3 \
    --aspect_ratio "1:1" \
    --disable_guardrail
else
  echo "🤷 No valid tasks found to process. Exiting."
fi

# --- Cleanup ---
rm "$MANIFEST_FILE"
echo "--------------------------------------------------------"
echo "🎉 Processing complete! Videos saved to: $COSMOS_OUTPUT_DIR"