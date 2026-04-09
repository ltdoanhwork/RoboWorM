#!/usr/bin/env python3
"""
Qwen2.5-VL Multi-GPU Inference Script (Updated for Parallel Batching)
"""

import os
# CRITICAL FIX: Force vLLM and Multiprocessing to use 'spawn'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import multiprocessing
# Force spawn for standard multiprocessing as well
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass
import argparse
import sys
import os
import torch
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

def setup_model(model_path, gpu_count=None, gpu_memory_utilization=0.9, tensor_parallel_size=None):
    """Initialize the model and processor with multi-GPU support"""
    
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
    else:
        available_gpus = 0
        
    if gpu_count is None:
        gpu_count = available_gpus
    
    # Set tensor parallel size
    if tensor_parallel_size is None:
        tensor_parallel_size = gpu_count if gpu_count > 0 else 1
    
    print(f"🚀 Initializing model on {gpu_count} GPUs (TP={tensor_parallel_size})...")
    
    # vLLM configuration
    llm_kwargs = {
        "model": model_path,
        # Limit frames to prevent OOM
        "limit_mm_per_prompt": {"image": 10, "video": 10}, 
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": tensor_parallel_size,
        "max_model_len": 8192,
        "enforce_eager": True,
        # CRITICAL: Use 'mp' instead of 'ray' to allow multiple parallel jobs on one node
        "distributed_executor_backend": "mp", 
    }
    
    try:
        llm = LLM(**llm_kwargs)
        print("✅ Model loaded successfully!")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        return llm, processor
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

def create_video_messages(video_path, text_prompt):
    """Create messages for video input with memory safety"""
    return [
        {"role": "system", "content": "You are a helpful robotic assistant."},
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": text_prompt},
                {
                    "type": "video", 
                    "video": video_path,
                    # CRITICAL: Sample 1 FPS to prevent OOM on long videos
                    "fps": 1.0, 
                }
            ]
        },
    ]

def create_image_messages(image_path, text_prompt):
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": text_prompt},
        ]},
    ]

def process_batch(file_paths, file_type, text_prompt, processor, llm, sampling_params, output_dir):
    """Process files in batches"""
    total_files = len(file_paths)
    print(f"🔄 Processing {total_files} files...")

    # We process 1 by 1 inside the python loop to ensure we handle errors gracefully
    # The 'Batching' happens at the Bash level (splitting files)
    for i, file_path in enumerate(file_paths, 1):
        # 1. Output Filename Logic
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        # Append _desc so we don't overwrite original files
        output_filename = f"{base_name}_result.txt"
        output_file = os.path.join(output_dir, output_filename)

        # 2. Resume Logic
        if os.path.exists(output_file):
            print(f"⏭️  Skipping {i}/{total_files}: {base_name} (Exists)")
            continue

        print(f"🎥 Processing {i}/{total_files}: {base_name}")

        try:
            # 3. Prepare Messages
            if file_type == 'video':
                messages = create_video_messages(file_path, text_prompt)
            else:
                messages = create_image_messages(file_path, text_prompt)

            # 4. Prepare vLLM Inputs
            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            mm_data = {}
            if image_inputs is not None: mm_data["image"] = image_inputs
            if video_inputs is not None: mm_data["video"] = video_inputs

            llm_inputs = {
                "prompt": text_input,
                "multi_modal_data": mm_data,
            }

            # 5. Generate
            outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
            generated_text = outputs[0].outputs[0].text

            # 6. Save
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(generated_text)
            print(f"✅ Saved: {output_filename}")

        except Exception as e:
            print(f"❌ Error processing {base_name}: {e}")
            # Don't exit, just try the next file
            continue

def main():
    parser = argparse.ArgumentParser()
    # Updated arguments to match Bash Script
    parser.add_argument('--file-list', type=str, help='Path to text file containing list of files to process')
    parser.add_argument('--folder', type=str, help='Path to folder (legacy support)')
    
    parser.add_argument('--file-types', type=str, default='video', choices=['image', 'video'])
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--gpu-count', type=int, default=1)
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--gpu-memory-util', type=float, default=0.9)
    parser.add_argument('--batch-size', type=int, default=1) # Not strictly used here, but passed by bash script
    
    args = parser.parse_args()
    
    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load File List
    files = []
    if args.file_list:
        if not os.path.exists(args.file_list):
            print(f"❌ File list not found: {args.file_list}")
            sys.exit(1)
        with open(args.file_list, 'r') as f:
            files = [line.strip() for line in f if line.strip()]
    elif args.folder:
        # Fallback for folder scanning
        exts = ('.mp4', '.avi', '.mov') if args.file_types == 'video' else ('.jpg', '.png')
        for root, _, filenames in os.walk(args.folder):
            for filename in filenames:
                if filename.lower().endswith(exts):
                    files.append(os.path.join(root, filename))
    
    if not files:
        print("❌ No files found to process!")
        sys.exit(1)

    print(f"📊 Found {len(files)} files.")

    # 2. Setup Model
    llm, processor = setup_model(
        args.model, 
        args.gpu_count, 
        args.gpu_memory_util, 
        args.tensor_parallel_size
    )

    # 3. Sampling Params
    sampling_params = SamplingParams(
        temperature=0.2,
        top_p=0.9,
        max_tokens=2048,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    # 4. Run Batch
    process_batch(
        files, 
        args.file_types, 
        args.text, 
        processor, 
        llm, 
        sampling_params, 
        args.output_dir
    )

if __name__ == "__main__":
    main()