# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Batch inference script for DreamDojo action-conditioned video generation.

Reads a batch_input.json file with format:
[
    {
        "input_video": "/path/to/image.png",
        "prompt": "text description",
        "output_video": "/path/to/output.mp4"   (optional, overridden by --output_dir)
    },
    ...
]

Supports image-only inputs with zero actions (no robot state data required).

Usage (single GPU):
    PYTHONPATH=. python cosmos_predict2/_src/predict2/action/inference/inference_batch.py \
        --experiment=dreamdojo_14b_480_640_gr1 \
        --ckpt_path /path/to/checkpoint/ \
        --batch_input /path/to/batch_input.json \
        --output_dir /path/to/output/ \
        --resolution 480,640 --guidance 0 --chunk_size 12

Usage (multi-GPU with context parallelism):
    torchrun --nproc_per_node=2 --master_port=12341 \
        -m cosmos_predict2._src.predict2.action.inference.inference_batch \
        --experiment=dreamdojo_14b_480_640_gr1 \
        --ckpt_path /path/to/checkpoint/ \
        --batch_input /path/to/batch_input.json \
        --output_dir /path/to/output/ \
        --resolution 480,640 --guidance 0 --chunk_size 12 \
        --context_parallel_size 2
"""

import argparse
import json
import os
import time
from pathlib import Path

import mediapy
import numpy as np
import torch
from loguru import logger

from cosmos_predict2._src.predict2.action.inference.inference_pipeline import (
    _DEFAULT_NEGATIVE_PROMPT,
    ActionVideo2WorldInference,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch inference for DreamDojo")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment config name")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--batch_input", type=str, required=True, help="Path to batch_input.json or .jsonl")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for generated videos")
    parser.add_argument("--s3_cred", type=str, default="credentials/s3_checkpoint.secret")
    parser.add_argument("--resolution", type=str, default="480,640", help="Resolution H,W")
    parser.add_argument("--guidance", type=int, default=0, help="Guidance scale (0=no guidance)")
    parser.add_argument("--chunk_size", type=int, default=12, help="Number of action frames per chunk")
    parser.add_argument("--action_dim", type=int, default=384, help="Action dimension")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--save_fps", type=int, default=5, help="Output video FPS")
    parser.add_argument("--context_parallel_size", type=int, default=1, help="Number of GPUs for context parallelism")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=_DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt for classifier-free guidance",
    )
    parser.add_argument("--use_neg_prompt", action="store_true", help="Use negative prompt")
    parser.add_argument("--num_latent_conditional_frames", type=int, default=1, help="Number of conditional frames (0, 1, or 2)")
    parser.add_argument("--timing_output", type=str, default=None, help="Path to save timing JSON (default: output_dir/timing.json)")
    return parser.parse_args()


def resolve_input_path(entry: dict, idx: int) -> str:
    input_path = entry.get("input_video") or entry.get("input_path")
    if input_path is None:
        raise KeyError(
            f"Batch entry at index {idx} is missing an input path. "
            f"Expected one of ['input_video', 'input_path'], got keys: {sorted(entry.keys())}"
        )
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path does not exist for entry {idx}: {input_path}")
    return input_path


def resolve_output_path(entry: dict, output_dir: str, input_path: str) -> str:
    if entry.get("output_video"):
        stem = Path(entry["output_video"]).stem
    elif entry.get("name"):
        stem = entry["name"]
    else:
        stem = Path(input_path).stem
    return os.path.join(output_dir, f"{stem}.mp4")


def main():
    torch.enable_grad(False)
    args = parse_arguments()

    # Load batch input (supports both JSON array and JSONL formats)
    with open(args.batch_input, "r") as f:
        content = f.read().strip()
    try:
        batch_entries = json.loads(content)
        if not isinstance(batch_entries, list):
            batch_entries = [batch_entries]
    except json.JSONDecodeError:
        # Try JSONL (one JSON object per line)
        batch_entries = [json.loads(line) for line in content.splitlines() if line.strip()]
    logger.info(f"Loaded {len(batch_entries)} entries from {args.batch_input}")

    # Parse resolution
    target_h, target_w = None, None
    if args.resolution != "none":
        target_h, target_w = map(int, args.resolution.split(","))

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize model
    logger.info(f"Loading model: experiment={args.experiment}, ckpt={args.ckpt_path}")
    torch.cuda.synchronize()
    t_model_start = time.perf_counter()

    video2world = ActionVideo2WorldInference(
        experiment_name=args.experiment,
        ckpt_path=args.ckpt_path,
        s3_credential_path=args.s3_cred,
        context_parallel_size=args.context_parallel_size,
    )

    torch.cuda.synchronize()
    t_model_load = time.perf_counter() - t_model_start
    logger.info(f"Model loaded in {t_model_load:.2f}s")

    mem_bytes = torch.cuda.memory_allocated()
    logger.info(f"GPU memory after model load: {mem_bytes / (1024**3):.2f} GB")

    # Check rank for distributed
    is_rank0 = True
    if args.context_parallel_size > 1:
        from cosmos_predict2._src.imaginaire.utils import distributed
        is_rank0 = distributed.get_rank() == 0

    # Timing records
    timing_records = []

    # Process each entry
    for idx, entry in enumerate(batch_entries):
        input_path = resolve_input_path(entry, idx)
        prompt = entry.get("prompt", "")
        output_path = resolve_output_path(entry, args.output_dir, input_path)

        if os.path.exists(output_path):
            logger.info(f"[{idx+1}/{len(batch_entries)}] Skipping (exists): {output_path}")
            continue

        logger.info(f"[{idx+1}/{len(batch_entries)}] Processing: {input_path}")
        logger.info(f"  Prompt: {prompt}")

        # Load input image
        img_array = mediapy.read_image(input_path)
        if target_h is not None and target_w is not None:
            img_array = mediapy.resize_image(img_array, (target_h, target_w))

        # Create zero actions (no robot state data)
        zero_actions = np.zeros((args.chunk_size, args.action_dim), dtype=np.float32)

        # Run inference with timing
        torch.cuda.synchronize()
        t_start = time.perf_counter()

        next_frame, video_clamped = video2world.step_inference(
            img_array=img_array,
            action=zero_actions,
            guidance=args.guidance,
            seed=args.seed,
            num_latent_conditional_frames=args.num_latent_conditional_frames,
            prompt=prompt,
            negative_prompt=args.negative_prompt if args.use_neg_prompt else "",
            use_neg_prompt=args.use_neg_prompt,
        )

        torch.cuda.synchronize()
        t_elapsed = time.perf_counter() - t_start

        # Save video (only on rank 0)
        if is_rank0:
            mediapy.write_video(output_path, video_clamped, fps=args.save_fps)
            logger.info(f"  Saved: {output_path} ({video_clamped.shape[0]} frames, {t_elapsed:.2f}s)")

        timing_records.append({
            "index": idx,
            "input": input_path,
            "prompt": prompt,
            "output": output_path,
            "num_frames": int(video_clamped.shape[0]),
            "resolution": f"{video_clamped.shape[1]}x{video_clamped.shape[2]}",
            "inference_time_s": round(t_elapsed, 3),
        })

    # Save timing summary
    if is_rank0 and timing_records:
        total_time = sum(r["inference_time_s"] for r in timing_records)
        avg_time = total_time / len(timing_records)
        total_frames = sum(r["num_frames"] for r in timing_records)

        timing_summary = {
            "model": args.experiment,
            "checkpoint": args.ckpt_path,
            "context_parallel_size": args.context_parallel_size,
            "resolution": args.resolution,
            "guidance": args.guidance,
            "chunk_size": args.chunk_size,
            "model_load_time_s": round(t_model_load, 3),
            "num_samples": len(timing_records),
            "total_inference_time_s": round(total_time, 3),
            "avg_inference_time_s": round(avg_time, 3),
            "total_frames_generated": total_frames,
            "avg_fps": round(total_frames / total_time, 2) if total_time > 0 else 0,
            "per_sample": timing_records,
        }

        timing_path = args.timing_output or os.path.join(args.output_dir, "timing.json")
        with open(timing_path, "w") as f:
            json.dump(timing_summary, f, indent=2)
        logger.info(f"\nTiming Summary:")
        logger.info(f"  Model load: {t_model_load:.2f}s")
        logger.info(f"  Samples: {len(timing_records)}")
        logger.info(f"  Total inference: {total_time:.2f}s")
        logger.info(f"  Avg per sample: {avg_time:.2f}s")
        logger.info(f"  Avg FPS: {total_frames / total_time:.2f}" if total_time > 0 else "  Avg FPS: N/A")
        logger.info(f"  Timing saved to: {timing_path}")

    # Cleanup
    if args.context_parallel_size > 1:
        video2world.cleanup()


if __name__ == "__main__":
    main()
