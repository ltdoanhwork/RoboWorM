#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Batch inference script for base/non-action Video2World generation.

Reads a batch_input.json or batch_input.jsonl file with entries like:
[
    {
        "input_video": "/path/to/image_or_video.png",
        "prompt": "text description",
        "output_video": "/path/to/output.mp4",
        "inference_type": "image2world"
    },
    {
        "prompt": "text-only generation",
        "name": "sample_0001",
        "inference_type": "text2world"
    }
]

If `inference_type` is omitted, it is inferred from `input_video` / `input_path`:
- missing input path -> text2world
- image path -> image2world
- video path -> video2world
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

from cosmos_predict2._src.predict2.inference.video2world import (
    _DEFAULT_NEGATIVE_PROMPT,
    Video2WorldInference,
)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
_VIDEO_EXTENSIONS = {".mp4"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch inference for base/non-action Video2World")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment config name")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to checkpoint directory or file")
    parser.add_argument("--batch_input", type=str, required=True, help="Path to batch_input.json or .jsonl")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for generated videos")
    parser.add_argument("--config_file", type=str, default="cosmos_predict2/_src/predict2/configs/video2world/config.py")
    parser.add_argument("--s3_cred", type=str, default="credentials/s3_checkpoint.secret")
    parser.add_argument("--resolution", type=str, default="none", help="Resolution H,W or none for model default")
    parser.add_argument("--guidance", type=int, default=7, help="Guidance scale")
    parser.add_argument("--num_output_frames", type=int, default=77, help="Number of video frames to generate")
    parser.add_argument("--num_steps", type=int, default=35, help="Number of sampling steps")
    parser.add_argument(
        "--num_latent_conditional_frames",
        type=int,
        default=None,
        help="Override number of conditional frames. Defaults to 0/1/2 from inferred inference type.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--save_fps", type=int, default=16, help="Output video FPS")
    parser.add_argument("--context_parallel_size", type=int, default=1, help="Number of GPUs for context parallelism")
    parser.add_argument("--enable_autoregressive", action="store_true", help="Use autoregressive sliding window mode")
    parser.add_argument("--chunk_size", type=int, default=77, help="Frames per chunk in autoregressive mode")
    parser.add_argument("--chunk_overlap", type=int, default=1, help="Overlap frames between chunks")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=_DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt for classifier-free guidance",
    )
    parser.add_argument("--use_neg_prompt", action="store_true", help="Use negative prompt")
    parser.add_argument(
        "--timing_output",
        type=str,
        default=None,
        help="Path to save timing JSON (default: output_dir/timing.json)",
    )
    parser.add_argument("--offload_diffusion_model", action="store_true", help="Offload diffusion model to CPU")
    parser.add_argument("--offload_tokenizer", action="store_true", help="Offload tokenizer to CPU")
    parser.add_argument("--offload_text_encoder", action="store_true", help="Offload text encoder to CPU")
    return parser.parse_args()


def load_batch_entries(path: str) -> list[dict]:
    with open(path, "r") as f:
        content = f.read().strip()
    try:
        entries = json.loads(content)
        if isinstance(entries, dict):
            entries = [entries]
    except json.JSONDecodeError:
        entries = [json.loads(line) for line in content.splitlines() if line.strip()]
    if not isinstance(entries, list):
        raise ValueError(f"Expected a list of batch entries in {path}")
    return entries


def infer_inference_type(input_path: str | None, explicit_type: str | None) -> str:
    if explicit_type is not None:
        return explicit_type
    if not input_path:
        return "text2world"
    suffix = Path(input_path).suffix.lower()
    if suffix in _IMAGE_EXTENSIONS:
        return "image2world"
    if suffix in _VIDEO_EXTENSIONS:
        return "video2world"
    raise ValueError(f"Unsupported input path extension '{suffix}' for {input_path}")


def infer_num_conditional_frames(inference_type: str, override: int | None) -> int:
    if override is not None:
        return override
    if inference_type == "text2world":
        return 0
    if inference_type == "image2world":
        return 1
    if inference_type == "video2world":
        return 2
    raise ValueError(f"Unsupported inference_type: {inference_type}")


def resolve_output_path(entry: dict, output_dir: str, input_path: str | None, index: int) -> str:
    output_video = entry.get("output_video")
    if output_video:
        base_name = Path(output_video).stem
    elif input_path:
        base_name = Path(input_path).stem
    else:
        base_name = entry.get("name") or f"sample_{index:04d}"
    return os.path.join(output_dir, f"{base_name}.mp4")


def to_video_frames(video: torch.Tensor) -> np.ndarray:
    video = ((1.0 + video[0]) / 2).clamp(0.0, 1.0)
    return (video.permute(1, 2, 3, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)


def main():
    torch.enable_grad(False)
    args = parse_arguments()
    batch_entries = load_batch_entries(args.batch_input)
    logger.info(f"Loaded {len(batch_entries)} entries from {args.batch_input}")

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Loading model: experiment={args.experiment}, ckpt={args.ckpt_path}")
    torch.cuda.synchronize()
    t_model_start = time.perf_counter()

    video2world = Video2WorldInference(
        experiment_name=args.experiment,
        ckpt_path=args.ckpt_path,
        s3_credential_path=args.s3_cred,
        context_parallel_size=args.context_parallel_size,
        config_file=args.config_file,
        offload_diffusion_model=args.offload_diffusion_model,
        offload_tokenizer=args.offload_tokenizer,
        offload_text_encoder=args.offload_text_encoder,
    )

    torch.cuda.synchronize()
    t_model_load = time.perf_counter() - t_model_start
    logger.info(f"Model loaded in {t_model_load:.2f}s")

    is_rank0 = True
    if args.context_parallel_size > 1:
        from cosmos_predict2._src.imaginaire.utils import distributed

        is_rank0 = distributed.get_rank() == 0

    timing_records = []

    for idx, entry in enumerate(batch_entries):
        input_path = entry.get("input_video") or entry.get("input_path")
        prompt = entry.get("prompt", "")
        inference_type = infer_inference_type(input_path, entry.get("inference_type"))
        num_conditional_frames = infer_num_conditional_frames(
            inference_type, entry.get("num_latent_conditional_frames", args.num_latent_conditional_frames)
        )
        output_path = resolve_output_path(entry, args.output_dir, input_path, idx)

        if os.path.exists(output_path):
            logger.info(f"[{idx + 1}/{len(batch_entries)}] Skipping (exists): {output_path}")
            continue

        logger.info(f"[{idx + 1}/{len(batch_entries)}] Processing: {input_path or inference_type}")
        logger.info(f"  Prompt: {prompt}")

        guidance = int(entry.get("guidance", args.guidance))
        seed = int(entry.get("seed", args.seed))
        num_output_frames = int(entry.get("num_output_frames", args.num_output_frames))
        num_steps = int(entry.get("num_steps", args.num_steps))
        resolution = entry.get("resolution", args.resolution)
        save_fps = int(entry.get("save_fps", args.save_fps))
        negative_prompt = entry.get("negative_prompt", args.negative_prompt)
        use_neg_prompt = bool(entry.get("use_neg_prompt", args.use_neg_prompt))
        enable_autoregressive = bool(entry.get("enable_autoregressive", args.enable_autoregressive))
        chunk_size = int(entry.get("chunk_size", args.chunk_size))
        chunk_overlap = int(entry.get("chunk_overlap", args.chunk_overlap))

        torch.cuda.synchronize()
        t_start = time.perf_counter()

        if enable_autoregressive:
            video = video2world.generate_autoregressive_from_batch(
                prompt=prompt,
                input_path=input_path,
                num_output_frames=num_output_frames,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                guidance=guidance,
                num_latent_conditional_frames=num_conditional_frames,
                resolution=resolution,
                seed=seed,
                negative_prompt=negative_prompt if use_neg_prompt else "",
                num_steps=num_steps,
            )
        else:
            video = video2world.generate_vid2world(
                prompt=prompt,
                input_path=input_path,
                guidance=guidance,
                num_video_frames=num_output_frames,
                num_latent_conditional_frames=num_conditional_frames,
                resolution=resolution,
                seed=seed,
                negative_prompt=negative_prompt if use_neg_prompt else "",
                num_steps=num_steps,
            )

        torch.cuda.synchronize()
        t_elapsed = time.perf_counter() - t_start

        if is_rank0:
            video_frames = to_video_frames(video)
            mediapy.write_video(output_path, video_frames, fps=save_fps)
            logger.info(f"  Saved: {output_path} ({video_frames.shape[0]} frames, {t_elapsed:.2f}s)")
        else:
            video_frames = None

        timing_records.append(
            {
                "index": idx,
                "input": input_path,
                "prompt": prompt,
                "inference_type": inference_type,
                "output": output_path,
                "num_frames": int(video.shape[2]),
                "resolution": resolution,
                "inference_time_s": round(t_elapsed, 3),
            }
        )

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
            "num_output_frames": args.num_output_frames,
            "num_steps": args.num_steps,
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
        logger.info(f"Timing saved to: {timing_path}")

    if args.context_parallel_size > 1:
        video2world.cleanup()


if __name__ == "__main__":
    main()
