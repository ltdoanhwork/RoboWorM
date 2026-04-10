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

import argparse
import json
import os

# Set TOKENIZERS_PARALLELISM environment variable to avoid deadlocks with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import tempfile
import time
from datetime import datetime

import torch
from megatron.core import parallel_state
from tqdm import tqdm

from cosmos_predict2.configs.base.config_video2world import (
    get_cosmos_predict2_video2world_pipeline,
)
from cosmos_predict2.pipelines.video2world import _IMAGE_EXTENSIONS, _VIDEO_EXTENSIONS, Video2WorldPipeline
from imaginaire.constants import (
    CosmosPredict2Video2WorldAspectRatio,
    CosmosPredict2Video2WorldModelSize,
    get_cosmos_predict2_video2world_checkpoint,
)
from imaginaire.utils import distributed, log, misc
from imaginaire.utils.io import save_image_or_video, save_text_prompts

# ---- Sparse Backend Integration ----
def _import_sparse_backends():
    import sys
    # Search for project root (where 'methods' directory resides)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # The script is in models/dreamgen/examples/
    # Project root should be 3 levels up
    project_root = os.path.normpath(os.path.join(current_dir, "../../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from methods.sparse_attention.pisa.config import add_pisa_args
    from methods.sparse_attention.radial.config import add_dreamgen_radial_args
    return add_pisa_args, add_dreamgen_radial_args
# -----------------------------------

_DEFAULT_NEGATIVE_PROMPT = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."


def validate_input_file(input_path: str, num_conditional_frames: int) -> bool:
    if not os.path.exists(input_path):
        log.warning(f"Input file does not exist, skipping: {input_path}")
        return False

    ext = os.path.splitext(input_path)[1].lower()

    if num_conditional_frames == 1:
        # Single frame conditioning: accept both images and videos
        if ext not in _IMAGE_EXTENSIONS and ext not in _VIDEO_EXTENSIONS:
            log.warning(
                f"Skipping file with unsupported extension for single frame conditioning: {input_path} "
                f"(expected: {_IMAGE_EXTENSIONS + _VIDEO_EXTENSIONS})"
            )
            return False
    elif num_conditional_frames == 5:
        # Multi-frame conditioning: only accept videos
        if ext not in _VIDEO_EXTENSIONS:
            log.warning(
                f"Skipping file for multi-frame conditioning (requires video): {input_path} "
                f"(expected: {_VIDEO_EXTENSIONS}, got: {ext})"
            )
            return False
    else:
        log.error(f"Invalid num_conditional_frames: {num_conditional_frames} (must be 1 or 5)")
        return False

    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video-to-World Generation with Cosmos Predict2")
    parser.add_argument(
        "--model_size",
        choices=CosmosPredict2Video2WorldModelSize.__args__,
        default="2B",
        help="Size of the model to use for video-to-world generation",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default="",
        help="Custom path to the DiT model checkpoint for post-trained models.",
    )
    parser.add_argument(
        "--load_ema",
        action="store_true",
        help="Use EMA weights for generation.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Text prompt for video generation",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="assets/video2world/input0.jpg",
        help="Path to input image or video for conditioning (include file extension)",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=_DEFAULT_NEGATIVE_PROMPT,
        help="Negative text prompt for video-to-world generation",
    )
    parser.add_argument(
        "--aspect_ratio",
        choices=CosmosPredict2Video2WorldAspectRatio.__args__,
        default="1:1", #,"16:9",
        type=str,
        help="Aspect ratio of the generated output (width:height)",
    )
    parser.add_argument(
        "--num_conditional_frames",
        type=int,
        default=1,
        choices=[1, 5],
        help="Number of frames to condition on (1 for single frame, 5 for multi-frame conditioning)",
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=6,
        help="Number of chunks to be generated for long video generation",
    )
    parser.add_argument(
        "--batch_input_json",
        type=str,
        default=None,
        help="Path to JSON file containing batch inputs. Each entry should have 'input_video', 'prompt', and 'output_video' fields.",
    )
    parser.add_argument("--guidance", type=float, default=7, help="Guidance value")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument(
        "--save_path",
        type=str,
        default="output/generated_video.mp4",
        help="Path to save the generated video (include file extension)",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for context parallel inference (should be a divisor of the total frames)",
    )
    parser.add_argument("--disable_guardrail", action="store_true", help="Disable guardrail checks on prompts")
    parser.add_argument(
        "--disable_prompt_refiner", action="store_true", help="Disable prompt refiner that enhances short prompts"
    )

    # ---- Sparse Backend Integration ----
    add_pisa_args, add_radial_args = _import_sparse_backends()
    add_pisa_args(parser)
    add_radial_args(parser)
    # -----------------------------------

    return parser.parse_args()


def setup_pipeline(args: argparse.Namespace):
    resolution = "720"
    fps = 16
    config = get_cosmos_predict2_video2world_pipeline(
        model_size=args.model_size,
        resolution=resolution,
        fps=fps,
        pisa=getattr(args, "use_pisa", False),
        pisa_density=getattr(args, "pisa_density", 0.5),
        pisa_block_size=getattr(args, "pisa_block_size", 64),
        pisa_start_layer_idx=getattr(args, "pisa_start_layer_idx", 0),
        pisa_use_bias=getattr(args, "pisa_use_bias", False),
        radial=getattr(args, "use_radial", False),
        radial_decay_factor=getattr(args, "radial_decay_factor", 1.0),
        radial_block_size=getattr(args, "radial_block_size", 64),
        radial_start_layer_idx=getattr(args, "radial_start_layer_idx", 4),
        radial_model_type=getattr(args, "radial_model_type", "hunyuan"),
    )
    if args.dit_path:
        dit_path = args.dit_path
    else:
        dit_path = get_cosmos_predict2_video2world_checkpoint(
            model_size=args.model_size, resolution=resolution, fps=fps, aspect_ratio=args.aspect_ratio
        )

    misc.set_random_seed(seed=args.seed, by_rank=True)
    # Initialize cuDNN.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # Floating-point precision settings.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize distributed environment for multi-GPU inference
    if args.num_gpus > 1:
        log.info(f"Initializing distributed environment with {args.num_gpus} GPUs for context parallelism")
        distributed.init()
        parallel_state.initialize_model_parallel(context_parallel_size=args.num_gpus)
        log.info(f"Context parallel group initialized with {args.num_gpus} GPUs")

    # Disable guardrail if requested
    if args.disable_guardrail:
        log.warning("Guardrail checks are disabled")
        config.guardrail_config.enabled = False

    # Disable prompt refiner if requested
    if args.disable_prompt_refiner:
        log.warning("Prompt refiner is disabled")
        config.prompt_refiner_config.enabled = False

    # Load models
    log.info(f"Initializing Video2WorldPipeline with model size: {args.model_size}")
    pipe = Video2WorldPipeline.from_config(
        config=config,
        dit_path=dit_path,
        device="cuda",
        torch_dtype=torch.bfloat16,
        load_ema_to_reg=args.load_ema,
        load_prompt_refiner=True,
    )

    return pipe


def process_single_generation(
    pipe: Video2WorldPipeline,
    input_path: str,
    prompt: str,
    output_path: str,
    negative_prompt: str,
    aspect_ratio: str,
    num_conditional_frames: int,
    num_chunks: int,
    guidance: float,
    seed: int,
) -> bool:
    # Validate input file
    if not validate_input_file(input_path, num_conditional_frames):
        log.warning(f"Input file validation failed: {input_path}")
        return False

    log.info(f"Running Video2WorldPipeline\ninput: {input_path}\nprompt: {prompt}")

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    all_chunks = []
    current_input_path = input_path

    with tempfile.TemporaryDirectory() as tmpdir:
        for chunk_id in tqdm(range(num_chunks)):
            log.info(f"Generating chunk {chunk_id + 1}/{num_chunks}...")

            video, prompt_used = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                aspect_ratio=aspect_ratio,
                input_path=current_input_path,
                num_conditional_frames=num_conditional_frames,
                guidance=guidance,
                seed=seed + chunk_id,  # change random seed to avoid repeat
                return_prompt=True,
            )
            # for the first chunk, we save the whole clip
            if chunk_id == 0:
                chunk = video
                prompts_used = [prompt_used]
            else:
                chunk = video[:, :, num_conditional_frames:, :, :]
                prompts_used.append(prompt_used)
            all_chunks.append(chunk.cpu())

            # Prepare for next chunk: use last `num_conditional_frames` frames as new condition
            last_frames = video[:, :, -num_conditional_frames:, :, :]  # (1, C, num_conditional_frames, H, W)
            # if num_conditional_frames is 1, save as image temp file
            # otherwise, save as video temp file
            if num_conditional_frames == 1:
                last_frames_path = os.path.join(tmpdir, f"chunk{chunk_id}_overlap.png")
            else:
                last_frames_path = os.path.join(tmpdir, f"chunk{chunk_id}_overlap.mp4")
            save_image_or_video(last_frames, last_frames_path, fps=16)
            current_input_path = last_frames_path

    if len(all_chunks) > 0:
        # Concatenate all chunks along time
        video = torch.cat(all_chunks, dim=2)
        # save the generated video
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        log.info(f"Saving generated video to: {output_path}")
        save_image_or_video(video, output_path, fps=16)
        log.success(f"Successfully saved video to: {output_path}")
        # save the prompts used to generate the video
        output_prompt_path = os.path.splitext(output_path)[0] + ".txt"
        prompts_to_save = {"prompt": prompt, "negative_prompt": negative_prompt}
        if (
            pipe.prompt_refiner is not None
            and getattr(pipe.config, "prompt_refiner_config", None) is not None
            and getattr(pipe.config.prompt_refiner_config, "enabled", False)
        ):
            prompts_to_save["refined_prompt"] = []
            for chunk_id, prompt_used in enumerate(prompts_used):  # noqa: B007
                prompts_to_save["refined_prompt"].append(prompt_used)
        save_text_prompts(prompts_to_save, output_prompt_path)
        log.success(f"Successfully saved prompt file to: {output_prompt_path}")

        torch.cuda.synchronize()
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        log.info(f"LVG Total inference time: {total_time:.4f} seconds for {num_chunks} chunks.")
        
        is_main_process = (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0)
        if is_main_process:
            metrics_dir = "./inference_times"
            try:
                os.makedirs(metrics_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                record = {
                    "timestamp": timestamp,
                    "total_inference_time_seconds": total_time,
                    "average_time_per_chunk_seconds": total_time / num_chunks,
                    "num_chunks": num_chunks,
                    "num_conditional_frames": num_conditional_frames,
                    "guidance": guidance,
                    "seed": seed,
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "model": "video2world_lvg"
                }
                output_file = os.path.join(metrics_dir, f"lvg_inference_{timestamp}.json")
                with open(output_file, "w") as f:
                    json.dump(record, f, indent=4)
                log.success(f"LVG Inference timing information saved to {output_file}")
            except Exception as e:
                log.error(f"Failed to save LVG inference timing information: {e}")

        return True
    return False


def generate_video(args: argparse.Namespace, pipe: Video2WorldPipeline) -> None:
    # Video-to-World
    if args.batch_input_json is not None:
        # Process batch inputs from JSON file
        log.info(f"Loading batch inputs from JSON file: {args.batch_input_json}")
        with open(args.batch_input_json) as f:
            batch_inputs = json.load(f)

        for idx, item in enumerate(tqdm(batch_inputs)):
            input_video = item.get("input_video", "")
            prompt = item.get("prompt", "")
            output_video = item.get("output_video", f"output_{idx}.mp4")

            if not input_video or not prompt:
                log.warning(f"Skipping item {idx}: Missing input_video or prompt")
                continue

            process_single_generation(
                pipe=pipe,
                input_path=input_video,
                prompt=prompt,
                output_path=output_video,
                negative_prompt=args.negative_prompt,
                aspect_ratio=args.aspect_ratio,
                num_conditional_frames=args.num_conditional_frames,
                num_chunks=args.num_chunks,
                guidance=args.guidance,
                seed=args.seed,
            )
    else:
        process_single_generation(
            pipe=pipe,
            input_path=args.input_path,
            prompt=args.prompt,
            output_path=args.save_path,
            negative_prompt=args.negative_prompt,
            aspect_ratio=args.aspect_ratio,
            num_conditional_frames=args.num_conditional_frames,
            num_chunks=args.num_chunks,
            guidance=args.guidance,
            seed=args.seed,
        )

    return


def cleanup_distributed():
    """Clean up the distributed environment if initialized."""
    if parallel_state.is_initialized():
        parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    try:
        pipe = setup_pipeline(args)
        generate_video(args, pipe)
    finally:
        # Make sure to clean up the distributed environment
        cleanup_distributed()

'''
torchrun --nproc_per_node=4 examples/video2world_lvg.py \
  --num_gpus 4 \
  --model_size 2B \
  --dit_path /mnt/data/sftp/data/vla_intern/workspace/binh/dreamgen/dreamgen/checkpoints/posttraining/video2world/2b_groot_gr1_480/checkpoints_40_per/model/iter_000000280.pt \
  --input_path data4infer/test_screenshots/frame_0.jpg \
  --prompt "robot pick all objects from cabinet to counter" \
  --num_conditional_frames 1 \
  --num_chunks 3 \
  --aspect_ratio "1:1" \
  --disable_guardrail 
'''
