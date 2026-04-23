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
A variant of predict2_video2world.py for GR00T models that:
1. Supports prompt prefix for robot task descriptions
2. Turns off the guardrail and prompt refiner
3. Supports two GR00T variants: GR1 and DROID
"""

import argparse
import json
import os

from imaginaire.constants import (
    CosmosPredict2Gr00tModelSize,
    CosmosPredict2Video2WorldAspectRatio,
    get_cosmos_predict2_gr00t_checkpoint,
)

# Set TOKENIZERS_PARALLELISM environment variable to avoid deadlocks with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from megatron.core import parallel_state
from tqdm import tqdm

from cosmos_predict2.configs.base.config_video2world import get_cosmos_predict2_video2world_pipeline
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from examples.video2world import _DEFAULT_NEGATIVE_PROMPT, validate_input_file
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

    from methods.cache_strategy.FasterCache.config import add_fastercache_args
    from methods.cache_strategy.FasterCache.runtime import apply_fastercache
    from methods.prunning.SiTo.config import add_dreamgen_sito_args
    from methods.prunning.importance_token_merge.config import add_dreamgen_itm_args
    from methods.sparse_attention.pisa.config import add_pisa_args
    from methods.sparse_attention.svg.config import add_dreamgen_svg_args
    from methods.sparse_attention.radial.config import add_dreamgen_radial_args
    from methods.cache_strategy.config import add_dreamgen_cache_args, build_cache_runtime_config
    return (
        add_pisa_args,
        add_dreamgen_svg_args,
        add_dreamgen_radial_args,
        add_dreamgen_sito_args,
        add_dreamgen_itm_args,
        add_dreamgen_cache_args,
        add_fastercache_args,
        apply_fastercache,
        build_cache_runtime_config,
    )
# -----------------------------------


def _resolve_tokenizer_path(dit_path: str, default_tokenizer_path: str) -> str:
    checkpoint_dir = os.path.dirname(dit_path)
    colocated_tokenizer_path = os.path.join(checkpoint_dir, "tokenizer", "tokenizer.pth")
    if os.path.exists(colocated_tokenizer_path):
        log.info(f"Using tokenizer from checkpoint directory: {colocated_tokenizer_path}")
        return colocated_tokenizer_path
    return default_tokenizer_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GR00T Video-to-World Generation with Cosmos Predict2")
    parser.add_argument(
        "--model_size",
        choices=['2B', '14B'], #CosmosPredict2Gr00tModelSize.__args__
        default="14B",
        help="Size of the model to use for GR00T video-to-world generation",
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
        default="16:9",
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
    parser.add_argument(
        "--disable_guardrail",
        action="store_true",
        help="Disable guardrail checks on prompts",
    )
    parser.add_argument(
        "--gr00t_variant", type=str, required=True, help="GR00T variant to use", choices=["gr1", "droid"]
    )
    parser.add_argument(
        "--prompt_prefix", type=str, default="The robot arm is performing a task. ", help="Prefix to add to all prompts"
    )

    # ---- Sparse Backend Integration ---
    add_pisa_args, add_svg_args, add_radial_args, add_sito_args, add_itm_args, add_cache_args, add_fastercache_args, apply_fastercache, build_cache_runtime_config = _import_sparse_backends()
    add_pisa_args(parser)
    add_svg_args(parser)
    add_radial_args(parser)
    add_sito_args(parser)
    add_itm_args(parser)
    add_cache_args(parser)
    add_fastercache_args(parser)
    # -----------------------------------

    return parser.parse_args()


def setup_pipeline(args: argparse.Namespace):
    resolution = "480"
    fps = 16
    if getattr(args, "use_fastercache", False) and args.num_gpus > 1:
        raise ValueError("[FasterCache] DreamGen FasterCache v1 is single-GPU only; set --num_gpus 1.")
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
        itm=getattr(args, "use_itm", False),
        itm_compress_ratio=getattr(args, "itm_compress_ratio", None),
        itm_prune_from_step=getattr(args, "itm_prune_from_step", 1),
        itm_merge_from_step=getattr(args, "itm_merge_from_step", 2),
        itm_merge_attn=getattr(args, "itm_merge_attn", True),
        itm_merge_crossattn=getattr(args, "itm_merge_crossattn", False),
        itm_merge_mlp=getattr(args, "itm_merge_mlp", False),
        itm_start_layer_idx=getattr(args, "itm_start_layer_idx", None),
        itm_keep_last_n_dense=getattr(args, "itm_keep_last_n_dense", 2),
        sito=getattr(args, "use_sito", False),
        sito_prune_ratio=getattr(args, "sito_prune_ratio", None),
        sito_start_layer_idx=getattr(args, "sito_start_layer_idx", None),
        sito_keep_last_n_dense=getattr(args, "sito_keep_last_n_dense", 2),
        sito_patch_h=getattr(args, "sito_patch_h", 2),
        sito_patch_w=getattr(args, "sito_patch_w", 2),
        sito_noise_alpha=getattr(args, "sito_noise_alpha", 0.1),
        sito_sim_beta=getattr(args, "sito_sim_beta", 1.0),
        svg=getattr(args, "use_svg", False),
        svg_variant=getattr(args, "svg_variant", "svg1"),
        svg_start_layer_idx=getattr(args, "svg_start_layer_idx", 4),
        svg_dense_step_frac=getattr(args, "svg_dense_step_frac", 0.1),
        svg_sparsity=getattr(args, "svg_sparsity", 0.25),
        svg_num_sampled_rows=getattr(args, "svg_num_sampled_rows", 64),
        svg_sample_mse_max_row=getattr(args, "svg_sample_mse_max_row", 3000),
        svg_num_q_centroids=getattr(args, "svg_num_q_centroids", 50),
        svg_num_k_centroids=getattr(args, "svg_num_k_centroids", 200),
        svg_top_p_kmeans=getattr(args, "svg_top_p_kmeans", 0.9),
        svg_min_kc_ratio=getattr(args, "svg_min_kc_ratio", 0.0),
        svg_kmeans_iter_init=getattr(args, "svg_kmeans_iter_init", 5),
        svg_kmeans_iter_step=getattr(args, "svg_kmeans_iter_step", 2),
        fastercache=getattr(args, "use_fastercache", False),
        fastercache_start_step=getattr(args, "fastercache_start_step", 0),
        fastercache_model_interval=getattr(args, "fastercache_model_interval", 5),
        fastercache_block_interval=getattr(args, "fastercache_block_interval", 3),
        fastercache_debug=getattr(args, "fastercache_debug", False),
    )
    config.prompt_refiner_config.enabled = False
    if args.dit_path:
        dit_path = args.dit_path
    else:
        dit_path = get_cosmos_predict2_gr00t_checkpoint(
            gr00t_variant=args.gr00t_variant,
            model_size=args.model_size,
            resolution=resolution,
            fps=fps,
            aspect_ratio=args.aspect_ratio,
        )
    log.info(f"Loading model from: {dit_path}")
    config.tokenizer.vae_pth = _resolve_tokenizer_path(dit_path, config.tokenizer.vae_pth)

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

    # Load models
    log.info(f"Initializing Video2WorldPipeline with GR00T variant: {args.gr00t_variant}")
    _, _, _, _, _, _, _, _, build_cache_runtime_config = _import_sparse_backends()
    cache_config = build_cache_runtime_config(args)
    pipe = Video2WorldPipeline.from_config(
        config=config,
        dit_path=dit_path,
        device="cuda",
        torch_dtype=torch.bfloat16,
        load_ema_to_reg=args.load_ema,
        load_prompt_refiner=False,  # Disable prompt refiner for GR00T
        cache_config=cache_config,
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
    guidance: float,
    seed: int,
    prompt_prefix: str,
) -> bool:
    # Validate input file
    if not validate_input_file(input_path, num_conditional_frames):
        log.warning(f"Input file validation failed: {input_path}")
        return False

    # Add prefix to prompt
    full_prompt = prompt_prefix + prompt
    log.info(f"Running Video2WorldPipeline\ninput: {input_path}\nprompt: {full_prompt}")

    video, prompt_used = pipe(
        prompt=full_prompt,
        negative_prompt=negative_prompt,
        aspect_ratio=aspect_ratio,
        input_path=input_path,
        num_conditional_frames=num_conditional_frames,
        guidance=guidance,
        seed=seed,
        return_prompt=True,
    )

    if video is not None:
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
            prompts_to_save["refined_prompt"] = prompt_used
        save_text_prompts(prompts_to_save, output_prompt_path)
        log.success(f"Successfully saved prompt file to: {output_prompt_path}")
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
                guidance=args.guidance,
                seed=args.seed,
                prompt_prefix=args.prompt_prefix,
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
            guidance=args.guidance,
            seed=args.seed,
            prompt_prefix=args.prompt_prefix,
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
