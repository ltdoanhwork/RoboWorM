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

from imaginaire.auxiliary.text_encoder import CosmosTextEncoder
from imaginaire.constants import (
    CosmosPredict2Video2WorldAspectRatio,
    CosmosPredict2Video2WorldFPS,
    CosmosPredict2Video2WorldModelSize,
    CosmosPredict2Video2WorldResolution,
    get_cosmos_predict2_video2world_checkpoint,
    print_environment_info,
)
from imaginaire.lazy_config.lazy import LazyConfig

# Set TOKENIZERS_PARALLELISM environment variable to avoid deadlocks with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time

import torch
from megatron.core import parallel_state

from cosmos_predict2.configs.base.config_video2world import (
    get_cosmos_predict2_video2world_pipeline,
)
from cosmos_predict2.pipelines.video2world import _IMAGE_EXTENSIONS, _VIDEO_EXTENSIONS, Video2WorldPipeline
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
    return (
        add_pisa_args,
        add_dreamgen_svg_args,
        add_dreamgen_radial_args,
        add_dreamgen_sito_args,
        add_dreamgen_itm_args,
        add_fastercache_args,
        apply_fastercache,
    )
# -----------------------------------


def _import_cache_backends():
    """Import the full DreamGen cache-arg registrar + runtime-config builder.

    Kept separate from `_import_sparse_backends` so its positional return
    signature (unpacked in several call sites) stays untouched. Provides
    WorldCache / DiCache / FasterCache / ScalingCache support for the non-gr00t
    Video2World flow (used by the bimanual multiview rollout).
    """
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(current_dir, "../../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from methods.cache_strategy.config import (
        add_dreamgen_cache_args,
        build_cache_runtime_config,
    )
    return add_dreamgen_cache_args, build_cache_runtime_config
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
        default="14B",
        help="Size of the model to use for video-to-world generation",
    )
    parser.add_argument(
        "--resolution",
        choices=CosmosPredict2Video2WorldResolution.__args__,
        default="720",
        type=str,
        help="Resolution of the model to use for video-to-world generation",
    )
    parser.add_argument(
        "--fps",
        choices=CosmosPredict2Video2WorldFPS.__args__,
        default=16,
        type=int,
        help="FPS of the model to use for video-to-world generation",
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
    parser.add_argument("--disable_guardrail", action="store_true", help="Disable guardrail checks on prompts")
    parser.add_argument("--offload_guardrail", action="store_true", help="Offload guardrail to CPU to save GPU memory")
    parser.add_argument(
        "--disable_prompt_refiner", action="store_true", help="Disable prompt refiner that enhances short prompts"
    )
    parser.add_argument(
        "--offload_prompt_refiner", action="store_true", help="Offload prompt refiner to CPU to save GPU memory"
    )
    parser.add_argument(
        "--offload_text_encoder", action="store_true", help="Offload text encoder to CPU to save GPU memory"
    )
    parser.add_argument(
        "--downcast_text_encoder",
        action="store_true",
        help="Cast text encoder from checkpoint precision to pipeline precision",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the generation in benchmark mode. It means that generation will be rerun a few times and the average generation time will be shown.",
    )
    parser.add_argument("--use_cuda_graphs", action="store_true", help="Use CUDA Graphs for the text2image inference.")
    parser.add_argument(
        "--natten",
        action="store_true",
        help="Run Video2World + NATTEN (sparse attention variant).",
    )

    # ---- Sparse Backend Integration ----
    add_pisa_args, add_svg_args, add_radial_args, add_sito_args, add_itm_args, add_fastercache_args, _ = _import_sparse_backends()
    add_pisa_args(parser)
    add_svg_args(parser)
    add_radial_args(parser)
    add_sito_args(parser)
    add_itm_args(parser)
    add_fastercache_args(parser)
    # Full cache backends (WorldCache / DiCache / FasterCache-v2 / ScalingCache)
    # routed through build_cache_runtime_config -> pipeline cache_config. This is
    # what enables WorldCache/DiCache on the non-gr00t (bimanual multiview) flow.
    add_dreamgen_cache_args, _ = _import_cache_backends()
    add_dreamgen_cache_args(parser)
    # -----------------------------------

    return parser.parse_args()


def setup_pipeline(args: argparse.Namespace, text_encoder: CosmosTextEncoder | None = None):
    print_environment_info(args)
    if getattr(args, "use_fastercache", False):
        if args.num_gpus > 1:
            raise ValueError("[FasterCache] DreamGen FasterCache v1 is single-GPU only; set --num_gpus 1.")
        if args.use_cuda_graphs:
            raise ValueError("[FasterCache] DreamGen FasterCache v1 is incompatible with --use_cuda_graphs.")
    # WorldCache / DiCache / FasterCache-v2 / ScalingCache / BSC are single-GPU only
    # (context_parallel_size must be 1). Guard here so a bad NUM_GPUS fails fast.
    # The selector list lives in methods.cache_strategy.config so a newly added backend
    # cannot be omitted here (BSC would have been).
    from methods.cache_strategy.config import enabled_cache_backends

    _enabled_caches = enabled_cache_backends(args)
    if _enabled_caches and getattr(args, "num_gpus", 1) > 1:
        raise ValueError(
            f"[Cache] DreamGen cache backends ({', '.join(_enabled_caches)}) are "
            "single-GPU only; set --num_gpus 1."
        )

    config = get_cosmos_predict2_video2world_pipeline(
        model_size=args.model_size,
        resolution=args.resolution,
        fps=args.fps,
        natten=getattr(args, "natten", False),
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
    )
    if hasattr(args, "dit_path") and args.dit_path:
        dit_path = args.dit_path
    else:
        dit_path = get_cosmos_predict2_video2world_checkpoint(
            model_size=args.model_size,
            resolution=args.resolution,
            fps=args.fps,
            aspect_ratio=args.aspect_ratio,
            natten=getattr(args, "natten", False),
        )
    log.info(f"Using dit_path: {dit_path}")

    misc.set_random_seed(seed=args.seed, by_rank=True)
    # Initialize cuDNN.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # Floating-point precision settings.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize distributed environment for multi-GPU inference
    if hasattr(args, "num_gpus") and args.num_gpus > 1:
        log.info(f"Initializing distributed environment with {args.num_gpus} GPUs for context parallelism")

        # Check if distributed environment is already initialized
        if not parallel_state.is_initialized():
            distributed.init()
            parallel_state.initialize_model_parallel(context_parallel_size=args.num_gpus)
            log.info(f"Context parallel group initialized with {args.num_gpus} GPUs")
        else:
            log.info("Distributed environment already initialized, skipping initialization")
            # Check if we need to reinitialize with different context parallel size
            current_cp_size = parallel_state.get_context_parallel_world_size()
            if current_cp_size != args.num_gpus:
                log.warning(f"Context parallel size mismatch: current={current_cp_size}, requested={args.num_gpus}")
                log.warning("Using existing context parallel configuration")
            else:
                log.info(f"Using existing context parallel group with {current_cp_size} GPUs")

    # Disable guardrail if requested
    if args.disable_guardrail:
        log.warning("Guardrail checks are disabled")
        config.guardrail_config.enabled = False
    config.guardrail_config.offload_model_to_cpu = args.offload_guardrail

    # Disable prompt refiner if requested
    if args.disable_prompt_refiner:
        log.warning("Prompt refiner is disabled")
        config.prompt_refiner_config.enabled = False
    config.prompt_refiner_config.offload_model_to_cpu = args.offload_prompt_refiner

    # Save config
    output_path = os.path.splitext(args.save_path)[0]
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    LazyConfig.save_yaml(config, f"{output_path}.yaml")

    # Load models
    log.info(f"Initializing Video2WorldPipeline with model size: {args.model_size}")
    _, build_cache_runtime_config = _import_cache_backends()
    cache_config = build_cache_runtime_config(args)
    if cache_config is not None:
        log.info(f"[Cache] Enabled cache backend: {type(cache_config).__name__}")
    pipe = Video2WorldPipeline.from_config(
        config=config,
        dit_path=dit_path,
        use_text_encoder=text_encoder is None,
        offload_text_encoder=args.offload_text_encoder,
        downcast_text_encoder=args.downcast_text_encoder,
        device="cuda",
        torch_dtype=torch.bfloat16,
        load_ema_to_reg=args.load_ema,
        load_prompt_refiner=True,
        cache_config=cache_config,
    )

    # Set the provided text encoder if one was passed
    if text_encoder is not None:
        pipe.text_encoder = text_encoder

    fastercache_config = getattr(config.net, "fastercache_config", None)
    if fastercache_config is not None:
        *_, apply_fastercache = _import_sparse_backends()
        apply_fastercache(pipe.dit, fastercache_config, cfg_mode="sequential")

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
    benchmark: bool = False,
    use_cuda_graphs: bool = False,
) -> bool:
    # Validate input file
    if not validate_input_file(input_path, num_conditional_frames):
        log.warning(f"Input file validation failed: {input_path}")
        return False

    log.info(f"Running Video2WorldPipeline\ninput: {input_path}\nprompt: {prompt}")

    num_repeats = 4 if benchmark else 1
    time_sum = 0
    for i in range(num_repeats):
        if benchmark and i > 0:
            torch.cuda.synchronize()
            start_time = time.time()
        video, prompt_used = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            aspect_ratio=aspect_ratio,
            input_path=input_path,
            num_conditional_frames=num_conditional_frames,
            guidance=guidance,
            seed=seed,
            use_cuda_graphs=use_cuda_graphs,
            return_prompt=True,
        )
        if benchmark and i > 0:
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            time_sum += elapsed
            log.info(f"[iter {i} / {num_repeats - 1}] Generation time: {elapsed:.1f} seconds.")
    if benchmark:
        time_avg = time_sum / (num_repeats - 1)
        log.critical(f"Average generation time for Video2WorldPipeline is {time_avg:.1f} seconds.")

    if video is not None:
        # save the generated video
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        log.info(f"Saving the generated video to: {output_path}")
        if pipe.config.state_t == 16:
            fps = 10
        else:
            fps = 16
        save_image_or_video(video, output_path, fps=fps)
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
    if args.benchmark:
        log.warning(
            "Running in benchmark mode. Each generation will be rerun a couple of times and the average generation time will be shown."
        )
    # Video-to-World
    if args.batch_input_json is not None:
        # Process batch inputs from JSON file
        log.info(f"Loading batch inputs from JSON file: {args.batch_input_json}")
        with open(args.batch_input_json) as f:
            batch_inputs = json.load(f)

        for idx, item in enumerate(batch_inputs):
            log.info(f"Processing batch item {idx + 1}/{len(batch_inputs)}")
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
                benchmark=args.benchmark,
                use_cuda_graphs=args.use_cuda_graphs,
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
            benchmark=args.benchmark,
            use_cuda_graphs=args.use_cuda_graphs,
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
