"""DreamGen singleview inference with correct fps conditioning matching training.

The default Video2WorldPipeline._get_data_batch_input() uses random fps in [16,32].
This wrapper monkey-patches it to pass the actual training fps per embodiment:
  - bimanual:   30 fps
  - single_arm: 15 fps
  - humanoid:   20 fps

This keeps state_t=24 (matching training architecture) while fixing fps conditioning.

Usage:
    torchrun --nproc_per_node=2 -m scripts.infer_dreamgen_singleview_native_fps \
        --embodiment single_arm --split makovian --max-episodes 3 \
        --save-root tmp/2

    # All embodiments:
    python -m scripts.infer_dreamgen_singleview_native_fps \
        --embodiment all --max-episodes 3 --save-root tmp/2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMBODIMENT_FPS = {"bimanual": 30, "single_arm": 15, "humanoid": 20}
FRAMES_PER_CHUNK = 93
FRAMES_PER_ADDITIONAL = 92
CAMERAS = {
    "bimanual": "observation.images.cam_high",
    "single_arm": "observation.images.wrist_left",
    "humanoid": "observation.images.ego_view_freq20",
}


def compute_num_chunks(source_frames: int, source_fps: float) -> int:
    """Compute num_chunks to cover full episode at 16fps output (state_t=24 → 93 frames)."""
    target_frames = source_frames * (16.0 / source_fps)
    if target_frames <= FRAMES_PER_CHUNK:
        return 1
    return math.ceil((target_frames - FRAMES_PER_CHUNK) / FRAMES_PER_ADDITIONAL) + 1


def get_frame_count(video_path: Path) -> int | None:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "stream=nb_frames",
         "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        try:
            return int(result.stdout.strip().split("\n")[0])
        except ValueError:
            pass
    return None


def extract_first_frame(video_path: Path, out_png: Path) -> bool:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path),
         "-vf", "scale=768:432", "-vframes", "1", "-q:v", "2", str(out_png)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0 and out_png.exists()


def find_episodes(dataset_dir: Path, embodiment: str, camera: str):
    """Find episodes for a given embodiment/split directory."""
    if embodiment == "bimanual":
        results = []
        for task_dir in sorted(dataset_dir.iterdir()):
            if not task_dir.is_dir() or task_dir.name in ("meta", "captions"):
                continue
            task_name = task_dir.name
            task_videos = task_dir / "videos"
            if not task_videos.exists():
                continue
            for mp4 in sorted(task_videos.glob(f"chunk-*/{camera}/*.mp4")):
                ep_match = re.search(r"(\d+)$", mp4.stem)
                if ep_match:
                    results.append((int(ep_match.group(1)), task_name, mp4))
        return results
    else:
        videos_root = dataset_dir / "videos"
        results = []
        for mp4 in sorted(videos_root.glob(f"chunk-*/{camera}/*.mp4")):
            ep_match = re.search(r"(\d+)$", mp4.stem)
            if ep_match:
                results.append((int(ep_match.group(1)), None, mp4))
        return results


def load_captions(dataset_dir: Path, embodiment: str) -> dict[int, str]:
    """Load per-episode captions/task descriptions."""
    captions = {}
    if embodiment == "humanoid":
        cap_dir = dataset_dir / "captions"
        if cap_dir.exists():
            for txt in cap_dir.glob("*.txt"):
                m = re.search(r"(\d+)$", txt.stem)
                if m:
                    captions[int(m.group(1))] = txt.read_text().strip()
    elif embodiment == "single_arm":
        ep_file = dataset_dir / "meta" / "episodes.jsonl"
        if ep_file.exists():
            with ep_file.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    ep_idx = rec.get("episode_index", -1)
                    tasks = rec.get("tasks", [])
                    if ep_idx >= 0 and tasks and tasks[0]:
                        captions[ep_idx] = tasks[0]
    return captions


def get_caption(ep_id: int, task_name: str | None, embodiment: str, captions: dict) -> str:
    if embodiment == "bimanual":
        name = task_name.replace("_", " ")
        name = name[0].upper() + name[1:]
        return f"A bimanual robot performing the task: {name}."
    elif embodiment == "humanoid":
        return captions.get(ep_id, "A GR-1 humanoid robot performing a manipulation task.")
    else:
        return captions.get(ep_id, "A robot arm performing a manipulation task.")


def monkey_patch_pipeline_fps(pipe, fps_value: int):
    """Monkey-patch the pipeline to use a fixed fps conditioning value instead of random."""
    original_get_data_batch = pipe._get_data_batch_input

    def patched_get_data_batch_input(video, prompt, negative_prompt="", num_latent_conditional_frames=1):
        batch = original_get_data_batch(video, prompt, negative_prompt, num_latent_conditional_frames)
        batch["fps"] = torch.tensor([fps_value], dtype=torch.long).cuda()
        return batch

    pipe._get_data_batch_input = patched_get_data_batch_input
    print(f"  [PATCH] Pipeline fps conditioning fixed to {fps_value}")


def _generate_video(
    pipe, input_path: str, prompt: str, output_path: str,
    num_chunks: int, guidance: float, seed: int, output_fps: int = 16,
    use_prompt_refinement: bool = False,
) -> bool:
    """Generate video using LVG (multi-chunk autoregressive) pipeline.

    When use_prompt_refinement=True, the prompt is refined at each chunk boundary
    using the last generated frame as visual context, so the text description
    evolves to reflect the current visual state.
    """
    from imaginaire.utils.io import save_image_or_video
    from imaginaire.utils import log

    input_p = Path(input_path)
    if not input_p.exists():
        log.warning(f"Input not found: {input_path}")
        return False

    log.info(f"Generating: {input_path} → {output_path} ({num_chunks} chunks, prompt_refine={use_prompt_refinement})")

    prompt_refiner = getattr(pipe, "prompt_refiner", None)
    if use_prompt_refinement and prompt_refiner is None:
        log.warning("prompt_refinement requested but no prompt_refiner loaded — falling back to static prompt")
        use_prompt_refinement = False

    all_chunks = []
    current_input_path = input_path
    current_prompt = prompt
    num_conditional_frames = 1

    with tempfile.TemporaryDirectory() as tmpdir:
        for chunk_id in range(num_chunks):
            # Refine prompt using current visual state (last frame of prev chunk)
            if use_prompt_refinement:
                log.info(f"  Chunk {chunk_id + 1}/{num_chunks}: refining prompt with {current_input_path}...")
                # Temporarily force enabled=True to bypass the skip-if-disabled check
                orig_enabled = prompt_refiner.enabled
                prompt_refiner.enabled = True
                current_prompt = prompt_refiner.refine_prompt(current_input_path, prompt)
                prompt_refiner.enabled = orig_enabled
                log.info(f"  Refined prompt: {current_prompt[:120]}...")

            log.info(f"  Chunk {chunk_id + 1}/{num_chunks} generating...")

            video, prompt_used = pipe(
                prompt=current_prompt,
                negative_prompt="",
                aspect_ratio="16:9",
                input_path=current_input_path,
                num_conditional_frames=num_conditional_frames,
                guidance=guidance,
                seed=seed + chunk_id,
                return_prompt=True,
            )

            if video is None:
                log.warning(f"  Generation returned None at chunk {chunk_id}")
                return False

            if chunk_id == 0:
                chunk = video
            else:
                chunk = video[:, :, num_conditional_frames:, :, :]
            all_chunks.append(chunk.cpu())

            # Save last frame as conditioning for next chunk (and prompt refinement input)
            last_frames = video[:, :, -num_conditional_frames:, :, :]
            if num_conditional_frames == 1:
                last_frames_path = os.path.join(tmpdir, f"chunk{chunk_id}_overlap.png")
            else:
                last_frames_path = os.path.join(tmpdir, f"chunk{chunk_id}_overlap.mp4")
            save_image_or_video(last_frames, last_frames_path, fps=16)
            current_input_path = last_frames_path

    if all_chunks:
        video = torch.cat(all_chunks, dim=2)
        output_dir = str(Path(output_path).parent)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        save_image_or_video(video, output_path, fps=output_fps)
        log.success(f"Saved: {output_path}")
        return True

    return False


def run_inference_for_embodiment(
    pipe, embodiment: str, split: str, dataset_dir: Path, save_root: Path,
    max_episodes: int | None, guidance: float, seed: int,
    use_prompt_refinement: bool = False,
):
    """Run inference for one embodiment/split with correct fps conditioning."""
    source_fps = EMBODIMENT_FPS[embodiment]
    camera = CAMERAS[embodiment]

    output_dir = save_root / embodiment / split / f"fps_{source_fps}_native"
    input_dir = save_root / embodiment / split / "input"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    episodes = find_episodes(dataset_dir, embodiment, camera)
    if max_episodes:
        episodes = episodes[:max_episodes]

    if not episodes:
        print(f"  [SKIP] No episodes found for {embodiment}/{split}")
        return

    captions = load_captions(dataset_dir, embodiment)
    print(f"  Found {len(episodes)} episodes, fps_conditioning={source_fps}")

    for ep_id, task_name, video_path in episodes:
        if embodiment == "bimanual":
            ep_name = f"{task_name}__episode_{ep_id:06d}"
        else:
            ep_name = f"episode_{ep_id:06d}"

        out_mp4 = output_dir / f"{ep_name}.mp4"
        if out_mp4.exists():
            print(f"  [SKIP] {ep_name}: already exists")
            continue

        frame_png = input_dir / "first_frames" / f"{ep_name}.png"
        if not frame_png.exists():
            if not extract_first_frame(video_path, frame_png):
                print(f"  [WARN] Failed to extract frame: {video_path.name}")
                continue

        n_frames = get_frame_count(video_path)
        num_chunks = compute_num_chunks(n_frames, source_fps) if n_frames else 3

        caption = get_caption(ep_id, task_name, embodiment, captions)
        print(f"  Generating {ep_name} ({num_chunks} chunks, {n_frames} src frames)...")

        success = _generate_video(
            pipe=pipe,
            input_path=str(frame_png),
            prompt=caption,
            output_path=str(out_mp4),
            num_chunks=num_chunks,
            guidance=guidance,
            seed=seed,
            output_fps=source_fps,
            use_prompt_refinement=use_prompt_refinement,
        )
        if success:
            print(f"  [OK] {ep_name} → {out_mp4}")
        else:
            print(f"  [FAIL] {ep_name}")


def main():
    parser = argparse.ArgumentParser(
        description="DreamGen singleview inference with native fps conditioning"
    )
    parser.add_argument("--embodiment", type=str, default="all",
                        choices=["all", "bimanual", "single_arm", "humanoid"])
    parser.add_argument("--split", type=str, default="makovian",
                        choices=["makovian", "non_makovian", "both"])
    parser.add_argument("--max-episodes", type=int, default=3)
    parser.add_argument("--save-root", type=str, default="tmp/2")
    parser.add_argument("--datasets-root", type=str, default=str(PROJECT_ROOT / "datasets"))
    parser.add_argument("--dit-path", type=str, default="")
    parser.add_argument("--iter-name", type=str, default="iter_000090000")
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--use-prompt-refinement", action="store_true",
                        help="Refine prompt at each chunk boundary using last generated frame")
    args = parser.parse_args()

    # Resolve checkpoint
    # BROKEN (2026-10-05 refactor): the combined all_singleview run does not exist on disk
    # (not in checkpoints/finetuned/ nor the HF backup); pass an explicit checkpoint.
    ckpt_base = PROJECT_ROOT / "checkpoints/finetuned/dreamgen/posttraining/singleview/all_singleview"
    if args.dit_path:
        dit_path = args.dit_path
    else:
        dit_path = str(ckpt_base / "checkpoints_40_per" / "model" / f"{args.iter_name}.pt")
        if not Path(dit_path).exists():
            dit_path = str(ckpt_base / "checkpoints_40_per" / "backup_60k_70k" / "model" / f"{args.iter_name}.pt")

    assert Path(dit_path).exists(), f"Checkpoint not found: {dit_path}"
    print(f"Checkpoint: {dit_path}")

    # Setup pipeline (always use fps=16 config for state_t=24 matching training)
    sys.path.insert(0, str(PROJECT_ROOT / "models" / "dreamgen"))

    from cosmos_predict2.configs.base.config_video2world import get_cosmos_predict2_video2world_pipeline
    from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
    from imaginaire.utils import distributed, misc
    from megatron.core import parallel_state

    misc.set_random_seed(seed=args.seed, by_rank=True)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if args.num_gpus > 1:
        distributed.init()
        parallel_state.initialize_model_parallel(context_parallel_size=args.num_gpus)

    config = get_cosmos_predict2_video2world_pipeline(
        model_size="2B", resolution="480", fps=16,
    )
    config.guardrail_config.enabled = False
    config.prompt_refiner_config.enabled = False

    print("Loading pipeline (state_t=24, matching training)...")
    pipe = Video2WorldPipeline.from_config(
        config=config,
        dit_path=dit_path,
        device="cuda",
        torch_dtype=torch.bfloat16,
        load_ema_to_reg=False,
        load_prompt_refiner=args.use_prompt_refinement,
    )

    # Determine which embodiments/splits to run
    embodiments = list(EMBODIMENT_FPS.keys()) if args.embodiment == "all" else [args.embodiment]
    splits = ["makovian", "non_makovian"] if args.split == "both" else [args.split]
    save_root = Path(args.save_root)

    for embodiment in embodiments:
        fps_val = EMBODIMENT_FPS[embodiment]
        print(f"\n{'='*60}")
        print(f"  {embodiment} — fps conditioning = {fps_val} (native)")
        print(f"{'='*60}")

        # Patch pipeline to use correct fps conditioning
        monkey_patch_pipeline_fps(pipe, fps_val)

        for split in splits:
            dataset_dir = Path(args.datasets_root) / embodiment / "singleview" / split
            if not dataset_dir.exists():
                print(f"  [SKIP] {dataset_dir} not found")
                continue

            print(f"\n  --- {embodiment}/{split} ---")
            run_inference_for_embodiment(
                pipe=pipe,
                embodiment=embodiment,
                split=split,
                dataset_dir=dataset_dir,
                save_root=save_root,
                max_episodes=args.max_episodes,
                guidance=args.guidance,
                seed=args.seed,
                use_prompt_refinement=args.use_prompt_refinement,
            )

    print(f"\nDone! Results at: {save_root}/")


if __name__ == "__main__":
    main()
