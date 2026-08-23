"""
Single-segment full-length autoregressive rollout for bimanual singleview DreamDojo.

For each episode:
  - Start from the first GT frame.
  - Rollout autoregressively (chunk_size actions per step) for the ENTIRE episode
    length — no GT reset between segments. Pure autoregressive.
  - GT video and actions come from the dataset pipeline (properly normalized/resized).

Output: full_gt.mp4, full_pred.mp4, full_merged.mp4, metrics.json.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import mediapy
import numpy as np
import piq
import torch
import torchvision

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models" / "DreamDojo"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", type=str, required=True)
    parser.add_argument("--experiment", type=str, default="dreamdojo_2b_480_640_aloha")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Comma-separated list of task directories")
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--save-fps", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--guidance", type=float, default=0)
    parser.add_argument("--save-video-only", action="store_true", default=True,
                        help="Only save the generated prediction video (no GT/merged/metrics). Default on.")
    parser.add_argument("--save-full", dest="save_video_only", action="store_false",
                        help="Also save full_gt.mp4, full_merged.mp4 and metrics.json.")

    # WorldCache
    parser.add_argument("--worldcache-enabled", action="store_true")
    parser.add_argument("--worldcache-num-steps", type=int, default=35)
    parser.add_argument("--worldcache-rel-l1-thresh", type=float, default=0.03)
    parser.add_argument("--worldcache-ret-ratio", type=float, default=0.4)
    parser.add_argument("--worldcache-probe-depth", type=int, default=4)
    parser.add_argument("--worldcache-motion-sensitivity", type=float, default=5.0)

    # FasterCache
    parser.add_argument("--fastercache-enabled", action="store_true")
    parser.add_argument("--fastercache-start-step", type=int, default=0)
    parser.add_argument("--fastercache-model-interval", type=int, default=5)
    parser.add_argument("--fastercache-block-interval", type=int, default=3)

    # DiCache
    parser.add_argument("--dicache-enabled", action="store_true")
    parser.add_argument("--dicache-num-steps", type=int, default=35)
    parser.add_argument("--dicache-rel-l1-thresh", type=float, default=0.08)
    parser.add_argument("--dicache-ret-ratio", type=float, default=0.2)
    parser.add_argument("--dicache-probe-depth", type=int, default=2)

    # SVG (Sparse-VideoGen 1 / 2)
    parser.add_argument("--use-svg", action="store_true")
    parser.add_argument("--svg-variant", type=str, default="svg1", choices=["svg1", "svg2"])
    parser.add_argument("--svg-start-layer-idx", type=int, default=None)
    parser.add_argument("--svg-dense-step-frac", type=float, default=None)
    parser.add_argument("--svg-sparsity", type=float, default=None)
    parser.add_argument("--svg-num-sampled-rows", type=int, default=None)
    parser.add_argument("--svg-sample-mse-max-row", type=int, default=None)
    parser.add_argument(
        "--svg-profile-recompute-every",
        type=int,
        default=None,
        help="SVG1: recompute the spatial/temporal head MSE profile every N attention "
        "calls (1 = every call, the upstream behaviour). Caching this is what makes "
        "SVG1 competitive (see dreamdojo-humanoid-singleview-sparse-attention rule).",
    )
    parser.add_argument("--svg-num-q-centroids", type=int, default=None)
    parser.add_argument("--svg-num-k-centroids", type=int, default=None)
    parser.add_argument("--svg-top-p-kmeans", type=float, default=None)
    parser.add_argument("--svg-min-kc-ratio", type=float, default=None)
    parser.add_argument("--svg-kmeans-iter-init", type=int, default=None)
    parser.add_argument("--svg-kmeans-iter-step", type=int, default=None)

    # PISA (piecewise sparse attention)
    parser.add_argument("--use-pisa", action="store_true")
    parser.add_argument("--pisa-density", type=float, default=0.2)
    parser.add_argument("--pisa-block-size", type=int, default=64)
    parser.add_argument("--pisa-start-layer-idx", type=int, default=2)
    parser.add_argument("--pisa-use-bias", action="store_true")

    # Radial Attention
    parser.add_argument("--use-radial", action="store_true")
    parser.add_argument("--radial-decay-factor", type=float, default=1.0)
    parser.add_argument("--radial-block-size", type=int, default=64, choices=[64, 128])
    parser.add_argument("--radial-start-layer-idx", type=int, default=4)
    parser.add_argument("--radial-model-type", type=str, default="hunyuan", choices=["hunyuan", "wan"])

    return parser.parse_args()


def build_experiment_opts(args):
    """Build hydra experiment overrides for the sparse-attention backends.

    Mirrors infer_humanoid_singleview_full_episode.py exactly — the SVG/PISA/Radial
    runtimes are shape-driven (not embodiment-specific), so this logic is identical.
    """
    opts = []
    if getattr(args, "use_svg", False):
        opts += [
            "model.config.net.self_attention_backend=svg",
            "model.config.net.cross_attention_backend=minimal_a2a",
            f"model.config.net.svg_variant={args.svg_variant}",
        ]
        for flag, key in (
            ("svg_start_layer_idx", "svg_start_layer_idx"),
            ("svg_dense_step_frac", "svg_dense_step_frac"),
            ("svg_sparsity", "svg_sparsity"),
            ("svg_num_sampled_rows", "svg_num_sampled_rows"),
            ("svg_sample_mse_max_row", "svg_sample_mse_max_row"),
            ("svg_profile_recompute_every", "svg_profile_recompute_every"),
            ("svg_num_q_centroids", "svg_num_q_centroids"),
            ("svg_num_k_centroids", "svg_num_k_centroids"),
            ("svg_top_p_kmeans", "svg_top_p_kmeans"),
            ("svg_min_kc_ratio", "svg_min_kc_ratio"),
            ("svg_kmeans_iter_init", "svg_kmeans_iter_init"),
            ("svg_kmeans_iter_step", "svg_kmeans_iter_step"),
        ):
            value = getattr(args, flag, None)
            if value is not None:
                opts.append(f"model.config.net.{key}={value}")
    if getattr(args, "use_pisa", False):
        opts += [
            "model.config.net.self_attention_backend=pisa",
            "model.config.net.cross_attention_backend=minimal_a2a",
            f"model.config.net.pisa_density={args.pisa_density}",
            f"model.config.net.pisa_block_size={args.pisa_block_size}",
            f"model.config.net.pisa_start_layer_idx={args.pisa_start_layer_idx}",
            f"model.config.net.pisa_use_bias={str(args.pisa_use_bias).lower()}",
        ]
    if getattr(args, "use_radial", False):
        opts += [
            "model.config.net.self_attention_backend=radial",
            "model.config.net.cross_attention_backend=minimal_a2a",
            f"model.config.net.radial_decay_factor={args.radial_decay_factor}",
            f"model.config.net.radial_block_size={args.radial_block_size}",
            f"model.config.net.radial_start_layer_idx={args.radial_start_layer_idx}",
            f"model.config.net.radial_model_type={args.radial_model_type}",
        ]

    enabled = sum(
        bool(getattr(args, flag, False)) for flag in ("use_svg", "use_pisa", "use_radial")
    )
    if enabled > 1:
        raise ValueError("SVG / PISA / Radial are mutually exclusive; enable at most one.")
    return opts


def build_cache_config_from_args(args):
    """Build cache config from CLI args (mirrors infer_humanoid_singleview_full_episode.py)."""
    from methods.cache_strategy.common import WorldCacheConfig, DiCacheConfig, FasterCacheConfig

    if getattr(args, "worldcache_enabled", False):
        return WorldCacheConfig(
            num_steps=args.worldcache_num_steps,
            rel_l1_thresh=args.worldcache_rel_l1_thresh,
            ret_ratio=args.worldcache_ret_ratio,
            probe_depth=args.worldcache_probe_depth,
            motion_sensitivity=args.worldcache_motion_sensitivity,
        )
    if getattr(args, "fastercache_enabled", False):
        return FasterCacheConfig(
            start_step=args.fastercache_start_step,
            model_interval=args.fastercache_model_interval,
            block_interval=args.fastercache_block_interval,
        )
    if getattr(args, "dicache_enabled", False):
        return DiCacheConfig(
            num_steps=args.dicache_num_steps,
            rel_l1_thresh=args.dicache_rel_l1_thresh,
            ret_ratio=args.dicache_ret_ratio,
            probe_depth=args.dicache_probe_depth,
        )
    return None


def build_model(args):
    from cosmos_predict2.action_conditioned_config import ActionConditionedSetupArguments
    from cosmos_predict2.config import MODEL_CHECKPOINTS
    from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference

    setup_args = ActionConditionedSetupArguments(
        model="2B/robot/action-cond",
        config_file="cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
        checkpoints_dir=args.checkpoints_dir,
        experiment=args.experiment,
        num_frames=13,
        dataset_path=args.dataset_path,
        save_dir=args.save_dir,
        output_dir=args.output_dir or args.save_dir,
        num_samples=1,
        data_split="full",
        single_base_index=False,
    )

    checkpoints_dir = Path(args.checkpoints_dir)
    last_checkpoint_file = checkpoints_dir / "latest_checkpoint.txt"
    if not last_checkpoint_file.exists():
        parent_file = checkpoints_dir.parent / "latest_checkpoint.txt"
        if parent_file.exists():
            checkpoints_dir = checkpoints_dir.parent
            last_checkpoint_file = parent_file

    if not last_checkpoint_file.exists():
        raise FileNotFoundError(f"Could not find latest_checkpoint.txt in {args.checkpoints_dir} or its parent.")

    with open(last_checkpoint_file) as f:
        last_checkpoint = f.read().strip()
    checkpoint_iter_dir = checkpoints_dir / last_checkpoint

    from examples.action_conditioned import resolve_checkpoint_path
    checkpoint_path = resolve_checkpoint_path(checkpoint_iter_dir)

    checkpoint = MODEL_CHECKPOINTS[setup_args.model_key]
    experiment = setup_args.experiment or checkpoint.experiment
    cache_config = build_cache_config_from_args(args)

    video2world_cli = Video2WorldInference(
        experiment_name=experiment,
        ckpt_path=checkpoint_path,
        s3_credential_path="",
        context_parallel_size=setup_args.context_parallel_size,
        config_file=setup_args.config_file,
        experiment_opts=build_experiment_opts(args),
        cache_config=cache_config,
    )

    return video2world_cli, checkpoint_iter_dir.name


def build_dataset(args):
    from groot_dreams.dataloader import MultiVideoActionDataset

    paths = [p.strip() for p in args.dataset_path.split(",") if p.strip()]
    valid_paths = [p for p in paths if list(Path(p).glob("data/*/*.parquet"))]

    dataset = MultiVideoActionDataset(
        num_frames=13,
        dataset_path=valid_paths,
        data_split="full",
        single_base_index=False,
        restrict_len=None,
        deterministic_uniform_sampling=False,
    )
    # ds_idx -> task name (dir basename), aligned with MultiVideoActionDataset order.
    task_names = [Path(p).name for p in valid_paths]
    return dataset, task_names


def get_episode_plan(dataset, chunk_size, task_names=None):
    """
    For each episode, collect data_ids stepping through by chunk_size.
    Each data_id provides chunk_size normalized actions via the dataset pipeline.
    """
    episodes = {}

    global_offset = 0
    for ds_idx, ds in enumerate(dataset.datasets):
        lerobot_ds = ds.lerobot_dataset
        for local_idx, (traj_id, base_index) in enumerate(lerobot_ds.all_steps):
            key = (ds_idx, int(traj_id))
            if key not in episodes:
                episodes[key] = []
            episodes[key].append((global_offset + local_idx, int(base_index)))
        global_offset += len(ds)

    delta_indices = dataset.datasets[0].lerobot_dataset.modality_configs["video"].delta_indices
    timestep_interval = delta_indices[1] - delta_indices[0]
    stride_raw = chunk_size * timestep_interval

    plan = []
    for key, steps in episodes.items():
        ds_idx, traj_id = key
        steps_sorted = sorted(steps, key=lambda x: x[1])
        if not steps_sorted:
            continue

        segment_indices = []
        next_base = 0
        for global_id, base_idx in steps_sorted:
            if base_idx >= next_base:
                segment_indices.append(global_id)
                next_base = base_idx + stride_raw

        if segment_indices:
            traj_length = int(dataset.datasets[ds_idx].lerobot_dataset.trajectory_lengths[
                np.where(dataset.datasets[ds_idx].lerobot_dataset.trajectory_ids == traj_id)[0][0]
            ])
            task_name = task_names[ds_idx] if task_names else None
            plan.append({
                "ds_idx": ds_idx,
                "task_name": task_name,
                "traj_id": int(traj_id),
                "traj_length": traj_length,
                "timestep_interval": int(timestep_interval),
                "segment_data_ids": segment_indices,
            })

    return plan


def main():
    args = parse_args()

    from cosmos_oss.init import init_environment, cleanup_environment
    init_environment()
    torch.enable_grad(False)

    print("Building model...")
    video2world_cli, iter_name = build_model(args)

    print("Building dataset...")
    dataset, task_names = build_dataset(args)

    print("Planning episodes...")
    plan = get_episode_plan(dataset, args.chunk_size, task_names)
    total_episodes = len(plan)
    num_episodes = min(args.num_episodes or total_episodes, total_episodes)
    print(f"Total episodes: {total_episodes}, processing: {num_episodes}")

    save_root = Path(args.save_dir) / iter_name
    save_root.mkdir(parents=True, exist_ok=True)

    all_psnr, all_ssim, all_lpips = [], [], []
    all_gen_times = []

    for ep_idx in range(num_episodes):
        ep_info = plan[ep_idx]
        traj_id = ep_info["traj_id"]
        task_name = ep_info.get("task_name")
        # Match DreamGen singleview naming: {task}__episode_{id:06d} (unique across tasks).
        if task_name:
            ep_dir_name = f"{task_name}__episode_{traj_id:06d}"
        else:
            ep_dir_name = f"episode_{traj_id:06d}"
        ep_save_dir = save_root / ep_dir_name

        pred_path = ep_save_dir / "full_pred.mp4"
        if pred_path.exists() and pred_path.stat().st_size > 1024:
            print(f"[{ep_idx}] {ep_dir_name} already exists, skipping.")
            continue
        if pred_path.exists():
            print(f"[{ep_idx}] {ep_dir_name} truncated pred ({pred_path.stat().st_size}B), regenerating.")
            pred_path.unlink()

        num_chunks = len(ep_info["segment_data_ids"])
        print(f"[{ep_idx}] {ep_dir_name} traj_id={traj_id}, traj_length={ep_info['traj_length']}, "
              f"chunks={num_chunks}")

        if num_chunks == 0:
            print("  No chunks, skipping.")
            continue

        ep_save_dir.mkdir(parents=True, exist_ok=True)

        # Get first frame from first segment
        first_sample = dataset[ep_info["segment_data_ids"][0]]
        img_array = first_sample["video"].transpose(0, 1)[:1]  # (1, C, H, W)

        gt_frames = []
        chunk_videos = []
        first_round = True

        gen_start_time = time.time()
        for chunk_idx, data_id in enumerate(ep_info["segment_data_ids"]):
            sample = dataset[data_id]

            video_tensor = sample["video"]
            gt_video_chunk = video_tensor.permute(1, 2, 3, 0).numpy()
            gt_frames.append(gt_video_chunk)

            actions = sample["action"][:args.chunk_size]
            if isinstance(actions, torch.Tensor):
                actions = actions.numpy()

            if actions.shape[0] != args.chunk_size:
                print(f"    chunk {chunk_idx}: only {actions.shape[0]} actions (need {args.chunk_size}), stopping.")
                break

            lam_video = sample.get("lam_video", None)
            current_lam_video = None
            if lam_video is not None and len(lam_video) >= args.chunk_size * 2:
                current_lam_video = lam_video[:args.chunk_size * 2]

            if not first_round:
                img_tensor = torchvision.transforms.functional.to_tensor(img_array).unsqueeze(0) * 255.0
            else:
                img_tensor = img_array
            first_round = False

            num_video_frames = actions.shape[0] + 1
            vid_input = torch.cat(
                [img_tensor, torch.zeros_like(img_tensor).repeat(num_video_frames - 1, 1, 1, 1)], dim=0
            )
            vid_input = vid_input.to(torch.uint8)
            vid_input = vid_input.unsqueeze(0).permute(0, 2, 1, 3, 4)

            video = video2world_cli.generate_vid2world(
                prompt="",
                input_path=vid_input,
                action=torch.from_numpy(actions).float()
                if isinstance(actions, np.ndarray)
                else actions,
                guidance=args.guidance,
                num_video_frames=num_video_frames,
                num_latent_conditional_frames=1,
                resolution="480,640",
                seed=chunk_idx,
                negative_prompt="The video captures a scene with low visual quality, blurring, jittering, or distortion.",
                lam_video=current_lam_video,
            )

            video_normalized = (video - (-1)) / (1 - (-1))
            video_clamped = (
                (torch.clamp(video_normalized[0], 0, 1) * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
            )

            # Pure autoregressive: use last predicted frame as next input
            img_array = video_clamped[-1]
            chunk_videos.append(video_clamped)

            print(f"    chunk {chunk_idx+1}/{num_chunks} done")

        if not chunk_videos:
            continue

        gen_elapsed = time.time() - gen_start_time
        all_gen_times.append(gen_elapsed)
        print(f"  [Timing] Episode {ep_idx}: generation took {gen_elapsed:.2f}s")

        chunk_list = [chunk_videos[0]] + [
            chunk_videos[i][:args.chunk_size] for i in range(1, len(chunk_videos))
        ]
        concat_pred = np.concatenate(chunk_list, axis=0)

        if args.save_video_only:
            # Trim prediction to the real GT episode length in the DOWNSAMPLED frame
            # space (dreamdojo samples every `timestep_interval` raw frames), so the
            # generated clip covers ~the full episode like DreamGen does.
            interval = max(1, int(ep_info.get("timestep_interval", 1)))
            expected_frames = -(-int(ep_info["traj_length"]) // interval)  # ceil div
            gt_total = min(len(concat_pred), expected_frames)
            concat_pred = concat_pred[:gt_total]
            mediapy.write_video(str(ep_save_dir / "full_pred.mp4"), concat_pred, fps=args.save_fps)
            print(f"  -> saved {len(concat_pred)} frames (video only), gen {gen_elapsed:.2f}s")
            continue

        gt_list = [gt_frames[0]] + [
            gt_frames[i][:args.chunk_size] for i in range(1, len(gt_frames))
        ]
        concat_gt = np.concatenate(gt_list, axis=0)

        min_len = min(len(concat_pred), len(concat_gt))
        concat_pred = concat_pred[:min_len]
        concat_gt = concat_gt[:min_len]

        mediapy.write_video(str(ep_save_dir / "full_pred.mp4"), concat_pred, fps=args.save_fps)
        mediapy.write_video(str(ep_save_dir / "full_gt.mp4"), concat_gt, fps=args.save_fps)
        concat_merged = np.concatenate([concat_gt, concat_pred], axis=2)
        mediapy.write_video(str(ep_save_dir / "full_merged.mp4"), concat_merged, fps=args.save_fps)

        x_batch = torch.clamp(torch.from_numpy(concat_pred.copy()) / 255.0, 0, 1).permute(0, 3, 1, 2)
        y_batch = torch.clamp(torch.from_numpy(concat_gt.copy()) / 255.0, 0, 1).permute(0, 3, 1, 2)
        psnr_val = piq.psnr(x_batch, y_batch).mean().item()
        ssim_val = piq.ssim(x_batch, y_batch).mean().item()
        lpips_val = piq.LPIPS()(x_batch, y_batch).mean().item()

        with open(ep_save_dir / "metrics.json", "w") as f:
            json.dump({
                "psnr": psnr_val, "ssim": ssim_val, "lpips": lpips_val,
                "num_chunks": len(chunk_videos),
                "total_frames_pred": len(concat_pred),
                "total_frames_gt": ep_info["traj_length"],
                "trajectory_id": traj_id,
                "task_name": task_name,
                "gen_time_s": round(gen_elapsed, 2),
                "mode": "single_segment_full_rollout",
            }, f, indent=2)

        all_psnr.append(psnr_val)
        all_ssim.append(ssim_val)
        all_lpips.append(lpips_val)
        print(f"  -> {len(concat_pred)} frames, PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}, LPIPS={lpips_val:.4f}")

    if all_psnr:
        summary = {
            "mean_psnr": sum(all_psnr) / len(all_psnr),
            "mean_ssim": sum(all_ssim) / len(all_ssim),
            "mean_lpips": sum(all_lpips) / len(all_lpips),
            "num_episodes_processed": len(all_psnr),
            "mode": "single_segment_full_rollout",
        }
        with open(save_root / "all_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n=== Summary ({len(all_psnr)} episodes) ===")
        print(f"PSNR: {summary['mean_psnr']:.3f}")
        print(f"SSIM: {summary['mean_ssim']:.4f}")
        print(f"LPIPS: {summary['mean_lpips']:.4f}")

    if all_gen_times:
        mean_gen_time = sum(all_gen_times) / len(all_gen_times)
        total_gen_time = sum(all_gen_times)
        print(f"\n[Timing] Generated {len(all_gen_times)} episodes in {total_gen_time:.2f}s total")
        print(f"[Timing] Average generation time per episode: {mean_gen_time:.2f}s")

    if os.environ.get("DD_ATTN_PROFILE") == "1":
        from cosmos_predict2._src.predict2.networks.minimal_v4_dit import _ATTN_PROFILE_STATS

        total = sum(v[1] for v in _ATTN_PROFILE_STATS.values())
        print(f"\n=== Attention profile (total {total:.2f}s of {sum(all_gen_times):.2f}s gen) ===")
        for key, (calls, secs) in sorted(_ATTN_PROFILE_STATS.items(), key=lambda kv: -kv[1][1]):
            backend, kind, shape = key
            print(
                f"{backend:>10} {kind:>5} q={shape}: {calls:6d} calls, {secs:7.2f}s "
                f"({secs / calls * 1000:.3f} ms/call)"
            )

    cleanup_environment()


if __name__ == "__main__":
    main()
