"""
Pure autoregressive rollout for single-arm singleview DreamDojo (DROID dataset).

For each episode:
  1. Take ONLY the first GT frame as conditioning.
  2. Use the full action sequence for the episode.
  3. Generate all frames autoregressively in chunks — each chunk uses the LAST predicted
     frame from the previous chunk (no GT re-anchoring).
  4. Output: one video per episode matching the GT episode length.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "models" / "DreamDojo"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", type=str, required=True)
    parser.add_argument("--experiment", type=str, default="dreamdojo_2b_480_640_single_arm_sv")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--num-frames", type=int, default=97,
                        help="Frames per segment (model generates num_frames-1 new frames + 1 conditioning)")
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--save-fps", type=int, default=15)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--guidance", type=float, default=0.0)

    # WorldCache
    parser.add_argument("--worldcache-enabled", action="store_true")
    parser.add_argument("--worldcache-num-steps", type=int, default=35)
    parser.add_argument("--worldcache-rel-l1-thresh", type=float, default=0.03)
    parser.add_argument("--worldcache-ret-ratio", type=float, default=0.4)
    parser.add_argument("--worldcache-probe-depth", type=int, default=4)
    parser.add_argument("--worldcache-motion-sensitivity", type=float, default=5.0)

    # DiCache
    parser.add_argument("--dicache-enabled", action="store_true")
    parser.add_argument("--dicache-num-steps", type=int, default=35)
    parser.add_argument("--dicache-rel-l1-thresh", type=float, default=0.08)
    parser.add_argument("--dicache-ret-ratio", type=float, default=0.2)
    parser.add_argument("--dicache-probe-depth", type=int, default=2)

    # FasterCache
    parser.add_argument("--fastercache-enabled", action="store_true")
    parser.add_argument("--fastercache-start-step", type=int, default=0)
    parser.add_argument("--fastercache-model-interval", type=int, default=5)
    parser.add_argument("--fastercache-block-interval", type=int, default=3)

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
    """Build a cache strategy config from CLI flags (WorldCache / DiCache / FasterCache)."""
    from methods.cache_strategy.common import WorldCacheConfig, DiCacheConfig, FasterCacheConfig

    if getattr(args, "worldcache_enabled", False):
        return WorldCacheConfig(
            num_steps=args.worldcache_num_steps,
            rel_l1_thresh=args.worldcache_rel_l1_thresh,
            ret_ratio=args.worldcache_ret_ratio,
            probe_depth=args.worldcache_probe_depth,
            motion_sensitivity=args.worldcache_motion_sensitivity,
        )
    if getattr(args, "dicache_enabled", False):
        return DiCacheConfig(
            num_steps=args.dicache_num_steps,
            rel_l1_thresh=args.dicache_rel_l1_thresh,
            ret_ratio=args.dicache_ret_ratio,
            probe_depth=args.dicache_probe_depth,
        )
    if getattr(args, "fastercache_enabled", False):
        return FasterCacheConfig(
            start_step=args.fastercache_start_step,
            model_interval=args.fastercache_model_interval,
            block_interval=args.fastercache_block_interval,
        )
    return None


def build_model(args):
    from cosmos_predict2.action_conditioned_config import ActionConditionedSetupArguments
    from cosmos_predict2.config import MODEL_CHECKPOINTS
    from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference


    # Resolve the LOCAL checkpoint FIRST.
    #
    # `ActionConditionedSetupArguments` runs a pydantic `validate_model` hook that
    # fills `checkpoint_path` from `MODEL_CHECKPOINTS[...]` when it is not given,
    # and that default is a *property* which eagerly downloads from the gated HF
    # repo `nvidia/Cosmos-Predict2.5-2B` -> 401 GatedRepoError, aborting before a
    # single frame is generated even though `--checkpoints-dir` points at a local
    # finetuned checkpoint we hand to Video2WorldInference anyway. Resolving
    # locally first and passing the path keeps the validator off the network.
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

    setup_args = ActionConditionedSetupArguments(
        model="2B/robot/action-cond",
        config_file="cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
        checkpoint_path=str(checkpoint_path),
        checkpoints_dir=args.checkpoints_dir,
        experiment=args.experiment,
        num_frames=args.num_frames,
        dataset_path=args.dataset_path,
        save_dir=args.save_dir,
        output_dir=args.output_dir or args.save_dir,
        num_samples=1,
        data_split="full",
        single_base_index=False,
    )

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

    # Fixed 13-frame windows (matching the model's state_t / chunk pipeline),
    # so all_steps provides dense base indices up to traj_length-13. The
    # chunk_size-strided plan then walks the full episode. (args.num_frames is
    # kept for CLI compatibility but does not bound the rollout length.)
    dataset = MultiVideoActionDataset(
        num_frames=13,
        dataset_path=args.dataset_path,
        data_split="full",
        single_base_index=False,
        restrict_len=None,
        deterministic_uniform_sampling=False,
    )
    return dataset


def get_episode_plan(dataset, chunk_size):
    """Group dataset indices by episode. Return plan with chunk-strided segment indices.

    Steps through each episode in increments of chunk_size timesteps so the
    autoregressive rollout covers the ENTIRE episode (matching the input video
    length), rather than stopping after a single num_frames window.
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
            plan.append({
                "ds_idx": ds_idx,
                "traj_id": int(traj_id),
                "traj_length": traj_length,
                "segment_data_ids": segment_indices,
            })

    return plan


_CHUNK_TIMES = []

# DD_MAX_CHUNKS / CW_MAX_CHUNKS: stop generating after N chunks.
#
# The benchmark unit is ONE chunk, and per-chunk cost is flat within an episode,
# so generating all 9-33 chunks of an episode adds GPU time without adding
# statistical power. The timing harness sets this to a small number; unset or 0
# keeps the full episode, so canonical runs are unaffected.
def _max_chunks():
    import os
    for k in ("BENCH_MAX_CHUNKS", "DD_MAX_CHUNKS", "CW_MAX_CHUNKS"):
        v = os.environ.get(k)
        if v and v.strip().isdigit() and int(v) > 0:
            return int(v)
    return 0


def generate_autoregressive(video2world_cli, first_frame, all_actions, chunk_size, all_lam_video=None, guidance=0):
    """
    Pure autoregressive rollout from a single GT frame.
    Each chunk uses the last predicted frame as conditioning (no GT).
    """
    img_array = first_frame
    chunk_video = []
    first_round = True

    for i in range(0, len(all_actions), chunk_size):
        actions_chunk = all_actions[i: i + chunk_size]
        if actions_chunk.shape[0] != chunk_size:
            break

        current_lam_video = None
        if all_lam_video is not None:
            current_lam_video = all_lam_video[i * 2: (i + chunk_size) * 2]
            if current_lam_video is not None and len(current_lam_video) < chunk_size * 2:
                current_lam_video = None

        if not first_round:
            img_tensor = torchvision.transforms.functional.to_tensor(img_array).unsqueeze(0) * 255.0
        else:
            img_tensor = img_array
        first_round = False

        num_video_frames = actions_chunk.shape[0] + 1
        vid_input = torch.cat(
            [img_tensor, torch.zeros_like(img_tensor).repeat(num_video_frames - 1, 1, 1, 1)], dim=0
        )
        vid_input = vid_input.to(torch.uint8)
        vid_input = vid_input.unsqueeze(0).permute(0, 2, 1, 3, 4)

        _chunk_t0 = time.time()
        video = video2world_cli.generate_vid2world(
            prompt="",
            input_path=vid_input,
            action=torch.from_numpy(actions_chunk).float()
            if isinstance(actions_chunk, np.ndarray)
            else actions_chunk,
            guidance=guidance,
            num_video_frames=num_video_frames,
            num_latent_conditional_frames=1,
            resolution="480,640",
            seed=i,
            negative_prompt="The video captures a scene with low visual quality, blurring, jittering, or distortion.",
            lam_video=current_lam_video,
        )
        _chunk_dt = time.time() - _chunk_t0
        _CHUNK_TIMES.append(_chunk_dt)
        print(f"  [ChunkTiming] chunk {len(_CHUNK_TIMES)}: {_chunk_dt:.3f}s", flush=True)
        _cap = _max_chunks()
        if _cap and len(_CHUNK_TIMES) >= _cap:
            print(f"  [ChunkTiming] stopping after {_cap} chunk(s) (cap)", flush=True)
            break

        video_normalized = (video - (-1)) / (1 - (-1))
        video_clamped = (
            (torch.clamp(video_normalized[0], 0, 1) * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
        )
        img_array = video_clamped[-1]
        chunk_video.append(video_clamped)

    if not chunk_video:
        return None

    chunk_list = [chunk_video[0]] + [
        chunk_video[i][:chunk_size] for i in range(1, len(chunk_video))
    ]
    return np.concatenate(chunk_list, axis=0)


def main():
    args = parse_args()

    from cosmos_oss.init import init_environment, cleanup_environment
    init_environment()
    torch.enable_grad(False)

    print("Building model...")
    video2world_cli, iter_name = build_model(args)

    print("Building dataset...")
    dataset = build_dataset(args)

    print("Planning episodes...")
    plan = get_episode_plan(dataset, args.chunk_size)
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
        traj_length = ep_info["traj_length"]
        ep_save_dir = save_root / f"episode_{traj_id:06d}"

        if (ep_save_dir / "full_pred.mp4").exists() and (ep_save_dir / "metrics.json").exists():
            print(f"[{ep_idx}/{num_episodes}] episode_{traj_id:06d} already exists, loading metrics.")
            try:
                with open(ep_save_dir / "metrics.json") as f:
                    m = json.load(f)
                if m.get("psnr") is not None:
                    all_psnr.append(m["psnr"])
                    all_ssim.append(m["ssim"])
                    all_lpips.append(m["lpips"])
            except (json.JSONDecodeError, KeyError):
                pass
            continue

        num_segments = len(ep_info["segment_data_ids"])
        print(f"[{ep_idx}/{num_episodes}] traj_id={traj_id}, "
              f"length={traj_length}, segments={num_segments}")

        # Load all segments for this episode
        all_actions_parts = []
        all_lam_parts = []
        gt_segments = []

        for seg_idx, data_id in enumerate(ep_info["segment_data_ids"]):
            sample = dataset[data_id]
            actions = sample["action"][:args.chunk_size]
            if isinstance(actions, torch.Tensor):
                actions = actions.numpy()
            all_actions_parts.append(actions)

            lam = sample.get("lam_video", None)
            if lam is not None:
                all_lam_parts.append(lam[:args.chunk_size * 2])

            gt_seg = sample["video"].permute(1, 2, 3, 0).numpy()
            gt_segments.append(gt_seg)

        # First frame from the first segment (the only GT we use)
        first_sample = dataset[ep_info["segment_data_ids"][0]]
        first_frame = first_sample["video"].transpose(0, 1)[:1]  # (1, C, H, W)

        # Concatenate all actions into one continuous stream
        full_actions = np.concatenate(all_actions_parts, axis=0)

        # Concatenate lam_video if available
        full_lam = None
        if all_lam_parts:
            full_lam = torch.cat(all_lam_parts, dim=0) if isinstance(all_lam_parts[0], torch.Tensor) else None

        # Pure autoregressive rollout from first GT frame only
        # The chunk budget is PER EPISODE, so clear the accumulator here or the
        # first episode spends it all and every later episode records nothing.
        _CHUNK_TIMES.clear()
        gen_start_time = time.time()
        pred_video = generate_autoregressive(
            video2world_cli, first_frame, full_actions, args.chunk_size, full_lam,
            guidance=args.guidance,
        )
        gen_elapsed = time.time() - gen_start_time

        if pred_video is None:
            print(f"  Skipping episode {traj_id}: could not generate any frames.")
            continue

        all_gen_times.append(gen_elapsed)
        print(f"  [Timing] Episode {ep_idx}: generation took {gen_elapsed:.2f}s")

        # Build GT by concatenating segments: first segment full, subsequent
        # segments contribute only their first chunk_size frames (segments are
        # chunk_size-strided so the tail would otherwise overlap).
        gt_list = [gt_segments[0]] + [
            gt_segments[i][:args.chunk_size] for i in range(1, len(gt_segments))
        ]
        concat_gt = np.concatenate(gt_list, axis=0)

        # Trim to same length
        min_len = min(len(pred_video), len(concat_gt))
        pred_video = pred_video[:min_len]
        concat_gt = concat_gt[:min_len]

        ep_save_dir.mkdir(parents=True, exist_ok=True)
        mediapy.write_video(str(ep_save_dir / "full_pred.mp4"), pred_video, fps=args.save_fps)
        mediapy.write_video(str(ep_save_dir / "full_gt.mp4"), concat_gt, fps=args.save_fps)
        concat_merged = np.concatenate([concat_gt, pred_video], axis=2)
        mediapy.write_video(str(ep_save_dir / "full_merged.mp4"), concat_merged, fps=args.save_fps)

        # Compute metrics
        x_batch = torch.clamp(torch.from_numpy(pred_video.copy()) / 255.0, 0, 1).permute(0, 3, 1, 2)
        y_batch = torch.clamp(torch.from_numpy(concat_gt.copy()) / 255.0, 0, 1).permute(0, 3, 1, 2)
        try:
            psnr_val = piq.psnr(x_batch, y_batch).mean().item()
            ssim_val = piq.ssim(x_batch, y_batch).mean().item()
            lpips_val = piq.LPIPS()(x_batch, y_batch).mean().item()
        except (RuntimeError, ValueError) as e:
            print(f"  Metrics failed: {e}")
            psnr_val = ssim_val = lpips_val = None

        with open(ep_save_dir / "metrics.json", "w") as f:
            json.dump({
                "psnr": psnr_val, "ssim": ssim_val, "lpips": lpips_val,
                "total_frames_pred": len(pred_video),
                "total_frames_gt": traj_length,
                "trajectory_id": traj_id,
                "gen_time_s": round(gen_elapsed, 2),
                "mode": "autoregressive",
            }, f, indent=2)

        if psnr_val is not None:
            all_psnr.append(psnr_val)
            all_ssim.append(ssim_val)
            all_lpips.append(lpips_val)
        print(f"  -> {len(pred_video)} frames, PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}, LPIPS={lpips_val:.4f}")

    if all_psnr:
        summary = {
            "mean_psnr": sum(all_psnr) / len(all_psnr),
            "mean_ssim": sum(all_ssim) / len(all_ssim),
            "mean_lpips": sum(all_lpips) / len(all_lpips),
            "num_episodes_processed": len(all_psnr),
            "mode": "autoregressive",
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
