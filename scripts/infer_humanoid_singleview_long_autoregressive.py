"""
Long autoregressive rollout for humanoid singleview DreamDojo using generate_vid2world_long.

Uses the built-in chunk_overlap mechanism for smoother transitions between chunks.
Only the first GT frame is used as conditioning — pure autoregressive after that.

For each episode:
  1. Take the first GT frame as conditioning.
  2. Collect the full action sequence.
  3. Call generate_vid2world_long with chunk_overlap for smooth long-horizon generation.
  4. Output: one video per episode matching the GT episode length.
"""

import argparse
import json
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
    parser.add_argument("--experiment", type=str, default="dreamdojo_2b_480_640_gr1")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--num-frames", type=int, default=49,
                        help="Model's native chunk size (frames per forward pass)")
    parser.add_argument("--chunk-overlap", type=int, default=4,
                        help="Number of overlapping frames between chunks for smooth transitions")
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--save-fps", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--guidance", type=float, default=0)
    parser.add_argument("--num-latent-conditional-frames", type=int, default=1,
                        help="Latent conditional frames (1=image2world, 2=video2world with 5 pixel frames)")
    parser.add_argument("--resolution", type=str, default="480,640")
    return parser.parse_args()


def build_model(args):
    from cosmos_predict2.action_conditioned_config import ActionConditionedSetupArguments
    from cosmos_predict2.config import MODEL_CHECKPOINTS
    from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference

    setup_args = ActionConditionedSetupArguments(
        model="2B/robot/action-cond",
        config_file="cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
        checkpoints_dir=args.checkpoints_dir,
        experiment=args.experiment,
        num_frames=args.num_frames,
        dataset_path=args.dataset_path,
        save_dir=args.save_dir,
        output_dir=args.output_dir or args.save_dir,
        num_samples=1,
        data_split="full",
        single_base_index=True,
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

    video2world_cli = Video2WorldInference(
        experiment_name=experiment,
        ckpt_path=checkpoint_path,
        s3_credential_path="",
        context_parallel_size=setup_args.context_parallel_size,
        config_file=setup_args.config_file,
        experiment_opts=[],
        cache_config=None,
    )

    return video2world_cli, checkpoint_iter_dir.name


def build_dataset(args):
    from groot_dreams.dataloader import MultiVideoActionDataset

    dataset = MultiVideoActionDataset(
        num_frames=args.num_frames,
        dataset_path=args.dataset_path,
        data_split="full",
        single_base_index=False,
        restrict_len=None,
        deterministic_uniform_sampling=False,
    )
    return dataset


def get_episode_plan(dataset, num_frames):
    """Group dataset indices by episode. Return plan with non-overlapping segment indices."""
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
    stride_raw = (num_frames - 1) * timestep_interval

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


def generate_long_autoregressive(video2world_cli, first_frame_tensor, all_actions,
                                  num_frames, chunk_overlap, guidance, resolution,
                                  num_latent_conditional_frames, lam_video=None):
    """
    Chunked autoregressive generation with overlap for smooth long-horizon videos.

    Manually implements chunk overlap because generate_autoregressive_from_batch
    does not slice actions per chunk (passes full action tensor → shape mismatch).

    Each chunk generates num_frames pixel frames using (num_frames - 1) actions.
    Chunks overlap by chunk_overlap frames: the last chunk_overlap frames of chunk N
    become the conditioning context for chunk N+1.

    Args:
        first_frame_tensor: (1, C, H, W) uint8 tensor of the first GT frame
        all_actions: numpy array of shape (total_actions, action_dim)
        num_frames: model's native capacity (pixel frames per forward pass)
        chunk_overlap: number of overlapping frames between chunks
        guidance: CFG scale
        resolution: "H,W" string
        num_latent_conditional_frames: 1 or 2
        lam_video: optional LAM video tensor
    """
    actions_per_chunk = num_frames - 1
    total_actions = len(all_actions)

    generated_chunks = []
    cond_frames = first_frame_tensor  # (1, C, H, W) for first chunk
    action_offset = 0
    chunk_idx = 0

    while action_offset < total_actions:
        remaining_actions = total_actions - action_offset
        chunk_actions_len = min(actions_per_chunk, remaining_actions)

        # Need at least chunk_overlap+1 actions for subsequent chunks to produce new frames,
        # and at least 2 actions for the first chunk
        if chunk_idx == 0 and chunk_actions_len < 2:
            break
        if chunk_idx > 0 and chunk_actions_len <= chunk_overlap:
            break

        actions_chunk = all_actions[action_offset: action_offset + chunk_actions_len]
        if isinstance(actions_chunk, np.ndarray):
            actions_chunk = torch.from_numpy(actions_chunk).float()

        num_video_frames = chunk_actions_len + 1

        # Build video input: conditioning frames + zeros
        if chunk_idx == 0:
            # First chunk: single GT frame
            vid_input = torch.cat(
                [cond_frames, torch.zeros_like(cond_frames).repeat(num_video_frames - 1, 1, 1, 1)],
                dim=0,
            )
        else:
            # Subsequent chunks: use last chunk_overlap frames from previous output
            num_cond = cond_frames.shape[0]  # chunk_overlap frames
            num_new = num_video_frames - num_cond
            if num_new <= 0:
                break
            vid_input = torch.cat(
                [cond_frames, torch.zeros(num_new, *cond_frames.shape[1:])],
                dim=0,
            )

        vid_input = vid_input.to(torch.uint8)
        vid_input = vid_input.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (1, C, T, H, W)

        # LAM video slice if available
        current_lam = None
        if lam_video is not None:
            lam_start = action_offset * 2
            lam_end = lam_start + chunk_actions_len * 2
            if lam_end <= len(lam_video):
                current_lam = lam_video[lam_start:lam_end]

        # Determine num_latent_conditional_frames for this chunk
        if chunk_idx == 0:
            chunk_cond_frames = num_latent_conditional_frames
        else:
            # For subsequent chunks, we condition on chunk_overlap pixel frames.
            # Tokenizer compresses time 4x: chunk_overlap pixel frames → (chunk_overlap+3)//4 latent frames.
            # Model only accepts 1 or 2 for num_latent_conditional_frames.
            chunk_cond_frames = min(2, max(1, (chunk_overlap + 3) // 4))

        video = video2world_cli.generate_vid2world(
            prompt="",
            input_path=vid_input,
            action=actions_chunk,
            guidance=guidance,
            num_video_frames=num_video_frames,
            num_latent_conditional_frames=chunk_cond_frames,
            resolution=resolution,
            seed=chunk_idx,
            negative_prompt="The video captures a scene with low visual quality, blurring, jittering, or distortion.",
            lam_video=current_lam,
        )

        # Convert output from [-1, 1] to uint8 numpy
        video_normalized = (video - (-1)) / (1 - (-1))
        video_clamped = (
            (torch.clamp(video_normalized[0], 0, 1) * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
        )

        if chunk_idx == 0:
            generated_chunks.append(video_clamped)
        else:
            # Remove overlap frames from beginning (they were conditioning context)
            generated_chunks.append(video_clamped[chunk_overlap:])

        # Prepare conditioning for next chunk: last chunk_overlap frames as pixel tensor
        tail_frames = video_clamped[-chunk_overlap:]  # (overlap, H, W, C) uint8 numpy
        cond_frames = torch.from_numpy(tail_frames).permute(0, 3, 1, 2).float()  # (overlap, C, H, W)

        # Advance: new frames produced = chunk_actions_len + 1 - chunk_overlap (for non-first)
        # Actions consumed for new (non-overlapping) output:
        #   chunk 0: all actions_per_chunk actions advance the timeline
        #   chunk N>0: we re-use chunk_overlap frames as context, so only advance by (chunk_actions_len - chunk_overlap)
        if chunk_idx == 0:
            action_offset += chunk_actions_len
        else:
            action_offset += chunk_actions_len - chunk_overlap

        chunk_idx += 1

    if not generated_chunks:
        return None

    return np.concatenate(generated_chunks, axis=0)


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
    plan = get_episode_plan(dataset, args.num_frames)
    total_episodes = len(plan)
    num_episodes = min(args.num_episodes or total_episodes, total_episodes)
    print(f"Total episodes: {total_episodes}, processing: {num_episodes}")

    save_root = Path(args.save_dir) / iter_name
    save_root.mkdir(parents=True, exist_ok=True)

    run_config = {
        "checkpoint": str(Path(args.checkpoints_dir).resolve()),
        "iter": iter_name,
        "experiment": args.experiment,
        "dataset_path": args.dataset_path,
        "num_frames": args.num_frames,
        "chunk_overlap": args.chunk_overlap,
        "num_latent_conditional_frames": args.num_latent_conditional_frames,
        "guidance": args.guidance,
        "resolution": args.resolution,
        "save_fps": args.save_fps,
        "num_episodes": num_episodes,
        "total_episodes": total_episodes,
        "mode": "long_autoregressive",
    }
    with open(save_root / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    all_psnr, all_ssim, all_lpips = [], [], []
    all_gen_times = []

    for ep_idx in range(num_episodes):
        ep_info = plan[ep_idx]
        traj_id = ep_info["traj_id"]
        ep_save_dir = save_root / f"episode_{ep_idx:04d}"

        if (ep_save_dir / "full_pred.mp4").exists() and (ep_save_dir / "metrics.json").exists():
            print(f"[{ep_idx}/{num_episodes}] episode_{ep_idx:04d} already exists, loading metrics.")
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
              f"length={ep_info['traj_length']}, segments={num_segments}")

        # Collect all actions and GT frames from all segments
        all_actions_parts = []
        all_lam_parts = []
        gt_segments = []

        for seg_idx, data_id in enumerate(ep_info["segment_data_ids"]):
            sample = dataset[data_id]
            actions = sample["action"][:args.num_frames - 1]
            if isinstance(actions, torch.Tensor):
                actions = actions.numpy()
            all_actions_parts.append(actions)

            lam = sample.get("lam_video", None)
            if lam is not None:
                all_lam_parts.append(lam)

            gt_seg = sample["video"].permute(1, 2, 3, 0).numpy()
            gt_segments.append(gt_seg)

        # First frame from the first segment
        first_sample = dataset[ep_info["segment_data_ids"][0]]
        first_frame = first_sample["video"].transpose(0, 1)[:1]  # (1, C, H, W)

        full_actions = np.concatenate(all_actions_parts, axis=0)

        full_lam = None
        if all_lam_parts:
            full_lam = torch.cat(all_lam_parts, dim=0) if isinstance(all_lam_parts[0], torch.Tensor) else None

        gen_start_time = time.time()

        pred_video = generate_long_autoregressive(
            video2world_cli,
            first_frame,
            full_actions,
            num_frames=args.num_frames,
            chunk_overlap=args.chunk_overlap,
            guidance=args.guidance,
            resolution=args.resolution,
            num_latent_conditional_frames=args.num_latent_conditional_frames,
            lam_video=full_lam,
        )

        gen_elapsed = time.time() - gen_start_time

        if pred_video is None or len(pred_video) == 0:
            print(f"  Skipping episode {traj_id}: could not generate any frames.")
            continue

        all_gen_times.append(gen_elapsed)

        # Build GT
        concat_gt = np.concatenate(gt_segments, axis=0)

        # Trim to same length
        min_len = min(len(pred_video), len(concat_gt))
        if len(pred_video) != len(concat_gt):
            print(f"  [Info] Frame mismatch: pred={len(pred_video)}, gt={len(concat_gt)}, using min={min_len}")
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
                "gen_time_s": round(gen_elapsed, 2),
                "total_frames_pred": len(pred_video),
                "total_frames_gt": ep_info["traj_length"],
                "trajectory_id": traj_id,
                "chunk_overlap": args.chunk_overlap,
                "num_latent_conditional_frames": args.num_latent_conditional_frames,
                "mode": "long_autoregressive",
            }, f, indent=2)

        if psnr_val is not None:
            all_psnr.append(psnr_val)
            all_ssim.append(ssim_val)
            all_lpips.append(lpips_val)
        print(f"  -> {len(pred_video)} frames ({gen_elapsed:.1f}s), "
              f"PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}, LPIPS={lpips_val:.4f}")

    # Summary
    timing_summary = {}
    if all_gen_times:
        mean_gen_time = sum(all_gen_times) / len(all_gen_times)
        total_gen_time = sum(all_gen_times)
        print(f"\n[Timing] Generated {len(all_gen_times)} episodes in {total_gen_time:.2f}s total")
        print(f"[Timing] Average generation time per episode: {mean_gen_time:.2f}s")
        timing_summary = {
            "num_episodes_generated": len(all_gen_times),
            "total_gen_time_s": round(total_gen_time, 2),
            "avg_gen_time_s": round(mean_gen_time, 2),
        }

    if all_psnr:
        summary = {
            "mean_psnr": sum(all_psnr) / len(all_psnr),
            "mean_ssim": sum(all_ssim) / len(all_ssim),
            "mean_lpips": sum(all_lpips) / len(all_lpips),
            "num_episodes_processed": len(all_psnr),
            "chunk_overlap": args.chunk_overlap,
            "num_latent_conditional_frames": args.num_latent_conditional_frames,
            "mode": "long_autoregressive",
            **timing_summary,
        }
        with open(save_root / "all_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n=== Summary ({len(all_psnr)} episodes) ===")
        print(f"PSNR: {summary['mean_psnr']:.3f}")
        print(f"SSIM: {summary['mean_ssim']:.4f}")
        print(f"LPIPS: {summary['mean_lpips']:.4f}")
    else:
        with open(save_root / "all_summary.json", "w") as f:
            json.dump({"psnr": None, "ssim": None, "lpips": None, **timing_summary}, f, indent=2)

    cleanup_environment()


if __name__ == "__main__":
    main()
