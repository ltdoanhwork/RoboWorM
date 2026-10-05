"""
Single-segment full-length autoregressive rollout for single-arm singleview DreamDojo.

For each episode:
  - Start from the first GT frame.
  - Rollout autoregressively (chunk_size actions per step) for the ENTIRE episode
    length — no GT reset between segments. Pure autoregressive drift test.
  - GT video and actions come from the dataset pipeline (properly normalized/resized).

Output matches the benchmark format: full_gt.mp4, full_pred.mp4, full_merged.mp4, metrics.json.
"""

import argparse
import json
import sys
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
    parser.add_argument("--experiment", type=str, default="dreamdojo_2b_480_640_single_arm_sv")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--save-fps", type=int, default=15)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def build_model(args):
    from cosmos_predict2.action_conditioned_config import ActionConditionedSetupArguments
    from cosmos_predict2.config import MODEL_CHECKPOINTS
    from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
    from cosmos_predict2.cache_runtime import build_cache_runtime_config


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
        num_frames=13,
        dataset_path=args.dataset_path,
        save_dir=args.save_dir,
        output_dir=args.output_dir or args.save_dir,
        num_samples=1,
        data_split="full",
        single_base_index=False,
    )

    checkpoint = MODEL_CHECKPOINTS[setup_args.model_key]
    experiment = setup_args.experiment or checkpoint.experiment
    cache_config = build_cache_runtime_config(setup_args)

    video2world_cli = Video2WorldInference(
        experiment_name=experiment,
        ckpt_path=checkpoint_path,
        s3_credential_path="",
        context_parallel_size=setup_args.context_parallel_size,
        config_file=setup_args.config_file,
        experiment_opts=[],
        cache_config=cache_config,
    )

    return video2world_cli, checkpoint_iter_dir.name


def build_dataset(args):
    """Build dataset with single_base_index=False to access all base_indices per episode."""
    from groot_dreams.dataloader import MultiVideoActionDataset

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
    """
    For each episode, collect data_ids whose actions cover the full episode.
    Each data_id provides chunk_size actions. We step through the episode
    in increments of chunk_size timesteps.
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
    num_episodes = min(args.num_episodes, total_episodes)
    print(f"Total episodes: {total_episodes}, testing: {num_episodes}")

    save_root = Path(args.save_dir) / iter_name
    save_root.mkdir(parents=True, exist_ok=True)

    all_psnr, all_ssim, all_lpips = [], [], []

    for ep_idx in range(num_episodes):
        ep_info = plan[ep_idx]
        traj_id = ep_info["traj_id"]
        ep_save_dir = save_root / f"episode_{traj_id:06d}"

        if (ep_save_dir / "full_pred.mp4").exists():
            print(f"[{ep_idx}] episode_{traj_id:06d} already exists, skipping.")
            continue

        num_chunks = len(ep_info["segment_data_ids"])
        print(f"[{ep_idx}] traj_id={traj_id}, traj_length={ep_info['traj_length']}, "
              f"chunks={num_chunks}")

        if num_chunks == 0:
            print("  No chunks, skipping.")
            continue

        ep_save_dir.mkdir(parents=True, exist_ok=True)

        # Get first frame from first segment (GT, properly resized by dataset pipeline)
        first_sample = dataset[ep_info["segment_data_ids"][0]]
        img_array = first_sample["video"].transpose(0, 1)[:1]  # (1, C, H, W) uint8

        # Collect GT video frames and actions from all segments
        gt_frames = []
        chunk_videos = []
        first_round = True

        for chunk_idx, data_id in enumerate(ep_info["segment_data_ids"]):
            sample = dataset[data_id]

            # GT frames for this chunk
            video_tensor = sample["video"]  # (C, T, H, W)
            gt_video_chunk = video_tensor.permute(1, 2, 3, 0).numpy()  # (T, H, W, C)
            gt_frames.append(gt_video_chunk)

            # Actions for this chunk (already normalized by dataset pipeline)
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

            # Prepare input frame
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
                guidance=0,
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

            # Use last predicted frame as next input (NO GT reset — pure autoregressive)
            img_array = video_clamped[-1]
            chunk_videos.append(video_clamped)

            print(f"    chunk {chunk_idx+1}/{num_chunks} done")

        if not chunk_videos:
            continue

        # Concatenate: first chunk full, subsequent chunks drop conditioning frame
        chunk_list = [chunk_videos[0]] + [
            chunk_videos[i][:args.chunk_size] for i in range(1, len(chunk_videos))
        ]
        concat_pred = np.concatenate(chunk_list, axis=0)

        # GT: similarly concatenate
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

    cleanup_environment()


if __name__ == "__main__":
    main()
