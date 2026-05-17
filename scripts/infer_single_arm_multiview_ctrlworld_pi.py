"""
Policy-conditioned full-episode inference using Ctrl-World on single_arm/multiview dataset.

Uses pi05 policy to predict actions (instead of GT replay), then generates video
frames autoregressively via Ctrl-World. Saves in benchmark format for comparison.

Requires: prepare_ctrlworld_single_arm_multiview.py to be run first.
"""

import sys
import os
import json
import datetime
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import torch
import einops
import mediapy
import piq

ctrl_world_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "Ctrl-World"
)
# Ctrl-World must be first to resolve its internal 'models' package before project root
sys.path.insert(0, os.path.join(ctrl_world_dir, "scripts"))
sys.path.insert(0, os.path.join(ctrl_world_dir, "openpi", "packages", "openpi-client", "src"))
sys.path.insert(0, os.path.join(ctrl_world_dir, "openpi", "src"))
sys.path.insert(0, ctrl_world_dir)

from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.ctrl_world import CrtlWorld
from decord import VideoReader, cpu
from accelerate import Accelerator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_EXAMPLE_BASE = os.path.join(PROJECT_ROOT, "models", "Ctrl-World", "dataset_example")


def compute_metrics(pred_frames, gt_frames):
    x = torch.clamp(torch.from_numpy(pred_frames.copy()) / 255.0, 0, 1).permute(0, 3, 1, 2)
    y = torch.clamp(torch.from_numpy(gt_frames.copy()) / 255.0, 0, 1).permute(0, 3, 1, 2)
    psnr_val = piq.psnr(x, y).mean().item()
    ssim_val = piq.ssim(x, y).mean().item()
    lpips_val = piq.LPIPS()(x, y).mean().item()
    return {"psnr": psnr_val, "ssim": ssim_val, "lpips": lpips_val}


def run_episode_with_policy(agent, episode_id, save_dir, input_save_dir=None):
    """Run full-episode policy-driven generation for one episode."""
    args = agent.args
    pred_step = args.pred_step  # 5
    num_history = args.num_history  # 6

    anno_path = f"{args.val_dataset_dir}/annotation/val/{episode_id}.json"
    with open(anno_path) as f:
        anno = json.load(f)
    episode_length = anno["video_length"]

    interact_num = (episode_length - 1) // (pred_step - 1)
    if interact_num < 1:
        print(f"  Episode {episode_id} too short ({episode_length} frames), skipping")
        return None

    total_steps_needed = (interact_num - 1) * (pred_step - 1) + pred_step
    instruction = anno["texts"][0]

    # Load GT trajectory for comparison (we still need it for metrics)
    eef_gt, joint_pos_gt, video_dict, video_latents, _ = agent.get_traj_info(
        episode_id, start_idx=0, steps=min(total_steps_needed, episode_length)
    )

    # Initialize history buffers
    his_cond = []
    his_joint = []
    his_eef = []
    first_latent = torch.cat([v[0] for v in video_latents], dim=1).unsqueeze(0)
    for _ in range(num_history * 4):
        his_cond.append(first_latent)
        his_joint.append(joint_pos_gt[0:1])
        his_eef.append(eef_gt[0:1])

    video_dict_pred = [v[0:1] for v in video_dict]
    video_to_save_pred = []
    video_to_save_gt = []
    history_idx = args.history_idx

    for i in range(interact_num):
        start_id = int(i * (pred_step - 1))
        end_id = start_id + pred_step

        if end_id > len(eef_gt):
            break

        video_latent_true = [v[start_id:end_id] for v in video_latents]

        # Policy forward: predict actions from current observations
        current_joint = his_joint[-1][0]
        current_pose = his_eef[-1][0]
        current_obs = [v[-1] for v in video_dict_pred]
        policy_in_out, joint_pos_pred, cartesian_pose = agent.forward_policy(
            current_obs, current_pose, current_joint, text=instruction
        )

        # World model forward with policy-predicted actions
        action_cond = np.concatenate([his_eef[idx] for idx in history_idx], axis=0)
        action_cond = np.concatenate([action_cond, cartesian_pose], axis=0)
        his_latent = torch.cat([his_cond[idx] for idx in history_idx], dim=0).unsqueeze(0)
        current_latent = his_cond[-1]

        # forward_wm returns: (videos_cat, true_videos, pred_videos, latents)
        # true_videos/pred_videos shape: (3_views, frames, H, W, 3)
        # latents shape: (3_views, frames, C, h, w)
        videos_cat, true_videos, pred_videos, predict_latents = agent.forward_wm(
            action_cond, video_latent_true, current_latent,
            his_cond=his_latent,
            text=instruction if args.text_cond else None,
        )

        # Update history with predicted latents (not GT)
        # predict_latents: (3, f, c, h, w) — cat 3 views along h → (c, 3*h, w) = (4, 72, 40)
        his_joint.append(joint_pos_pred[pred_step - 1][None, :])
        his_eef.append(cartesian_pose[pred_step - 1][None, :])
        his_cond.append(
            torch.cat([predict_latents[v_i][pred_step - 1] for v_i in range(predict_latents.shape[0])], dim=1).unsqueeze(0)
        )
        # Update observation buffer with predicted frames (3 views)
        video_dict_pred = [pred_videos[view_i] for view_i in range(pred_videos.shape[0])]

        # Collect frames (view 0 = exterior_1_left for metrics)
        if i == interact_num - 1:
            video_to_save_pred.append(pred_videos[0])
            video_to_save_gt.append(true_videos[0])
        else:
            video_to_save_pred.append(pred_videos[0][:pred_step - 1])
            video_to_save_gt.append(true_videos[0][:pred_step - 1])

        if (i + 1) % 10 == 0:
            print(f"    Step {i+1}/{interact_num}")

    if not video_to_save_pred:
        return None

    concat_pred = np.concatenate(video_to_save_pred, axis=0)
    concat_gt = np.concatenate(video_to_save_gt, axis=0)
    min_len = min(len(concat_pred), len(concat_gt))
    concat_pred = concat_pred[:min_len]
    concat_gt = concat_gt[:min_len]

    # Save output (predictions)
    ep_save_dir = Path(save_dir) / f"episode_{episode_id}"
    ep_save_dir.mkdir(parents=True, exist_ok=True)

    mediapy.write_video(str(ep_save_dir / "full_pred.mp4"), concat_pred, fps=5)
    merged = np.concatenate([concat_gt, concat_pred], axis=2)
    mediapy.write_video(str(ep_save_dir / "full_merged.mp4"), merged, fps=5)

    # Save input (GT video, actions, metadata)
    if input_save_dir is not None:
        ep_input_dir = Path(input_save_dir) / f"episode_{episode_id}"
        ep_input_dir.mkdir(parents=True, exist_ok=True)

        mediapy.write_video(str(ep_input_dir / "full_gt.mp4"), concat_gt, fps=5)

        for vid_idx, vid_info in enumerate(anno["videos"]):
            src_video = Path(args.val_dataset_dir) / vid_info["video_path"]
            dst_video = ep_input_dir / f"view_{vid_idx}.mp4"
            if src_video.exists() and not dst_video.exists():
                import shutil
                shutil.copy2(str(src_video), str(dst_video))

        input_meta = {
            "episode_id": episode_id,
            "instruction": instruction,
            "num_frames": len(concat_gt),
            "episode_length_original": episode_length,
            "fps": 5,
            "interact_num": interact_num,
            "pred_step": pred_step,
            "mode": "policy_predicted",
            "policy_type": args.policy_type,
            "states": anno["states"][:len(concat_gt)],
            "joints": anno["joints"][:len(concat_gt)],
        }
        with open(ep_input_dir / "metadata.json", "w") as f:
            json.dump(input_meta, f, indent=2)

    # Metrics
    metrics = compute_metrics(concat_pred, concat_gt)
    metrics.update({
        "episode_id": episode_id,
        "instruction": instruction,
        "num_frames_pred": len(concat_pred),
        "num_frames_gt": episode_length,
        "interact_num": interact_num,
        "mode": "policy_predicted",
        "policy_type": args.policy_type,
    })
    with open(ep_save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    parser = ArgumentParser()
    parser.add_argument("--svd_model_path", type=str, required=True)
    parser.add_argument("--clip_model_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--pi_ckpt", type=str, required=True)
    parser.add_argument("--policy_type", type=str, default="pi05",
                        choices=["pi05", "pi0", "pi0fast"])
    parser.add_argument("--subset", choices=["makovian", "non_makovian"], required=True)
    parser.add_argument("--dataset_example_dir", type=str, default=DATASET_EXAMPLE_BASE)
    parser.add_argument("--dataset_meta_info_path", type=str,
                        default=os.path.join(PROJECT_ROOT, "models", "Ctrl-World", "dataset_meta_info"))
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--input_save_dir", type=str, default=None)
    parser.add_argument("--num_episodes", type=int, default=None)
    args = parser.parse_args()

    subset_name = f"single_arm_multiview_{args.subset}"
    val_dataset_dir = os.path.join(args.dataset_example_dir, subset_name)

    if args.save_dir is None:
        args.save_dir = os.path.join(
            PROJECT_ROOT, "sampling_dataset", "dense", "single_arm",
            "output", "multiview", "ctrlworld", args.subset
        )

    if args.input_save_dir is None:
        args.input_save_dir = os.path.join(
            PROJECT_ROOT, "sampling_dataset", "dense", "single_arm",
            "input", "multiview", "ctrlworld", args.subset
        )

    # Build model args
    from config import wm_args
    model_args = wm_args(task_type="replay")
    model_args.svd_model_path = args.svd_model_path
    model_args.clip_model_path = args.clip_model_path
    model_args.ckpt_path = args.ckpt_path
    model_args.val_model_path = args.ckpt_path
    model_args.val_dataset_dir = val_dataset_dir
    model_args.dataset_meta_info_path = args.dataset_meta_info_path
    model_args.data_stat_path = os.path.join(args.dataset_meta_info_path, "droid_subset", "stat.json")
    model_args.pi_ckpt = args.pi_ckpt
    model_args.policy_type = args.policy_type
    model_args.__post_init__()
    model_args.val_dataset_dir = val_dataset_dir

    # Import and create the policy-enabled agent from rollout_interact_pi
    from rollout_interact_pi import agent as PolicyAgent

    # Patch torch.load to handle CUDA map_location gracefully
    _original_torch_load = torch.load
    def _patched_torch_load(f, *a, **kw):
        if "map_location" not in kw:
            kw["map_location"] = "cuda" if torch.cuda.is_available() else "cpu"
        return _original_torch_load(f, *a, **kw)
    torch.load = _patched_torch_load

    Agent = PolicyAgent(model_args)
    torch.load = _original_torch_load

    # Find all episodes
    anno_dir = Path(val_dataset_dir) / "annotation" / "val"
    episode_ids = sorted([f.stem for f in anno_dir.glob("*.json")])
    if args.num_episodes:
        episode_ids = episode_ids[:args.num_episodes]

    print(f"\n{'='*60}")
    print(f"Ctrl-World Policy-Conditioned Inference")
    print(f"Subset: {args.subset}")
    print(f"Policy: {args.policy_type}")
    print(f"Dataset dir: {val_dataset_dir}")
    print(f"Output dir: {args.save_dir}")
    print(f"Input dir: {args.input_save_dir}")
    print(f"Episodes: {len(episode_ids)}")
    print(f"pred_step={model_args.pred_step}, num_history={model_args.num_history}")
    print(f"{'='*60}\n")

    all_metrics = []
    for ep_idx, episode_id in enumerate(episode_ids):
        ep_save_path = Path(args.save_dir) / f"episode_{episode_id}" / "full_pred.mp4"
        if ep_save_path.exists():
            print(f"[{ep_idx+1}/{len(episode_ids)}] Episode {episode_id} already done, skipping")
            continue

        print(f"[{ep_idx+1}/{len(episode_ids)}] Episode {episode_id}")
        try:
            metrics = run_episode_with_policy(
                Agent, episode_id, args.save_dir, input_save_dir=args.input_save_dir
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            metrics = None

        if metrics:
            all_metrics.append(metrics)
            print(f"  -> {metrics['num_frames_pred']} frames, "
                  f"PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}, LPIPS={metrics['lpips']:.4f}")

    # Save summary
    if all_metrics:
        summary = {
            "mean_psnr": sum(m["psnr"] for m in all_metrics) / len(all_metrics),
            "mean_ssim": sum(m["ssim"] for m in all_metrics) / len(all_metrics),
            "mean_lpips": sum(m["lpips"] for m in all_metrics) / len(all_metrics),
            "num_episodes": len(all_metrics),
            "subset": args.subset,
            "mode": "policy_predicted",
            "policy_type": args.policy_type,
        }
        summary_path = Path(args.save_dir) / "all_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n=== Summary ({len(all_metrics)} episodes) ===")
        print(f"PSNR: {summary['mean_psnr']:.3f}")
        print(f"SSIM: {summary['mean_ssim']:.4f}")
        print(f"LPIPS: {summary['mean_lpips']:.4f}")


if __name__ == "__main__":
    main()
