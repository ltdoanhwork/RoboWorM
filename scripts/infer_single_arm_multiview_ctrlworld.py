"""
Full-episode GT-anchored rollout using Ctrl-World on single_arm/multiview dataset.

For each episode:
  1. Compute interact_num based on episode length to match original duration.
  2. Use GT actions (replay mode) — no policy model needed.
  3. Generate video frames autoregressively using Ctrl-World.
  4. Save predicted video + metrics for benchmark evaluation.

Requires: prepare_ctrlworld_single_arm_multiview.py to be run first.
"""

import sys
import os
import importlib
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
sys.path.insert(0, ctrl_world_dir)

from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.ctrl_world import CrtlWorld
from decord import VideoReader, cpu
from accelerate import Accelerator


DATASET_EXAMPLE_BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "Ctrl-World", "dataset_example"
)


class CtrlWorldAgent:
    def __init__(self, args):
        args.val_model_path = args.ckpt_path
        self.args = args
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.dtype = args.dtype

        self.model = CrtlWorld(args)
        self.model.load_state_dict(torch.load(args.val_model_path))
        self.model.to(self.device).to(self.dtype)
        self.model.eval()
        print("Ctrl-World model loaded")

        with open(args.data_stat_path, "r") as f:
            data_stat = json.load(f)
            self.state_p01 = np.array(data_stat["state_01"])[None, :]
            self.state_p99 = np.array(data_stat["state_99"])[None, :]

    def normalize_bound(self, data, data_min, data_max, clip_min=-1, clip_max=1, eps=1e-8):
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        return np.clip(ndata, clip_min, clip_max)

    def get_traj_info(self, episode_id, start_idx=0, steps=8):
        val_dataset_dir = self.args.val_dataset_dir
        annotation_path = f"{val_dataset_dir}/annotation/val/{episode_id}.json"
        with open(annotation_path) as f:
            anno = json.load(f)
            length = anno["video_length"]

        frames_ids = np.arange(start_idx, start_idx + steps)
        max_ids = np.ones_like(frames_ids) * (length - 1)
        frames_ids = np.min([frames_ids, max_ids], axis=0).astype(int)

        instruction = anno["texts"][0]
        car_action = np.array(anno["states"])
        car_action = car_action[frames_ids]
        joint_pos = np.array(anno["joints"])
        joint_pos = joint_pos[frames_ids]

        video_dict = []
        video_latent = []
        for vid_info in anno["videos"]:
            video_path = f"{val_dataset_dir}/{vid_info['video_path']}"
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
            actual_video_len = len(vr)
            if length > actual_video_len:
                length = actual_video_len
                frames_ids = np.clip(frames_ids, 0, length - 1)
            try:
                true_video = vr.get_batch(range(length)).asnumpy()
            except:
                true_video = vr.get_batch(range(length)).numpy()
            true_video = true_video[frames_ids]
            video_dict.append(true_video)

            device = self.device
            true_video_t = torch.from_numpy(true_video).to(self.dtype).to(device)
            x = true_video_t.permute(0, 3, 1, 2) / 255.0 * 2 - 1
            vae = self.model.pipeline.vae
            with torch.no_grad():
                latents = []
                for i in range(0, len(x), 32):
                    batch = x[i:i + 32]
                    latent = vae.encode(batch).latent_dist.sample().mul_(vae.config.scaling_factor)
                    latents.append(latent)
                x = torch.cat(latents, dim=0)
            video_latent.append(x)

        return car_action, joint_pos, video_dict, video_latent, instruction

    def forward_wm(self, action_cond, video_latent_true, video_latent_cond, his_cond=None, text=None):
        args = self.args
        image_cond = video_latent_cond

        action_cond = self.normalize_bound(action_cond, self.state_p01, self.state_p99)
        action_cond = torch.tensor(action_cond).unsqueeze(0).to(self.device).to(self.dtype)

        with torch.no_grad():
            if text is not None:
                text_token = self.model.action_encoder(
                    action_cond, text, self.model.tokenizer, self.model.text_encoder
                )
            else:
                text_token = self.model.action_encoder(action_cond)
            pipeline = self.model.pipeline

            _, latents = CtrlWorldDiffusionPipeline.__call__(
                pipeline,
                image=image_cond,
                text=text_token,
                width=args.width,
                height=int(args.height * 3),
                num_frames=args.num_frames,
                history=his_cond,
                num_inference_steps=args.num_inference_steps,
                decode_chunk_size=args.decode_chunk_size,
                max_guidance_scale=args.guidance_scale,
                fps=args.fps,
                motion_bucket_id=args.motion_bucket_id,
                mask=None,
                output_type="latent",
                return_dict=False,
                frame_level_cond=True,
            )
        latents = einops.rearrange(latents, "b f c (m h) (n w) -> (b m n) f c h w", m=3, n=1)

        # Decode GT
        true_video = torch.stack(video_latent_true, dim=0)
        decoded_video = []
        bsz, frame_num = true_video.shape[:2]
        true_video_flat = true_video.flatten(0, 1)
        for i in range(0, true_video_flat.shape[0], args.decode_chunk_size):
            chunk = true_video_flat[i:i + args.decode_chunk_size] / pipeline.vae.config.scaling_factor
            decoded_video.append(pipeline.vae.decode(chunk, num_frames=chunk.shape[0]).sample)
        true_video_dec = torch.cat(decoded_video, dim=0)
        true_video_dec = true_video_dec.reshape(bsz, frame_num, *true_video_dec.shape[1:])
        true_video_dec = ((true_video_dec / 2.0 + 0.5).clamp(0, 1) * 255)
        true_video_dec = true_video_dec.detach().to(torch.float32).cpu().numpy().transpose(0, 1, 3, 4, 2).astype(np.uint8)

        # Decode predicted
        decoded_video = []
        bsz, frame_num = latents.shape[:2]
        x = latents.flatten(0, 1)
        for i in range(0, x.shape[0], args.decode_chunk_size):
            chunk = x[i:i + args.decode_chunk_size] / pipeline.vae.config.scaling_factor
            decoded_video.append(pipeline.vae.decode(chunk, num_frames=chunk.shape[0]).sample)
        videos = torch.cat(decoded_video, dim=0)
        videos = videos.reshape(bsz, frame_num, *videos.shape[1:])
        videos = ((videos / 2.0 + 0.5).clamp(0, 1) * 255)
        videos = videos.detach().to(torch.float32).cpu().numpy().transpose(0, 1, 3, 4, 2).astype(np.uint8)

        return true_video_dec, videos, latents


def compute_metrics(pred_frames, gt_frames):
    """Compute PSNR, SSIM, LPIPS between pred and gt frame arrays."""
    x = torch.clamp(torch.from_numpy(pred_frames.copy()) / 255.0, 0, 1).permute(0, 3, 1, 2)
    y = torch.clamp(torch.from_numpy(gt_frames.copy()) / 255.0, 0, 1).permute(0, 3, 1, 2)
    psnr_val = piq.psnr(x, y).mean().item()
    ssim_val = piq.ssim(x, y).mean().item()
    lpips_val = piq.LPIPS()(x, y).mean().item()
    return {"psnr": psnr_val, "ssim": ssim_val, "lpips": lpips_val}


def run_episode(agent, episode_id, save_dir, input_save_dir=None):
    """Run full-episode replay generation for one episode."""
    args = agent.args
    pred_step = args.num_frames  # 5
    num_history = args.num_history  # 6

    # Read annotation to get episode length
    anno_path = f"{args.val_dataset_dir}/annotation/val/{episode_id}.json"
    with open(anno_path) as f:
        anno = json.load(f)
    episode_length = anno["video_length"]

    # Calculate interact_num to cover full episode
    # Each iteration uses frames[i*4 : i*4+5], so max end = (interact_num-1)*4+5 <= episode_length
    interact_num = (episode_length - 1) // (pred_step - 1)
    if interact_num < 1:
        print(f"  Episode {episode_id} too short ({episode_length} frames), skipping")
        return None

    total_steps_needed = (interact_num - 1) * (pred_step - 1) + pred_step

    # Get trajectory info
    eef_gt, joint_pos_gt, video_dict, video_latents, instruction = agent.get_traj_info(
        episode_id, start_idx=0, steps=min(total_steps_needed, episode_length)
    )

    # Initialize history buffers
    his_cond = []
    his_eef = []
    first_latent = torch.cat([v[0] for v in video_latents], dim=1).unsqueeze(0)
    for _ in range(num_history * 4):
        his_cond.append(first_latent)
        his_eef.append(eef_gt[0:1])

    video_to_save_pred = []
    video_to_save_gt = []
    history_idx = [0, 0, -8, -6, -4, -2]

    for i in range(interact_num):
        start_id = int(i * (pred_step - 1))
        end_id = start_id + pred_step

        if end_id > len(eef_gt):
            break

        video_latent_true = [v[start_id:end_id] for v in video_latents]
        cartesian_pose = eef_gt[start_id:end_id]

        # Prepare history
        his_pose = np.concatenate([his_eef[idx] for idx in history_idx], axis=0)
        action_cond = np.concatenate([his_pose, cartesian_pose], axis=0)
        his_cond_input = torch.cat([his_cond[idx] for idx in history_idx], dim=0).unsqueeze(0)
        current_latent = his_cond[-1]

        # Forward world model
        true_videos, pred_videos, predicted_latents = agent.forward_wm(
            action_cond, video_latent_true, current_latent,
            his_cond=his_cond_input,
            text=instruction if args.text_cond else None,
        )

        # Update history
        his_eef.append(cartesian_pose[pred_step - 1:pred_step])
        his_cond.append(
            torch.cat([v[pred_step - 1] for v in predicted_latents], dim=1).unsqueeze(0)
        )

        # Collect frames — all 3 views
        # true_videos/pred_videos shape: (3_views, frames, H, W, 3)
        if i == interact_num - 1:
            video_to_save_pred.append(pred_videos)
            video_to_save_gt.append(true_videos)
        else:
            video_to_save_pred.append(pred_videos[:, :pred_step - 1])
            video_to_save_gt.append(true_videos[:, :pred_step - 1])

        if (i + 1) % 10 == 0:
            print(f"    Step {i+1}/{interact_num}")

    if not video_to_save_pred:
        return None

    # Concatenate — shape: (3_views, total_frames, H, W, 3)
    concat_pred = np.concatenate(video_to_save_pred, axis=1)
    concat_gt = np.concatenate(video_to_save_gt, axis=1)
    num_views = concat_pred.shape[0]
    min_len = min(concat_pred.shape[1], concat_gt.shape[1])
    concat_pred = concat_pred[:, :min_len]
    concat_gt = concat_gt[:, :min_len]

    # Build all-views strips: (frames, H, 3*W, 3)
    pred_strip = np.concatenate([concat_pred[v] for v in range(num_views)], axis=2)
    gt_strip = np.concatenate([concat_gt[v] for v in range(num_views)], axis=2)
    # Full pred: GT top + Pred bottom, all views side by side
    full_pred = np.concatenate([gt_strip, pred_strip], axis=1)

    # Save output (predictions)
    ep_save_dir = Path(save_dir) / f"episode_{episode_id}"
    ep_save_dir.mkdir(parents=True, exist_ok=True)

    mediapy.write_video(str(ep_save_dir / "full_pred.mp4"), full_pred, fps=5)
    mediapy.write_video(str(ep_save_dir / "pred_all_views.mp4"), pred_strip, fps=5)
    mediapy.write_video(str(ep_save_dir / "gt_all_views.mp4"), gt_strip, fps=5)

    # Save input (GT video, actions, metadata)
    if input_save_dir is not None:
        ep_input_dir = Path(input_save_dir) / f"episode_{episode_id}"
        ep_input_dir.mkdir(parents=True, exist_ok=True)

        mediapy.write_video(str(ep_input_dir / "full_gt.mp4"), gt_strip, fps=5)

        # Save all 3 views GT videos from annotation source
        anno_path = f"{args.val_dataset_dir}/annotation/val/{episode_id}.json"
        with open(anno_path) as f:
            anno_data = json.load(f)
        for vid_idx, vid_info in enumerate(anno_data["videos"]):
            src_video = Path(args.val_dataset_dir) / vid_info["video_path"]
            dst_video = ep_input_dir / f"view_{vid_idx}.mp4"
            if src_video.exists() and not dst_video.exists():
                import shutil
                shutil.copy2(str(src_video), str(dst_video))

        # Save actions and states
        input_meta = {
            "episode_id": episode_id,
            "instruction": instruction,
            "num_frames": min_len,
            "episode_length_original": episode_length,
            "fps": 5,
            "interact_num": interact_num,
            "pred_step": pred_step,
            "mode": "replay",
            "states": anno_data["states"][:min_len],
            "joints": anno_data["joints"][:min_len],
        }
        with open(ep_input_dir / "metadata.json", "w") as f:
            json.dump(input_meta, f, indent=2)

    # Metrics (per-view)
    view_names = ["exterior_1_left", "exterior_2_left", "wrist_left"]
    w_per_view = concat_pred.shape[3]
    per_view_metrics = {}
    for v_i in range(num_views):
        per_view_metrics[view_names[v_i]] = compute_metrics(concat_pred[v_i], concat_gt[v_i])
    avg_psnr = np.mean([m["psnr"] for m in per_view_metrics.values()])
    avg_ssim = np.mean([m["ssim"] for m in per_view_metrics.values()])
    avg_lpips = np.mean([m["lpips"] for m in per_view_metrics.values()])
    metrics = {"psnr": float(avg_psnr), "ssim": float(avg_ssim), "lpips": float(avg_lpips)}
    metrics["per_view"] = per_view_metrics
    metrics.update({
        "episode_id": episode_id,
        "instruction": instruction,
        "num_frames_pred": min_len,
        "num_frames_gt": episode_length,
        "interact_num": interact_num,
        "mode": "replay",
    })
    with open(ep_save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def _import_cache_module(backend, module_name):
    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return importlib.import_module(f"methods.cache_strategy.{backend}.{module_name}")


def _import_pruning_module(backend, module_name):
    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return importlib.import_module(f"methods.prunning.{backend}.{module_name}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--svd_model_path", type=str, required=True)
    parser.add_argument("--clip_model_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--subset", choices=["makovian", "non_makovian"], required=True)
    parser.add_argument("--dataset_example_dir", type=str, default=DATASET_EXAMPLE_BASE)
    parser.add_argument("--dataset_meta_info_path", type=str,
                        default="./models/Ctrl-World/dataset_meta_info")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--input_save_dir", type=str, default=None)
    parser.add_argument("--num_episodes", type=int, default=None)

    # Cache backend args (mutually exclusive)
    _import_cache_module("WorldCache", "config").add_worldcache_args(parser)
    _import_cache_module("DiCache", "config").add_dicache_args(parser)
    _import_cache_module("FasterCache", "config").add_fastercache_args(parser)

    # Pruning/sparse attention backend args
    _import_pruning_module("SiTo", "config").add_sito_args(parser)
    _import_pruning_module("importance_token_merge", "config").add_itm_args(parser)

    args = parser.parse_args()

    from methods.cache_strategy.ctrl_world_utils import validate_ctrl_world_backend_args
    validate_ctrl_world_backend_args(args)

    # Determine paths
    subset_name = f"single_arm_multiview_{args.subset}"
    val_dataset_dir = os.path.join(args.dataset_example_dir, subset_name)

    if args.save_dir is None:
        args.save_dir = (
            f"/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/"
            f"video_gen_physics/sampling_dataset/dense/single_arm/output/multiview/"
            f"ctrlworld/{args.subset}"
        )

    if args.input_save_dir is None:
        args.input_save_dir = (
            f"/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/"
            f"video_gen_physics/sampling_dataset/dense/single_arm/input/multiview/"
            f"ctrlworld/{args.subset}"
        )

    # Build model args from config
    sys.path.insert(0, ctrl_world_dir)
    from config import wm_args
    model_args = wm_args(task_type="replay")
    model_args.svd_model_path = args.svd_model_path
    model_args.clip_model_path = args.clip_model_path
    model_args.ckpt_path = args.ckpt_path
    model_args.val_model_path = args.ckpt_path
    model_args.val_dataset_dir = val_dataset_dir
    model_args.dataset_meta_info_path = args.dataset_meta_info_path
    model_args.data_stat_path = os.path.join(args.dataset_meta_info_path, "droid_subset", "stat.json")
    model_args.__post_init__()
    # Override val_dataset_dir after __post_init__ since it may reset
    model_args.val_dataset_dir = val_dataset_dir

    # Create agent
    agent = CtrlWorldAgent(model_args)

    # Enable cache backend on UNet (mutually exclusive, validated above)
    if getattr(args, "use_worldcache", False):
        adapter = _import_cache_module("WorldCache", "adapter")
        adapter.enable_worldcache(
            agent.model.unet,
            num_steps=model_args.num_inference_steps,
            rel_l1_thresh=args.worldcache_rel_l1_thresh,
            ret_ratio=args.worldcache_ret_ratio,
            probe_depth=args.worldcache_probe_depth,
            motion_sensitivity=args.worldcache_motion_sensitivity,
            hf_enabled=args.worldcache_hf_enabled,
            hf_thresh=args.worldcache_hf_thresh,
            saliency_enabled=args.worldcache_saliency_enabled,
            saliency_weight=args.worldcache_saliency_weight,
            osi_enabled=args.worldcache_osi_enabled,
            dynamic_decay=args.worldcache_dynamic_decay,
        )
        print(f"[Cache] WorldCache enabled: thresh={args.worldcache_rel_l1_thresh}, "
              f"ret_ratio={args.worldcache_ret_ratio}, probe_depth={args.worldcache_probe_depth}")
    if getattr(args, "use_dicache", False):
        adapter = _import_cache_module("DiCache", "adapter")
        adapter.enable_dicache(
            agent.model.unet,
            num_steps=model_args.num_inference_steps,
            rel_l1_thresh=args.dicache_rel_l1_thresh,
            ret_ratio=args.dicache_ret_ratio,
            probe_depth=args.dicache_probe_depth,
        )
        print(f"[Cache] DiCache enabled: thresh={args.dicache_rel_l1_thresh}, "
              f"ret_ratio={args.dicache_ret_ratio}, probe_depth={args.dicache_probe_depth}")
    if getattr(args, "use_fastercache", False):
        adapter = _import_cache_module("FasterCache", "adapter")
        adapter.enable_fastercache(
            agent.model.unet,
            start_step=args.fastercache_start_step,
            model_interval=args.fastercache_model_interval,
            block_interval=args.fastercache_block_interval,
            first_layers_fp=2,
        )
        print(f"[Cache] FasterCache enabled: start_step={args.fastercache_start_step}, "
              f"model_interval={args.fastercache_model_interval}, block_interval={args.fastercache_block_interval}")
    if getattr(args, "use_sito", False):
        adapter = _import_pruning_module("SiTo", "adapter")
        adapter.enable_sito(
            agent.model.unet,
            start_layer_idx=args.sito_start_layer_idx or 0,
            prune_ratio=args.sito_prune_ratio,
            patch_h=args.sito_patch_h,
            patch_w=args.sito_patch_w,
            noise_alpha=args.sito_noise_alpha,
            sim_beta=args.sito_sim_beta,
            max_downsample_ratio=args.sito_max_downsample_ratio,
        )
    if getattr(args, "use_itm", False):
        adapter = _import_pruning_module("importance_token_merge", "adapter")
        itm_state = {}
        adapter.enable_itm(
            agent.model.unet,
            itm_state=itm_state,
            start_layer_idx=args.itm_start_layer_idx or 0,
            compress_ratio=args.itm_compress_ratio,
            prune_from_step=args.itm_prune_from_step,
            merge_from_step=args.itm_merge_from_step,
            merge_attn=args.itm_merge_attn,
            merge_crossattn=args.itm_merge_crossattn,
            merge_mlp=args.itm_merge_mlp,
            max_downsample_ratio=args.itm_max_downsample_ratio,
        )
        if model_args.guidance_scale <= 1.0:
            model_args.guidance_scale = 2.0
            print(f"[ITM] guidance_scale overridden to {model_args.guidance_scale} (ITM requires > 1.0)")

    # Find all episodes
    anno_dir = Path(val_dataset_dir) / "annotation" / "val"
    episode_ids = sorted([f.stem for f in anno_dir.glob("*.json")])
    if args.num_episodes:
        episode_ids = episode_ids[:args.num_episodes]

    print(f"\nSubset: {args.subset}")
    print(f"Dataset dir: {val_dataset_dir}")
    print(f"Output dir: {args.save_dir}")
    print(f"Input dir: {args.input_save_dir}")
    print(f"Episodes to process: {len(episode_ids)}")
    print(f"pred_step={model_args.num_frames}, num_history={model_args.num_history}")

    all_metrics = []
    for ep_idx, episode_id in enumerate(episode_ids):
        ep_save_path = Path(args.save_dir) / f"episode_{episode_id}" / "full_pred.mp4"
        if ep_save_path.exists():
            print(f"[{ep_idx+1}/{len(episode_ids)}] Episode {episode_id} already done, skipping")
            continue

        print(f"[{ep_idx+1}/{len(episode_ids)}] Episode {episode_id}")
        metrics = run_episode(agent, episode_id, args.save_dir, input_save_dir=args.input_save_dir)
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
