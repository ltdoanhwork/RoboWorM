"""
Full-episode GT-anchored replay using a finetuned Ctrl-World on bimanual/multiview.

Reads from the SAME prepared training dataset dir (dataset_example/bimanual_multiview)
whose annotations already contain: 3-view resized videos, `states` (14-D qpos,
frame-aligned), `video_length`, `texts`. No separate inference prep needed.

For each episode:
  1. interact_num computed from episode length to cover the full clip.
  2. Replay GT 14-D qpos as action condition (no policy).
  3. Autoregressive generation with Ctrl-World.
  4. Save predicted video + PSNR/SSIM/LPIPS per episode.

Mirrors scripts/infer_single_arm_multiview_ctrlworld.py but for bimanual:
  - 3 views: cam_high, cam_left_wrist, cam_right_wrist
  - 14-D state, bimanual stat.json
  - down_sample consistent with training (native, since arrays are frame-aligned)
"""

import sys
import os
import json
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


class CtrlWorldAgent:
    def __init__(self, args):
        args.val_model_path = args.ckpt_path
        self.args = args
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.dtype = args.dtype

        self.model = CrtlWorld(args)
        state_dict = torch.load(args.val_model_path, map_location="cpu")
        model_sd = self.model.state_dict()
        filtered = {k: v for k, v in state_dict.items()
                    if k in model_sd and model_sd[k].shape == v.shape}
        dropped = [k for k in state_dict if k not in filtered]
        missing, unexpected = self.model.load_state_dict(filtered, strict=False)
        if dropped or missing:
            print(f"[load] dropped {len(dropped)} shape-mismatch tensors "
                  f"(e.g. {dropped[:2]}); missing {len(missing)} (reinit). "
                  f"A finetuned 14-D bimanual checkpoint should load with 0 dropped.")
        self.model.to(self.device).to(self.dtype)
        self.model.eval()
        print("Ctrl-World (bimanual) model loaded")

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
        if not os.path.exists(annotation_path):
            annotation_path = f"{val_dataset_dir}/annotation/train/{episode_id}.json"
        with open(annotation_path) as f:
            anno = json.load(f)
            length = anno["video_length"]

        frames_ids = np.arange(start_idx, start_idx + steps)
        max_ids = np.ones_like(frames_ids) * (length - 1)
        frames_ids = np.min([frames_ids, max_ids], axis=0).astype(int)

        instruction = anno["texts"][0]
        qpos_action = np.array(anno["states"])  # 14-D, frame-aligned
        qpos_action = qpos_action[frames_ids]

        video_latent = []
        video_dict = []
        for vid_info in anno["videos"]:
            video_path = f"{val_dataset_dir}/{vid_info['video_path']}"
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
            actual_video_len = len(vr)
            if length > actual_video_len:
                length = actual_video_len
                frames_ids = np.clip(frames_ids, 0, length - 1)
            try:
                true_video = vr.get_batch(range(length)).asnumpy()
            except Exception:
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

        return qpos_action, video_dict, video_latent, instruction

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
    x = torch.clamp(torch.from_numpy(pred_frames.copy()) / 255.0, 0, 1).permute(0, 3, 1, 2)
    y = torch.clamp(torch.from_numpy(gt_frames.copy()) / 255.0, 0, 1).permute(0, 3, 1, 2)
    psnr_val = piq.psnr(x, y).mean().item()
    ssim_val = piq.ssim(x, y).mean().item()
    lpips_val = piq.LPIPS()(x, y).mean().item()
    return {"psnr": psnr_val, "ssim": ssim_val, "lpips": lpips_val}


def find_annotation(val_dataset_dir, episode_id):
    for split in ["val", "train"]:
        p = f"{val_dataset_dir}/annotation/{split}/{episode_id}.json"
        if os.path.exists(p):
            return p, split
    return None, None


def run_episode(agent, episode_id, save_dir, input_save_dir=None):
    args = agent.args
    pred_step = args.num_frames  # 5
    num_history = args.num_history  # 6

    anno_path, split = find_annotation(args.val_dataset_dir, episode_id)
    if anno_path is None:
        print(f"  no annotation for {episode_id}")
        return None
    with open(anno_path) as f:
        anno = json.load(f)
    episode_length = anno["video_length"]

    interact_num = (episode_length - 1) // (pred_step - 1)
    if interact_num < 1:
        print(f"  Episode {episode_id} too short ({episode_length} frames), skipping")
        return None

    total_steps_needed = (interact_num - 1) * (pred_step - 1) + pred_step

    eef_gt, video_dict, video_latents, instruction = agent.get_traj_info(
        episode_id, start_idx=0, steps=min(total_steps_needed, episode_length)
    )

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
        qpos_pose = eef_gt[start_id:end_id]

        his_pose = np.concatenate([his_eef[idx] for idx in history_idx], axis=0)
        action_cond = np.concatenate([his_pose, qpos_pose], axis=0)
        his_cond_input = torch.cat([his_cond[idx] for idx in history_idx], dim=0).unsqueeze(0)
        current_latent = his_cond[-1]

        true_videos, pred_videos, predicted_latents = agent.forward_wm(
            action_cond, video_latent_true, current_latent,
            his_cond=his_cond_input,
            text=instruction if args.text_cond else None,
        )

        his_eef.append(qpos_pose[pred_step - 1:pred_step])
        his_cond.append(
            torch.cat([v[pred_step - 1] for v in predicted_latents], dim=1).unsqueeze(0)
        )

        if i == interact_num - 1:
            video_to_save_pred.append(pred_videos)
            video_to_save_gt.append(true_videos)
        else:
            video_to_save_pred.append(pred_videos[:, :pred_step - 1])
            video_to_save_gt.append(true_videos[:, :pred_step - 1])

        if (i + 1) % 20 == 0:
            print(f"    Step {i+1}/{interact_num}")

    if not video_to_save_pred:
        return None

    concat_pred = np.concatenate(video_to_save_pred, axis=1)
    concat_gt = np.concatenate(video_to_save_gt, axis=1)
    num_views = concat_pred.shape[0]
    min_len = min(concat_pred.shape[1], concat_gt.shape[1])
    concat_pred = concat_pred[:, :min_len]
    concat_gt = concat_gt[:, :min_len]

    pred_strip = np.concatenate([concat_pred[v] for v in range(num_views)], axis=2)
    gt_strip = np.concatenate([concat_gt[v] for v in range(num_views)], axis=2)
    full_pred = np.concatenate([gt_strip, pred_strip], axis=1)

    out_fps = max(1, int(round(30 / args.infer_down_sample)))
    ep_save_dir = Path(save_dir) / f"episode_{episode_id}"
    ep_save_dir.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(ep_save_dir / "full_pred.mp4"), full_pred, fps=out_fps)
    mediapy.write_video(str(ep_save_dir / "pred_all_views.mp4"), pred_strip, fps=out_fps)
    mediapy.write_video(str(ep_save_dir / "gt_all_views.mp4"), gt_strip, fps=out_fps)

    if input_save_dir is not None:
        ep_input_dir = Path(input_save_dir) / f"episode_{episode_id}"
        ep_input_dir.mkdir(parents=True, exist_ok=True)
        mediapy.write_video(str(ep_input_dir / "full_gt.mp4"), gt_strip, fps=out_fps)
        for vid_idx, vid_info in enumerate(anno["videos"]):
            src_video = Path(args.val_dataset_dir) / vid_info["video_path"]
            dst_video = ep_input_dir / f"view_{vid_idx}.mp4"
            if src_video.exists() and not dst_video.exists():
                import shutil
                shutil.copy2(str(src_video), str(dst_video))
        input_meta = {
            "episode_id": episode_id,
            "instruction": instruction,
            "num_frames": min_len,
            "episode_length_original": episode_length,
            "fps": out_fps,
            "interact_num": interact_num,
            "pred_step": pred_step,
            "mode": "replay",
            "states": anno["states"][:min_len],
        }
        with open(ep_input_dir / "metadata.json", "w") as f:
            json.dump(input_meta, f, indent=2)

    view_names = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    per_view_metrics = {}
    for v_i in range(num_views):
        per_view_metrics[view_names[v_i]] = compute_metrics(concat_pred[v_i], concat_gt[v_i])
    metrics = {
        "psnr": float(np.mean([m["psnr"] for m in per_view_metrics.values()])),
        "ssim": float(np.mean([m["ssim"] for m in per_view_metrics.values()])),
        "lpips": float(np.mean([m["lpips"] for m in per_view_metrics.values()])),
        "per_view": per_view_metrics,
        "episode_id": episode_id,
        "instruction": instruction,
        "num_frames_pred": min_len,
        "num_frames_gt": episode_length,
        "interact_num": interact_num,
        "mode": "replay",
    }
    with open(ep_save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main():
    parser = ArgumentParser()
    parser.add_argument("--svd_model_path", type=str, required=True)
    parser.add_argument("--clip_model_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Prepared bimanual_multiview dir (has annotation/{train,val}, videos/).")
    parser.add_argument("--dataset_meta_info_path", type=str,
                        default="./models/Ctrl-World/dataset_meta_info")
    parser.add_argument("--dataset_name", type=str, default="bimanual_multiview")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--input_save_dir", type=str, default=None)
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None,
                        help="CFG scale for action conditioning. >1.0 amplifies "
                             "action-following (config default 1.0 = CFG off).")
    parser.add_argument("--infer_down_sample", type=int, default=1,
                        help="Must match the down_sample used at prep/train (for output fps).")
    parser.add_argument("--split", choices=["val", "train", "all"], default="all")
    args = parser.parse_args()

    sys.path.insert(0, ctrl_world_dir)
    from config_bimanual import wm_args_bimanual
    model_args = wm_args_bimanual()
    model_args.svd_model_path = args.svd_model_path
    model_args.clip_model_path = args.clip_model_path
    model_args.ckpt_path = args.ckpt_path
    model_args.val_model_path = args.ckpt_path
    model_args.val_dataset_dir = args.dataset_dir
    model_args.dataset_meta_info_path = args.dataset_meta_info_path
    model_args.data_stat_path = os.path.join(args.dataset_meta_info_path, args.dataset_name, "stat.json")
    model_args.infer_down_sample = args.infer_down_sample
    if args.num_inference_steps is not None:
        model_args.num_inference_steps = args.num_inference_steps
    if args.guidance_scale is not None:
        model_args.guidance_scale = args.guidance_scale
        print(f"[override] guidance_scale = {args.guidance_scale}")

    agent = CtrlWorldAgent(model_args)

    splits = ["val", "train"] if args.split == "all" else [args.split]
    episode_ids = []
    for split in splits:
        anno_dir = Path(args.dataset_dir) / "annotation" / split
        if anno_dir.exists():
            episode_ids += sorted([f.stem for f in anno_dir.glob("*.json")])
    episode_ids = sorted(set(episode_ids))
    if args.num_episodes:
        episode_ids = episode_ids[:args.num_episodes]

    print(f"Dataset dir: {args.dataset_dir}")
    print(f"Output dir: {args.save_dir}")
    print(f"Episodes to process: {len(episode_ids)}")

    all_metrics = []
    for ep_idx, episode_id in enumerate(episode_ids):
        ep_save_path = Path(args.save_dir) / f"episode_{episode_id}" / "full_pred.mp4"
        if ep_save_path.exists():
            print(f"[{ep_idx+1}/{len(episode_ids)}] {episode_id} done, skipping")
            continue
        print(f"[{ep_idx+1}/{len(episode_ids)}] Episode {episode_id}")
        metrics = run_episode(agent, episode_id, args.save_dir, input_save_dir=args.input_save_dir)
        if metrics:
            all_metrics.append(metrics)
            print(f"  -> {metrics['num_frames_pred']} frames, "
                  f"PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}, LPIPS={metrics['lpips']:.4f}")

    if all_metrics:
        summary = {
            "mean_psnr": sum(m["psnr"] for m in all_metrics) / len(all_metrics),
            "mean_ssim": sum(m["ssim"] for m in all_metrics) / len(all_metrics),
            "mean_lpips": sum(m["lpips"] for m in all_metrics) / len(all_metrics),
            "num_episodes": len(all_metrics),
        }
        summary_path = Path(args.save_dir) / "all_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n=== Summary ({len(all_metrics)} episodes) ===")
        print(f"PSNR: {summary['mean_psnr']:.3f}  SSIM: {summary['mean_ssim']:.4f}  LPIPS: {summary['mean_lpips']:.4f}")


if __name__ == "__main__":
    main()
