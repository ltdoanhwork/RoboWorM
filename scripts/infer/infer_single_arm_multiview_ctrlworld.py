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
import time
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import torch
import einops
import mediapy
import piq

ctrl_world_dir = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")),
    "models", "Ctrl-World"
)
sys.path.insert(0, ctrl_world_dir)

from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.ctrl_world import CrtlWorld
from decord import VideoReader, cpu
from accelerate import Accelerator


DATASET_EXAMPLE_BASE = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")),
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

    _ep_chunks = 0
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
        _chunk_t0 = time.time()
        true_videos, pred_videos, predicted_latents = agent.forward_wm(
            action_cond, video_latent_true, current_latent,
            his_cond=his_cond_input,
            text=instruction if args.text_cond else None,
        )
        _chunk_dt = time.time() - _chunk_t0
        _CHUNK_TIMES.append(_chunk_dt)
        print(f"  [ChunkTiming] chunk {len(_CHUNK_TIMES)}: {_chunk_dt:.3f}s", flush=True)
        _cap = _max_chunks()
        _ep_chunks += 1
        if _cap and _ep_chunks >= _cap:
            print(f"  [ChunkTiming] stopping after {_cap} chunk(s) (cap)", flush=True)
            break

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
    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return importlib.import_module(f"methods.cache_strategy.{backend}.{module_name}")


def _import_pruning_module(backend, module_name):
    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return importlib.import_module(f"methods.prunning.{backend}.{module_name}")


def _install_attn_profiler(unet):
    """Env-gated (CW_ATTN_PROFILE=1) per-attention-layer profiler.

    Wraps every attention processor's __call__ to record, per (layer, T_q):
    self/cross, CUDA-synced ms/call, and % of attention. Also times the whole
    UNet.forward so we get attention-as-fraction-of-UNet. This is the ground
    truth for "does the sparse backend actually fire on the big spatial layer"
    and "where does UNet time go". No-op unless CW_ATTN_PROFILE=1.
    """
    if os.environ.get("CW_ATTN_PROFILE", "0") != "1":
        return
    import atexit as _atexit
    from collections import defaultdict as _dd

    stats = _dd(lambda: {"calls": 0, "ms": 0.0, "t_q": 0, "t_kv": 0, "self": False})

    class _ProfWrap:
        def __init__(self, name, inner):
            self._name = name
            self._inner = inner

        def __getattr__(self, item):
            return getattr(self._inner, item)

        def __call__(self, attn, hidden_states, encoder_hidden_states=None, *a, **kw):
            t_q = hidden_states.shape[1]
            kv = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
            t_kv = kv.shape[1]
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            out = self._inner(attn, hidden_states, encoder_hidden_states, *a, **kw)
            ev1.record()
            torch.cuda.synchronize()
            s = stats[(self._name, t_q)]
            s["calls"] += 1
            s["ms"] += ev0.elapsed_time(ev1)
            s["t_q"] = t_q
            s["t_kv"] = t_kv
            s["self"] = (t_q == t_kv)
            return out

    n_wrapped = 0
    for _mname, _mod in unet.named_modules():
        if hasattr(_mod, "processor") and hasattr(_mod, "to_q"):
            _mod.processor = _ProfWrap(_mname, _mod.processor)
            n_wrapped += 1

    def _dump():
        rows = sorted(stats.items(), key=lambda kv: -kv[1]["ms"])
        total = sum(v["ms"] for _, v in rows)
        print("\n===== [CW_ATTN_PROFILE] per-(layer,seqlen) attention time =====")
        print(f"{'layer':<46}{'T_q':>7}{'T_kv':>7}{'self':>6}{'calls':>7}"
              f"{'ms_tot':>10}{'ms/call':>9}{'%attn':>7}")
        by_seq = _dd(lambda: {"ms": 0.0, "calls": 0})
        for (name, _tq), v in rows:
            by_seq[v["t_q"]]["ms"] += v["ms"]
            by_seq[v["t_q"]]["calls"] += v["calls"]
            short = name if len(name) <= 44 else "..." + name[-41:]
            print(f"{short:<46}{v['t_q']:>7}{v['t_kv']:>7}{str(v['self']):>6}"
                  f"{v['calls']:>7}{v['ms']:>10.1f}{v['ms']/max(v['calls'],1):>9.3f}"
                  f"{100*v['ms']/max(total,1e-9):>7.1f}")
        print(f"----- attention time grouped by T_q (total {total:.1f} ms) -----")
        for tq, v in sorted(by_seq.items(), key=lambda kv: -kv[1]["ms"]):
            print(f"  T_q={tq:>6}  calls={v['calls']:>6}  ms={v['ms']:>9.1f}"
                  f"  ({100*v['ms']/max(total,1e-9):>5.1f}% of attn)")
        print("================================================================\n")

    _atexit.register(_dump)
    print(f"[CW_ATTN_PROFILE] attention profiler installed on {n_wrapped} attn modules.")

    _fwd = {"ms": 0.0, "calls": 0}
    _orig_fwd = unet.forward

    def _timed_fwd(*a, **kw):
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        out = _orig_fwd(*a, **kw)
        ev1.record()
        torch.cuda.synchronize()
        _fwd["ms"] += ev0.elapsed_time(ev1)
        _fwd["calls"] += 1
        return out

    unet.forward = _timed_fwd

    def _dump_fwd():
        attn_ms = sum(v["ms"] for v in stats.values())
        print(f"[CW_ATTN_PROFILE] UNet.forward total {_fwd['ms']:.1f} ms over "
              f"{_fwd['calls']} calls; attention = {attn_ms:.1f} ms "
              f"({100*attn_ms/max(_fwd['ms'],1e-9):.1f}% of UNet fwd). "
              f"Non-attention = {100*(1-attn_ms/max(_fwd['ms'],1e-9)):.1f}%.")

    _atexit.register(_dump_fwd)



def _persist_wall_time(save_dir, episode_id, metrics):
    """Re-dump metrics.json so it also carries `wall_time_s`.

    run_episode() writes metrics.json before returning, and the caller only knows
    the elapsed time afterwards, so without this the on-disk per-episode metrics
    have no timing at all. That is the reason no Ctrl-World cell could have its
    speedup computed from sampling_dataset and every timing had to be re-measured.
    """
    try:
        p = Path(save_dir) / f"episode_{episode_id}" / "metrics.json"
        if p.is_file():
            with open(p, "w") as f:
                json.dump(metrics, f, indent=2)
    except Exception as exc:  # never let bookkeeping kill a finished episode
        print(f"  [warn] could not persist wall_time_s for {episode_id}: {exc}")


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
    parser.add_argument("--num_inference_steps", type=int, default=None,
                        help="Override number of denoising steps (default: use model config, typically 50).")

    # Cache backend args (mutually exclusive)
    _import_cache_module("WorldCache", "config").add_worldcache_args(parser)
    _import_cache_module("DiCache", "config").add_dicache_args(parser)
    _import_cache_module("FasterCache", "config").add_fastercache_args(parser)

    # Pruning/sparse attention backend args
    _import_pruning_module("SiTo", "config").add_sito_args(parser)
    _import_pruning_module("importance_token_merge", "config").add_itm_args(parser)

    # Sparse-attention backend args (PISA / SVG / Radial — SVD-UNet flavored)
    importlib.import_module("methods.sparse_attention.pisa.config").add_pisa_args(parser)
    importlib.import_module("methods.sparse_attention.svg.config").add_svg_args(parser)
    importlib.import_module("methods.sparse_attention.radial.config").add_radial_args(parser)

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

    if args.num_inference_steps is not None:
        model_args.num_inference_steps = args.num_inference_steps
        print(f"[Override] num_inference_steps = {args.num_inference_steps}")

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
        if getattr(args, "sito_spatiotemporal_hold", False):
            st_hold = _import_pruning_module("SiTo", "spatiotemporal_hold")
            st_hold.enable_spatiotemporal_hold(
                agent.model.unet,
                keep_ratio=args.sito_st_keep_ratio,
                max_downsample_ratio=args.sito_max_downsample_ratio,
                recompute_every=(args.sito_plan_recompute_every or 999999),
            )
        else:
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
                plan_recompute_every=getattr(args, "sito_plan_recompute_every", 0),
            )
    if getattr(args, "use_itm", False):
        itm_state = {}
        if getattr(args, "itm_spatiotemporal_hold", False):
            st_hold = _import_pruning_module("SiTo", "spatiotemporal_hold")
            st_hold.enable_spatiotemporal_hold(
                agent.model.unet,
                keep_ratio=args.itm_st_keep_ratio,
                max_downsample_ratio=args.itm_max_downsample_ratio,
                recompute_every=(args.itm_plan_recompute_every or 999999),
                similarity_recover=True,
            )
        elif getattr(args, "itm_block_hold", False):
            block_hold = _import_pruning_module("importance_token_merge", "block_hold")
            itm_state["prune_from_step"] = args.itm_prune_from_step
            itm_state["merge_from_step"] = args.itm_merge_from_step
            block_hold.enable_itm_block_hold(
                agent.model.unet,
                itm_state=itm_state,
                start_layer_idx=args.itm_start_layer_idx or 0,
                compress_ratio=args.itm_compress_ratio,
                max_downsample_ratio=args.itm_max_downsample_ratio,
                self_importance=True,
                plan_recompute_every=getattr(args, "itm_plan_recompute_every", 0),
            )
            # Block-hold with self-importance derives token scores from the
            # hidden states, so it does NOT need classifier-free guidance and
            # can stay at guidance_scale=1 (single batch) — this is what keeps
            # it faster than dense instead of paying the 2x CFG cost.
        else:
            adapter = _import_pruning_module("importance_token_merge", "adapter")
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

    # ---- Sparse-attention backends (PISA / SVG / Radial) --------------------
    # 3-view strip stacked as ONE canvas: 576x320 px -> 72x40 latent (T=2880).
    # Same geometry as humanoid 3-view in infer_bimanual_multiview_ctrlworld.py
    # (grid_rows=3, grid_cols=1, treat stacked canvas as num_views=1).
    unet = agent.model.unet
    _h_per_view = 24 * 3
    _w = 40
    _num_views = 1

    if getattr(args, "use_pisa", False):
        importlib.import_module("methods.sparse_attention.pisa.adapter").enable_pisa(
            unet,
            density=args.pisa_density,
            block_size=args.pisa_block_size,
            start_layer_idx=args.pisa_start_layer_idx,
        )
        print(f"[Sparse] PISA enabled: density={args.pisa_density}, "
              f"block_size={args.pisa_block_size}, start_layer={args.pisa_start_layer_idx}")
    if getattr(args, "use_svg", False):
        importlib.import_module("methods.sparse_attention.svg.adapter").enable_svg(
            unet,
            pattern=args.svg_pattern,
            num_views=_num_views,
            H_per_view=_h_per_view,
            W=_w,
            sparsity=args.svg_sparsity,
            num_sampled_rows=args.svg_num_sampled_rows,
            first_layers_fp=args.svg_first_layers_fp,
            num_q_centroids=args.svg_num_q_centroids,
            num_k_centroids=args.svg_num_k_centroids,
            top_p_kmeans=args.svg_top_p_kmeans,
            min_kc_ratio=getattr(args, "svg_min_kc_ratio", 0.0),
            kmeans_iter_init=args.svg_kmeans_iter_init,
            kmeans_iter_step=args.svg_kmeans_iter_step,
        )
        print(f"[Sparse] SVG {args.svg_pattern} enabled: sparsity={args.svg_sparsity}, "
              f"geom={_num_views}x{_h_per_view}x{_w}")
    if getattr(args, "use_radial", False):
        importlib.import_module("methods.sparse_attention.radial.adapter").enable_radial_attention(
            unet,
            decay_factor=args.radial_decay_factor,
            block_size=args.radial_block_size,
            pad_small_layers=args.radial_pad_small_layers,
            first_layers_fp=args.radial_first_layers_fp,
            num_views=_num_views,
            H_per_view=_h_per_view,
            W=_w,
            model_type=args.radial_model_type,
        )
        print(f"[Sparse] Radial enabled: decay={args.radial_decay_factor}, "
              f"block_size={args.radial_block_size}, model_type={args.radial_model_type}, "
              f"geom={_num_views}x{_h_per_view}x{_w}")

    _install_attn_profiler(unet)

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
        _t0 = time.time()
        metrics = run_episode(agent, episode_id, args.save_dir, input_save_dir=args.input_save_dir)
        _dt = time.time() - _t0
        if metrics:
            metrics["wall_time_s"] = _dt
            _persist_wall_time(args.save_dir, episode_id, metrics)
            all_metrics.append(metrics)
            print(f"  -> {metrics['num_frames_pred']} frames, "
                  f"PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}, LPIPS={metrics['lpips']:.4f} "
                  f"| {_dt:.1f}s")

    # Save summary
    if all_metrics:
        summary = {
            "mean_psnr": sum(m["psnr"] for m in all_metrics) / len(all_metrics),
            "mean_ssim": sum(m["ssim"] for m in all_metrics) / len(all_metrics),
            "mean_lpips": sum(m["lpips"] for m in all_metrics) / len(all_metrics),
            "num_episodes": len(all_metrics),
            "subset": args.subset,
        }
        _times = [m["wall_time_s"] for m in all_metrics if "wall_time_s" in m]
        if _times:
            summary["mean_wall_time_s"] = sum(_times) / len(_times)
        summary_path = Path(args.save_dir) / "all_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n=== Summary ({len(all_metrics)} episodes) ===")
        print(f"PSNR: {summary['mean_psnr']:.3f}")
        print(f"SSIM: {summary['mean_ssim']:.4f}")
        print(f"LPIPS: {summary['mean_lpips']:.4f}")
        if "mean_wall_time_s" in summary:
            print(f"time: {summary['mean_wall_time_s']:.1f}s")


if __name__ == "__main__":
    main()
