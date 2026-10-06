"""
AUGMENT copy of scripts/infer/infer_bimanual_multiview_ctrlworld.py — do NOT use for the
dense/acceleration benchmark (that one must stay untouched so runs stay comparable).

Identical replay logic, plus `--no-gt`: an augmented (recolored) scene has no
ground truth, so this variant skips `gt_all_views.mp4` / `full_gt.mp4` and all
reference metrics (PSNR/SSIM/LPIPS), and writes `full_pred.mp4` as pred-only.
Reference-free metrics are computed afterwards by
scripts/augment/compute_augment_metrics.py. See .cursor/rules/augment-track-layout.

Used for bimanual AND humanoid Ctrl-World multiview (`--config bimanual|humanoid`),
reading the edited annotation tree built by scripts/augment/edit_ctrlworld_views.py.

Full-episode GT-anchored replay using a finetuned Ctrl-World on bimanual/multiview.

Reads from the SAME prepared training dataset dir (dataset_example/bimanual_multiview)
whose annotations already contain: 3-view resized videos, `states` (14-D qpos,
frame-aligned), `video_length`, `texts`. No separate inference prep needed.

For each episode:
  1. interact_num computed from episode length to cover the full clip.
  2. Replay GT 14-D qpos as action condition (no policy).
  3. Autoregressive generation with Ctrl-World.
  4. Save predicted video (+ PSNR/SSIM/LPIPS unless --no-gt).
"""

import sys
import os
import json
import time
import importlib
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import torch
import einops
import mediapy
import piq

ctrl_world_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "Ctrl-World"
)
sys.path.insert(0, ctrl_world_dir)

from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.ctrl_world import CrtlWorld
from decord import VideoReader, cpu
from accelerate import Accelerator


def _install_attn_profiler(unet):
    """Env-gated (CW_ATTN_PROFILE=1) per-attention-layer profiler.

    Wraps every attention processor's __call__ to record, per layer:
      - the sequence length (T_q, T_kv) and whether it is self-attention
      - CUDA-synced wall time per call
    Aggregated by (layer_name, T_q) and dumped at process exit. This is the
    ground-truth for "where does UNet attention time go" and "does the sparse
    backend actually fire on the big spatial layers" — read before trusting any
    tuning claim. No-op unless CW_ATTN_PROFILE=1.
    """
    import os as _os
    if _os.environ.get("CW_ATTN_PROFILE", "0") != "1":
        return
    import atexit as _atexit
    from collections import defaultdict as _dd

    stats = _dd(lambda: {"calls": 0, "ms": 0.0, "t_q": 0, "t_kv": 0, "self": False})

    class _ProfWrap:
        """Callable wrapper — diffusers invokes processor(attn, ...) which resolves
        __call__ on the TYPE, so we must wrap with a real object, not patch the
        instance's __call__ attribute (that would be ignored)."""
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

    # Wrap the .processor on each Attention module in-place (avoids
    # set_attn_processor type/key validation).
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

    # Also time the whole UNet forward so we can report attention-as-fraction-of-UNet.
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
              f"Non-attention (conv/resnet/temporal-mlp/proj) = "
              f"{100*(1-attn_ms/max(_fwd['ms'],1e-9)):.1f}%.")

    _atexit.register(_dump_fwd)


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

        # View layout. Default = 3-view vertical strip (3 rows x 1 col).
        # Grid variants set num_views/grid_rows/grid_cols on the config.
        self.num_views = int(getattr(args, "num_views", 3))
        self.grid_rows = int(getattr(args, "grid_rows", self.num_views))
        self.grid_cols = int(getattr(args, "grid_cols", 1))

    def stack_views_to_canvas(self, per_view_latents):
        """Place a list of per-view latents (each (..., h, w)) ROW-MAJOR into one
        (..., grid_rows*h, grid_cols*w) canvas. For the 3-view default this is a
        pure vertical concat; for the 2x2 grid it is a proper grid layout."""
        lat_h, lat_w = per_view_latents[0].shape[-2:]
        canvas = torch.zeros(
            (*per_view_latents[0].shape[:-2], self.grid_rows * lat_h, self.grid_cols * lat_w),
            dtype=per_view_latents[0].dtype, device=per_view_latents[0].device,
        )
        for v_i, v in enumerate(per_view_latents):
            r = v_i // self.grid_cols
            c = v_i % self.grid_cols
            canvas[..., r * lat_h:(r + 1) * lat_h, c * lat_w:(c + 1) * lat_w] = v
        return canvas

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
                width=int(args.width * self.grid_cols),
                height=int(args.height * self.grid_rows),
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
        latents = einops.rearrange(latents, "b f c (m h) (n w) -> (b m n) f c h w", m=self.grid_rows, n=self.grid_cols)

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


def split_category(episode_id):
    """Split a `{category}_{task}__{idx}` annotation stem into (category, name).

    Bimanual/humanoid annotation stems embed the makovian/non_makovian category
    (e.g. `non_makovian_fold_blue_towel__000070`). Per the sampling-dataset-layout
    rule, `{category}` must be its own path segment, not baked into the episode
    dir name. Returns (category, stripped_episode_name). If no known category
    prefix is found, category is None and the name is returned unchanged.
    """
    if episode_id.startswith("non_makovian_"):
        return "non_makovian", episode_id[len("non_makovian_"):]
    if episode_id.startswith("makovian_"):
        return "makovian", episode_id[len("makovian_"):]
    return None, episode_id


def episode_output_dir(base_dir, episode_id):
    """Path <base>/<category>/episode_<name> (falls back to flat if no category)."""
    category, name = split_category(episode_id)
    base = Path(base_dir)
    if category is not None:
        base = base / category
    return base / f"episode_{name}"


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
    first_latent = agent.stack_views_to_canvas([v[0] for v in video_latents]).unsqueeze(0)
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
            agent.stack_views_to_canvas([v[pred_step - 1] for v in predicted_latents]).unsqueeze(0)
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

    # AUGMENT: the edited scene has no ground truth, so `no_gt` drops every GT
    # artifact and every reference metric (see .cursor/rules/augment-track-layout).
    no_gt = getattr(args, "no_gt", False)

    pred_strip = np.concatenate([concat_pred[v] for v in range(num_views)], axis=2)
    gt_strip = np.concatenate([concat_gt[v] for v in range(num_views)], axis=2)
    full_pred = pred_strip if no_gt else np.concatenate([gt_strip, pred_strip], axis=1)

    out_fps = max(1, int(round(30 / args.infer_down_sample)))
    ep_save_dir = episode_output_dir(save_dir, episode_id)
    ep_save_dir.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(ep_save_dir / "full_pred.mp4"), full_pred, fps=out_fps)
    mediapy.write_video(str(ep_save_dir / "pred_all_views.mp4"), pred_strip, fps=out_fps)
    if not no_gt:
        mediapy.write_video(str(ep_save_dir / "gt_all_views.mp4"), gt_strip, fps=out_fps)

    if input_save_dir is not None:
        ep_input_dir = episode_output_dir(input_save_dir, episode_id)
        ep_input_dir.mkdir(parents=True, exist_ok=True)
        if not no_gt:
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
            "mode": "replay" if not no_gt else "replay_augment_no_gt",
            "states": anno["states"][:min_len],
        }
        with open(ep_input_dir / "metadata.json", "w") as f:
            json.dump(input_meta, f, indent=2)

    if no_gt:
        # Reference metrics are omitted entirely (not set to null): the edited scene
        # has no ground truth, so PSNR/SSIM/LPIPS keys would invite averaging
        # meaningless values. Reference-free metrics come from
        # scripts/augment/compute_augment_metrics.py.
        metrics = {
            "episode_id": episode_id,
            "instruction": instruction,
            "num_frames_pred": min_len,
            "interact_num": interact_num,
            "mode": "replay_augment_no_gt",
        }
        with open(ep_save_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        # Provenance: copy the edit record the editor wrote next to the annotation,
        # so an output dir alone explains what was perturbed.
        prov_src = Path(args.val_dataset_dir) / "augment_info" / f"{episode_id}.json"
        if prov_src.exists():
            with open(ep_save_dir / "augment_info.json", "w") as f:
                f.write(prov_src.read_text())
        return metrics

    # Prefer the real camera keys recorded at prep time (humanoid names vary per
    # task); fall back to the bimanual defaults for the 3/4-view layouts.
    view_names = anno.get("view_keys")
    if not view_names or len(view_names) != num_views:
        if getattr(agent, "num_views", 3) == 4:
            # 2x2 grid: row-major [TL, TR, BL, BR]
            view_names = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
        else:
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


def _enable_backend(agent, args, num_inference_steps):
    """Enable the (single, validated) acceleration backend on agent.model.unet.

    Mirrors scripts/infer/infer_single_arm_multiview_ctrlworld.py so bimanual/multiview
    supports the exact same WorldCache / DiCache / FasterCache / SiTo / ITM flags
    (same CrtlWorld UNet). Returns a possibly-adjusted guidance_scale.
    """
    guidance_scale = agent.args.guidance_scale
    unet = agent.model.unet

    if getattr(args, "use_worldcache", False):
        adapter = _import_cache_module("WorldCache", "adapter")
        adapter.enable_worldcache(
            unet,
            num_steps=num_inference_steps,
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
            unet,
            num_steps=num_inference_steps,
            rel_l1_thresh=args.dicache_rel_l1_thresh,
            ret_ratio=args.dicache_ret_ratio,
            probe_depth=args.dicache_probe_depth,
        )
        print(f"[Cache] DiCache enabled: thresh={args.dicache_rel_l1_thresh}, "
              f"ret_ratio={args.dicache_ret_ratio}, probe_depth={args.dicache_probe_depth}")
    if getattr(args, "use_fastercache", False):
        adapter = _import_cache_module("FasterCache", "adapter")
        adapter.enable_fastercache(
            unet,
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
                unet,
                keep_ratio=args.sito_st_keep_ratio,
                max_downsample_ratio=args.sito_max_downsample_ratio,
                recompute_every=(args.sito_plan_recompute_every or 999999),
            )
            print(f"[Prune] SiTo ST-hold enabled: keep_ratio={args.sito_st_keep_ratio}, "
                  f"max_downsample_ratio={args.sito_max_downsample_ratio}")
        else:
            adapter = _import_pruning_module("SiTo", "adapter")
            adapter.enable_sito(
                unet,
                start_layer_idx=args.sito_start_layer_idx or 0,
                prune_ratio=args.sito_prune_ratio,
                patch_h=args.sito_patch_h,
                patch_w=args.sito_patch_w,
                noise_alpha=args.sito_noise_alpha,
                sim_beta=args.sito_sim_beta,
                max_downsample_ratio=args.sito_max_downsample_ratio,
                plan_recompute_every=getattr(args, "sito_plan_recompute_every", 0),
            )
            print(f"[Prune] SiTo enabled: prune_ratio={args.sito_prune_ratio}")
    if getattr(args, "use_itm", False):
        itm_state = {}
        if getattr(args, "itm_spatiotemporal_hold", False):
            st_hold = _import_pruning_module("SiTo", "spatiotemporal_hold")
            st_hold.enable_spatiotemporal_hold(
                unet,
                keep_ratio=args.itm_st_keep_ratio,
                max_downsample_ratio=args.itm_max_downsample_ratio,
                recompute_every=(args.itm_plan_recompute_every or 999999),
                similarity_recover=True,
            )
            print(f"[Prune] ITM ST-hold enabled: keep_ratio={args.itm_st_keep_ratio}, "
                  f"max_downsample_ratio={args.itm_max_downsample_ratio}")
        elif getattr(args, "itm_block_hold", False):
            block_hold = _import_pruning_module("importance_token_merge", "block_hold")
            itm_state["prune_from_step"] = args.itm_prune_from_step
            itm_state["merge_from_step"] = args.itm_merge_from_step
            block_hold.enable_itm_block_hold(
                unet,
                itm_state=itm_state,
                start_layer_idx=args.itm_start_layer_idx or 0,
                compress_ratio=args.itm_compress_ratio,
                max_downsample_ratio=args.itm_max_downsample_ratio,
                self_importance=True,
                plan_recompute_every=getattr(args, "itm_plan_recompute_every", 0),
            )
        else:
            adapter = _import_pruning_module("importance_token_merge", "adapter")
            adapter.enable_itm(
                unet,
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
            if guidance_scale <= 1.0:
                guidance_scale = 2.0
                print(f"[ITM] guidance_scale overridden to {guidance_scale} (ITM requires > 1.0)")

    # ---- Sparse-attention backends (PISA / SVG / Radial) --------------------
    # These inject attention processors into the SVD UNet via set_attn_processor.
    # Geometry: the SVD UNet sees ONE stacked image. For the 3-view strip that is
    # 3 rows x 1 col (num_views=3, H_per_view=24, W=40 -> 72x40 latent); for the
    # 4-view 2x2 grid it is a single 48x80 canvas, which we pass as num_views=1,
    # H_per_view=48, W=80 (the SAP/stripe masks operate on the flat token grid, so
    # a single-canvas geometry is correct — there is no vertical view-stripe).
    _grid_rows = int(getattr(agent, "grid_rows", 3) or 3)
    _grid_cols = int(getattr(agent, "grid_cols", 1) or 1)
    _h_per_view = 24 * _grid_rows          # 192px/8 = 24 latent rows per view row
    _w = 40 * _grid_cols                   # 320px/8 = 40 latent cols per view col
    _num_views = 1                         # treat the stacked canvas as one image

    if getattr(args, "use_pisa", False):
        import importlib as _il
        _il.import_module("methods.sparse_attention.pisa.adapter").enable_pisa(
            unet,
            density=args.pisa_density,
            block_size=args.pisa_block_size,
            start_layer_idx=args.pisa_start_layer_idx,
        )
        print(f"[Sparse] PISA enabled: density={args.pisa_density}, "
              f"block_size={args.pisa_block_size}, start_layer={args.pisa_start_layer_idx}")
    if getattr(args, "use_svg", False):
        import importlib as _il
        _il.import_module("methods.sparse_attention.svg.adapter").enable_svg(
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
        import importlib as _il
        _il.import_module("methods.sparse_attention.radial.adapter").enable_radial_attention(
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
    return guidance_scale


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
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Split the episode list into N shards for parallel multi-GPU runs.")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="Which shard (0-based) this process handles. Requires --num_shards.")
    parser.add_argument("--config", choices=["bimanual", "bimanual_grid", "humanoid", "humanoid_grid"],
                        default="bimanual",
                        help="Which model config (sets action_dim, view grid, etc.).")
    parser.add_argument("--no-gt", dest="no_gt", action="store_true",
                        help="AUGMENT mode: the edited scene has no ground truth, so skip "
                             "gt_all_views.mp4 / full_gt.mp4 and all reference metrics "
                             "(PSNR/SSIM/LPIPS). full_pred.mp4 becomes pred-only.")

    # Cache backend args (mutually exclusive)
    _import_cache_module("WorldCache", "config").add_worldcache_args(parser)
    _import_cache_module("DiCache", "config").add_dicache_args(parser)
    _import_cache_module("FasterCache", "config").add_fastercache_args(parser)

    # Pruning/sparse attention backend args
    _import_pruning_module("SiTo", "config").add_sito_args(parser)
    _import_pruning_module("importance_token_merge", "config").add_itm_args(parser)

    # Sparse-attention backend args (PISA / SVG / Radial — SVD-UNet flavored)
    import importlib as _il
    _il.import_module("methods.sparse_attention.pisa.config").add_pisa_args(parser)
    _il.import_module("methods.sparse_attention.svg.config").add_svg_args(parser)
    _il.import_module("methods.sparse_attention.radial.config").add_radial_args(parser)

    args = parser.parse_args()

    from methods.cache_strategy.ctrl_world_utils import validate_ctrl_world_backend_args
    validate_ctrl_world_backend_args(args)

    sys.path.insert(0, ctrl_world_dir)
    if args.config == "humanoid":
        from config_humanoid import wm_args_humanoid as _wm_args
    elif args.config == "humanoid_grid":
        from config_humanoid_grid import wm_args_humanoid_grid as _wm_args
    elif args.config == "bimanual_grid":
        from config_bimanual_grid import wm_args_bimanual_grid as _wm_args
    else:
        from config_bimanual import wm_args_bimanual as _wm_args
    model_args = _wm_args()
    model_args.svd_model_path = args.svd_model_path
    model_args.clip_model_path = args.clip_model_path
    model_args.ckpt_path = args.ckpt_path
    model_args.val_model_path = args.ckpt_path
    model_args.val_dataset_dir = args.dataset_dir
    model_args.dataset_meta_info_path = args.dataset_meta_info_path
    model_args.data_stat_path = os.path.join(args.dataset_meta_info_path, args.dataset_name, "stat.json")
    model_args.infer_down_sample = args.infer_down_sample
    # run_episode() reads its flags off agent.args (== model_args), so the CLI
    # no-gt flag must be copied across or it silently stays False and GT is written.
    model_args.no_gt = args.no_gt
    if args.num_inference_steps is not None:
        model_args.num_inference_steps = args.num_inference_steps
    if args.guidance_scale is not None:
        model_args.guidance_scale = args.guidance_scale
        print(f"[override] guidance_scale = {args.guidance_scale}")

    agent = CtrlWorldAgent(model_args)

    # Enable acceleration backend (validated mutually-exclusive) on the UNet.
    guidance_scale = _enable_backend(agent, args, model_args.num_inference_steps)
    if guidance_scale != model_args.guidance_scale:
        model_args.guidance_scale = guidance_scale
    _install_attn_profiler(agent.model.unet)

    splits = ["val", "train"] if args.split == "all" else [args.split]
    episode_ids = []
    for split in splits:
        anno_dir = Path(args.dataset_dir) / "annotation" / split
        if anno_dir.exists():
            episode_ids += sorted([f.stem for f in anno_dir.glob("*.json")])
    episode_ids = sorted(set(episode_ids))
    if args.num_episodes:
        episode_ids = episode_ids[:args.num_episodes]
    if args.num_shards > 1:
        # Interleaved sharding keeps each shard's episode-length mix balanced.
        total_eps = len(episode_ids)
        episode_ids = episode_ids[args.shard_id::args.num_shards]
        print(f"[shard {args.shard_id}/{args.num_shards}] {len(episode_ids)}/{total_eps} episodes")

    print(f"Dataset dir: {args.dataset_dir}")
    print(f"Output dir: {args.save_dir}")
    print(f"Episodes to process: {len(episode_ids)}")

    all_metrics = []
    for ep_idx, episode_id in enumerate(episode_ids):
        ep_save_path = episode_output_dir(args.save_dir, episode_id) / "full_pred.mp4"
        if ep_save_path.exists():
            print(f"[{ep_idx+1}/{len(episode_ids)}] {episode_id} done, skipping")
            continue
        print(f"[{ep_idx+1}/{len(episode_ids)}] Episode {episode_id}")
        _t0 = time.time()
        metrics = run_episode(agent, episode_id, args.save_dir, input_save_dir=args.input_save_dir)
        _dt = time.time() - _t0
        if metrics:
            metrics["wall_time_s"] = _dt
            all_metrics.append(metrics)
            if args.no_gt:
                print(f"  -> {metrics['num_frames_pred']} frames "
                      f"(no-gt augment mode, reference metrics skipped) | {_dt:.1f}s")
            else:
                print(f"  -> {metrics['num_frames_pred']} frames, "
                      f"PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}, LPIPS={metrics['lpips']:.4f} "
                      f"| {_dt:.1f}s")

    if all_metrics:
        def _write_summary(metrics_list, out_dir, label):
            if args.no_gt:
                summary = {"num_episodes": len(metrics_list), "mode": "augment_no_gt"}
            else:
                summary = {
                    "mean_psnr": sum(m["psnr"] for m in metrics_list) / len(metrics_list),
                    "mean_ssim": sum(m["ssim"] for m in metrics_list) / len(metrics_list),
                    "mean_lpips": sum(m["lpips"] for m in metrics_list) / len(metrics_list),
                    "num_episodes": len(metrics_list),
                }
            _times = [m["wall_time_s"] for m in metrics_list if "wall_time_s" in m]
            if _times:
                summary["mean_wall_time_s"] = sum(_times) / len(_times)
            out_dir.mkdir(parents=True, exist_ok=True)
            # Sharded runs each write their own file so they don't clobber each
            # other; merge_ctrlworld_shard_summaries.py aggregates them after.
            fname = ("all_summary.json" if args.num_shards <= 1
                     else f"all_summary.shard{args.shard_id}of{args.num_shards}.json")
            with open(out_dir / fname, "w") as f:
                json.dump(summary, f, indent=2)
            if args.no_gt:
                print(f"=== Summary [{label}] ({len(metrics_list)} episodes, no-gt augment) ==="
                      + (f"  time: {summary['mean_wall_time_s']:.1f}s" if "mean_wall_time_s" in summary else ""))
                return summary
            print(f"=== Summary [{label}] ({len(metrics_list)} episodes) === "
                  f"PSNR: {summary['mean_psnr']:.3f}  "
                  f"SSIM: {summary['mean_ssim']:.4f}  "
                  f"LPIPS: {summary['mean_lpips']:.4f}"
                  + (f"  time: {summary['mean_wall_time_s']:.1f}s" if "mean_wall_time_s" in summary else ""))
            return summary

        # Per-category summaries (mirrors single_arm: one all_summary.json per
        # {category}/ dir) so bimanual/humanoid match the layout convention.
        by_category = {}
        for m in all_metrics:
            category, _ = split_category(m["episode_id"])
            by_category.setdefault(category, []).append(m)

        print()
        for category, metrics_list in by_category.items():
            out_dir = Path(args.save_dir)
            if category is not None:
                out_dir = out_dir / category
            _write_summary(metrics_list, out_dir, category or "all")


if __name__ == "__main__":
    main()
