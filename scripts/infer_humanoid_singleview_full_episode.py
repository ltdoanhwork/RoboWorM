"""
Full-episode autoregressive rollout for humanoid singleview DreamDojo.

For each episode:
  1. Split into non-overlapping segments of (num_frames - 1) sampled frames.
  2. The first segment starts from the GT initial frame; every subsequent segment
     starts from the last predicted frame (pure autoregressive rollout).
  3. Generate (num_frames - 1) frames per segment via chunked autoregressive diffusion.
  4. Concatenate all segments into one video matching the original episode length.

Episode output dirs are named by the REAL trajectory_id (episode_{id:06d}),
matching DreamGen humanoid singleview naming. Episodes shorter than num_frames
are skipped.
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
    parser.add_argument("--experiment", type=str, default="dreamdojo_2b_480_640_gr1")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--num-frames", type=int, default=49,
                        help="Frames per segment (model generates num_frames-1 new frames + 1 conditioning)")
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--save-fps", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--guidance", type=float, default=0)
    parser.add_argument("--save-video-only", action="store_true", default=True,
                        help="Only save the generated prediction video (no GT/merged/metrics). Default on.")
    parser.add_argument("--save-full", dest="save_video_only", action="store_false",
                        help="Also save full_gt.mp4, full_merged.mp4 and metrics.json.")
    parser.add_argument("--gt-only", action="store_true",
                        help="Write ONLY full_gt.mp4 for episodes that already have a "
                             "full_pred.mp4, and skip diffusion entirely (no GPU, no "
                             "checkpoint load). Backfills the reference for runs made "
                             "with the default video-only save mode.")
    parser.add_argument("--force-gt", action="store_true",
                        help="With --gt-only, overwrite an existing full_gt.mp4.")

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
        "calls (1 = every call, the upstream behaviour). The profile costs ~0.58ms at "
        "S=4800 (~47%% of a dense call), so caching it is what makes SVG1 beat dense.",
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
    """Build hydra experiment overrides for the sparse-attention backends."""
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
    """Build cache config from CLI args."""
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
    """Build model mirroring action_conditioned.py inference() flow."""
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
    # finetuned checkpoint we then pass to Video2WorldInference anyway. Resolving
    # locally first and handing the path to the constructor keeps the validator
    # off the network entirely.
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
        single_base_index=True,
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
    """Build dataset with single_base_index=False to access all base_indices."""
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
    """Build a plan: for each episode, list the dataset indices for non-overlapping segments."""
    episodes = {}  # (ds_idx, traj_id) -> [(global_data_id, base_index), ...]

    global_offset = 0
    for ds_idx, ds in enumerate(dataset.datasets):
        lerobot_ds = ds.lerobot_dataset
        for local_idx, (traj_id, base_index) in enumerate(lerobot_ds.all_steps):
            key = (ds_idx, int(traj_id))
            if key not in episodes:
                episodes[key] = []
            episodes[key].append((global_offset + local_idx, int(base_index)))
        global_offset += len(ds)

    # For each episode, pick non-overlapping segments
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
                "timestep_interval": int(timestep_interval),
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



def generate_segment(video2world_cli, img_array, actions, chunk_size, lam_video=None, guidance=0):
    """Generate one segment: initial GT frame + actions -> generated video frames."""
    chunk_video = []
    first_round = True

    for i in range(0, len(actions), chunk_size):
        actions_chunk = actions[i: i + chunk_size]
        if actions_chunk.shape[0] != chunk_size:
            break

        current_lam_video = None
        if lam_video is not None:
            current_lam_video = lam_video[i * 2: (i + chunk_size) * 2]

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


def _resolve_iter_name(args) -> str:
    """The checkpoint-iteration directory name, without loading the model.

    Factored out of build_model so --gt-only lands in the SAME save_root as the run that
    produced the predictions. Reading it from latest_checkpoint.txt is exactly what
    build_model does, just without the 2B weight load.
    """
    checkpoints_dir = Path(args.checkpoints_dir)
    last_checkpoint_file = checkpoints_dir / "latest_checkpoint.txt"
    if not last_checkpoint_file.exists():
        parent_file = checkpoints_dir.parent / "latest_checkpoint.txt"
        if parent_file.exists():
            checkpoints_dir = checkpoints_dir.parent
            last_checkpoint_file = parent_file
    if not last_checkpoint_file.exists():
        raise FileNotFoundError(
            f"Could not find latest_checkpoint.txt in {args.checkpoints_dir} or its parent.")
    with open(last_checkpoint_file) as f:
        return (checkpoints_dir / f.read().strip()).name


def _count_frames(path) -> int:
    """Exact frame count of an existing mp4, via the decoder rather than metadata.

    Container `nb_frames` is absent or wrong on several of these files; GT length is
    matched to the prediction, so a wrong count here would silently misalign the pair.
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    n = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    cap.release()
    return n


def main():
    args = parse_args()

    from cosmos_oss.init import init_environment, cleanup_environment
    init_environment()
    torch.enable_grad(False)

    if args.gt_only:
        # No diffusion, so no checkpoint and no GPU. iter_name still has to match the
        # generating run's directory, and it comes from latest_checkpoint.txt rather
        # than from the loaded model, so it can be read directly.
        iter_name = _resolve_iter_name(args)
        video2world_cli = None
        print(f"[gt-only] skipping model load; iter={iter_name}")
    else:
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
        "chunk_size": args.chunk_size,
        "guidance": args.guidance,
        "save_fps": args.save_fps,
        "num_episodes": num_episodes,
        "total_episodes": total_episodes,
        "worldcache_enabled": getattr(args, "worldcache_enabled", False),
        "worldcache_num_steps": getattr(args, "worldcache_num_steps", None),
        "worldcache_rel_l1_thresh": getattr(args, "worldcache_rel_l1_thresh", None),
        "worldcache_ret_ratio": getattr(args, "worldcache_ret_ratio", None),
        "worldcache_probe_depth": getattr(args, "worldcache_probe_depth", None),
        "worldcache_motion_sensitivity": getattr(args, "worldcache_motion_sensitivity", None),
        "fastercache_enabled": getattr(args, "fastercache_enabled", False),
        "fastercache_start_step": getattr(args, "fastercache_start_step", None),
        "fastercache_model_interval": getattr(args, "fastercache_model_interval", None),
        "fastercache_block_interval": getattr(args, "fastercache_block_interval", None),
        "dicache_enabled": getattr(args, "dicache_enabled", False),
        "dicache_num_steps": getattr(args, "dicache_num_steps", None),
        "dicache_rel_l1_thresh": getattr(args, "dicache_rel_l1_thresh", None),
        "dicache_ret_ratio": getattr(args, "dicache_ret_ratio", None),
        "dicache_probe_depth": getattr(args, "dicache_probe_depth", None),
        "use_svg": getattr(args, "use_svg", False),
        "svg_variant": getattr(args, "svg_variant", None),
        "svg_start_layer_idx": getattr(args, "svg_start_layer_idx", None),
        "svg_dense_step_frac": getattr(args, "svg_dense_step_frac", None),
        "svg_sparsity": getattr(args, "svg_sparsity", None),
        "svg_num_sampled_rows": getattr(args, "svg_num_sampled_rows", None),
        "svg_sample_mse_max_row": getattr(args, "svg_sample_mse_max_row", None),
        "svg_profile_recompute_every": getattr(args, "svg_profile_recompute_every", None),
        "svg_num_q_centroids": getattr(args, "svg_num_q_centroids", None),
        "svg_num_k_centroids": getattr(args, "svg_num_k_centroids", None),
        "svg_top_p_kmeans": getattr(args, "svg_top_p_kmeans", None),
        "svg_min_kc_ratio": getattr(args, "svg_min_kc_ratio", None),
        "use_pisa": getattr(args, "use_pisa", False),
        "pisa_density": getattr(args, "pisa_density", None),
        "pisa_block_size": getattr(args, "pisa_block_size", None),
        "pisa_start_layer_idx": getattr(args, "pisa_start_layer_idx", None),
        "use_radial": getattr(args, "use_radial", False),
        "radial_decay_factor": getattr(args, "radial_decay_factor", None),
        "radial_block_size": getattr(args, "radial_block_size", None),
        "radial_start_layer_idx": getattr(args, "radial_start_layer_idx", None),
        "radial_model_type": getattr(args, "radial_model_type", None),
    }
    with open(save_root / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    guidance = args.guidance
    all_psnr, all_ssim, all_lpips = [], [], []
    all_gen_times = []

    for ep_idx in range(num_episodes):
        ep_info = plan[ep_idx]
        traj_id = ep_info["traj_id"]
        # Match DreamGen humanoid singleview naming: episode_{id:06d} (id = real trajectory_id).
        ep_dir_name = f"episode_{traj_id:06d}"
        ep_save_dir = save_root / ep_dir_name

        if args.gt_only:
            # Backfill the reference beside an EXISTING prediction, no diffusion.
            #
            # The GT frames are the same `dataset[data_id]` tensors the generating run
            # concatenated, taken over the same segment list, so this reproduces exactly
            # what --save-full would have written. Only the final length differs in how
            # it is obtained: --save-full trims to min(pred, gt), and here the prediction
            # is already on disk, so it is read and used directly. That keeps pred and GT
            # frame-aligned, which is the one property PSNR depends on.
            pred_path = ep_save_dir / "full_pred.mp4"
            gt_path = ep_save_dir / "full_gt.mp4"
            if not pred_path.exists():
                print(f"[{ep_idx}/{num_episodes}] {ep_dir_name}: no full_pred.mp4, skipping.")
                continue
            if gt_path.exists() and not args.force_gt:
                print(f"[{ep_idx}/{num_episodes}] {ep_dir_name}: full_gt.mp4 exists, skipping.")
                continue
            gt_segs = []
            for data_id in ep_info["segment_data_ids"]:
                sample = dataset[data_id]
                gt_segs.append(sample["video"].permute(1, 2, 3, 0).numpy())
            if not gt_segs:
                print(f"[{ep_idx}/{num_episodes}] {ep_dir_name}: no GT segments, skipping.")
                continue
            concat_gt = np.concatenate(gt_segs, axis=0)
            n_pred = _count_frames(pred_path)
            n = min(len(concat_gt), n_pred)
            mediapy.write_video(str(gt_path), concat_gt[:n], fps=args.save_fps)
            print(f"[{ep_idx}/{num_episodes}] {ep_dir_name}: wrote full_gt.mp4 "
                  f"{n} frames (pred={n_pred}, gt_avail={len(concat_gt)})")
            continue

        if (ep_save_dir / "full_pred.mp4").exists():
            print(f"[{ep_idx}/{num_episodes}] {ep_dir_name} already exists, skipping.")
            continue

        num_segments = len(ep_info["segment_data_ids"])
        print(f"[{ep_idx}/{num_episodes}] {ep_dir_name} traj_id={traj_id}, "
              f"length={ep_info['traj_length']}, segments={num_segments}")

        ep_save_dir.mkdir(parents=True, exist_ok=True)
        segment_videos = []
        gt_segment_videos = []
        last_pred_frame = None  # autoregressive: carry forward last predicted frame

        # The chunk budget is PER EPISODE, so clear the accumulator here or the
        # first episode spends it all and every later episode records nothing.
        _CHUNK_TIMES.clear()
        gen_start_time = time.time()
        for seg_idx, data_id in enumerate(ep_info["segment_data_ids"]):
            sample = dataset[data_id]

            video_tensor = sample["video"]
            gt_video = video_tensor.permute(1, 2, 3, 0).numpy()  # (T, H, W, C)
            actions = sample["action"][:args.num_frames - 1]
            if isinstance(actions, torch.Tensor):
                actions = actions.numpy()
            lam_video = sample.get("lam_video", None)

            if seg_idx == 0:
                # Only the very first segment uses GT initial frame
                img_array = video_tensor.transpose(0, 1)[:1]  # (1, C, H, W)
            else:
                # Subsequent segments use the last predicted frame (autoregressive rollout)
                img_array = torchvision.transforms.functional.to_tensor(last_pred_frame).unsqueeze(0) * 255.0

            _cap = _max_chunks()
            if _cap and len(_CHUNK_TIMES) >= _cap:
                print(f"  [ChunkTiming] episode budget of {_cap} chunk(s) spent,"
                      " skipping remaining segments", flush=True)
                break
            segment = generate_segment(
                video2world_cli, img_array, actions, args.chunk_size, lam_video,
                guidance=guidance,
            )

            if segment is None:
                print(f"  Segment {seg_idx} failed (actions too short for chunk_size), stopping episode.")
                break

            last_pred_frame = segment[-1]  # (H, W, C) uint8 numpy
            segment_videos.append(segment)
            gt_segment_videos.append(gt_video)

        gen_elapsed = time.time() - gen_start_time

        if not segment_videos:
            continue

        all_gen_times.append(gen_elapsed)
        print(f"  [Timing] Episode {ep_idx}: generation took {gen_elapsed:.2f}s")

        # Concatenate all segments
        concat_pred = np.concatenate(segment_videos, axis=0)

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

        concat_gt = np.concatenate(gt_segment_videos, axis=0)

        min_len = min(len(concat_pred), len(concat_gt))
        if len(concat_pred) != len(concat_gt):
            print(f"  [Info] Frame mismatch: pred={len(concat_pred)}, gt={len(concat_gt)}, using min={min_len}")
        concat_pred = concat_pred[:min_len]
        concat_gt = concat_gt[:min_len]

        # Save videos
        mediapy.write_video(str(ep_save_dir / "full_pred.mp4"), concat_pred, fps=args.save_fps)
        mediapy.write_video(str(ep_save_dir / "full_gt.mp4"), concat_gt, fps=args.save_fps)
        concat_merged = np.concatenate([concat_gt, concat_pred], axis=2)
        mediapy.write_video(str(ep_save_dir / "full_merged.mp4"), concat_merged, fps=args.save_fps)

        # Compute metrics
        x_batch = torch.clamp(torch.from_numpy(concat_pred.copy()) / 255.0, 0, 1).permute(0, 3, 1, 2)
        y_batch = torch.clamp(torch.from_numpy(concat_gt.copy()) / 255.0, 0, 1).permute(0, 3, 1, 2)
        psnr_val = piq.psnr(x_batch, y_batch).mean().item()
        ssim_val = piq.ssim(x_batch, y_batch).mean().item()
        lpips_val = piq.LPIPS()(x_batch, y_batch).mean().item()

        with open(ep_save_dir / "metrics.json", "w") as f:
            json.dump({
                "psnr": psnr_val, "ssim": ssim_val, "lpips": lpips_val,
                "gen_time_s": round(gen_elapsed, 2),
                "num_segments": len(segment_videos),
                "total_frames_pred": len(concat_pred),
                "total_frames_gt": ep_info["traj_length"],
                "trajectory_id": ep_info["traj_id"],
            }, f, indent=2)

        all_psnr.append(psnr_val)
        all_ssim.append(ssim_val)
        all_lpips.append(lpips_val)
        print(f"  -> {len(concat_pred)} frames, PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}, LPIPS={lpips_val:.4f}")

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
