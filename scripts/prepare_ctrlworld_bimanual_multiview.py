"""
Prepare datasets/bimanual/multiview (ALOHA LeRobot) into Ctrl-World TRAINING format.

Unlike prepare_ctrlworld_single_arm_multiview.py (which writes the *inference*
annotation), this script writes the *training* format expected by the Ctrl-World
training dataloader: pre-encoded SVD-VAE latents (.pt) + annotation JSON with a
per-frame-aligned 14-D qpos state array.

Bimanual specifics (see meta/info.json + modality.json):
  - fps = 30, 4 cameras, 480x640, observation.state = 42-D (qpos14+qvel14+effort14)
  - We use 3 views: cam_high, cam_left_wrist, cam_right_wrist (drop cam_low) to
    match Ctrl-World's hardcoded 3-view latent stacking (height 72 = 3*24).
  - Action/state condition = observation.state[:, 0:14] (qpos: 6 joints + gripper
    per arm), the direct analog of DROID's 7-D cartesian+gripper.

Downsampling:
  - --down-sample D takes every D-th frame (default 1 -> keep native 30fps).
    Video and state are downsampled by the SAME factor so the stored arrays are
    aligned 1:1 with the latent frames. The training dataset therefore uses
    down_sample=1 internally (state_id == rgb_id).

Output layout (matches DROID dataset_example layout):
  {output_dir}/{name}/annotation/{split}/{id}.json
  {output_dir}/{name}/videos/{split}/{id}/{0,1,2}.mp4          (resized 192x320)
  {output_dir}/{name}/latent_videos/{split}/{id}/{0,1,2}.pt    (SVD-VAE latents)

where {name} = bimanual_multiview_{subset} (or a merged name) and {id} =
"{task}__{episode_index}" (namespaced to avoid cross-task episode-id collisions).
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import mediapy
from diffusers.models import AutoencoderKLTemporalDecoder


DATASET_BASE = "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/datasets/bimanual/multiview"
OUTPUT_BASE = "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/models/Ctrl-World/dataset_example"
SVD_PATH = "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/checkpoints/stabilityai/stable-video-diffusion-img2vid"

# 3 views used for training/inference (cam_low dropped)
VIEW_ORDER = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]

TARGET_H = 192
TARGET_W = 320
QPOS_DIM = 14  # observation.state[:, 0:14]


def find_tasks(subset_dir):
    """Return sorted list of task names (subdirectories with a data/ folder)."""
    tasks = []
    for d in sorted(Path(subset_dir).iterdir()):
        if d.is_dir() and (d / "data").exists():
            tasks.append(d.name)
    return tasks


def find_episodes(task_dir):
    """Return sorted list of (chunk_id, episode_id) for a task."""
    data_dir = Path(task_dir) / "data"
    episodes = []
    for chunk_dir in sorted(data_dir.glob("chunk-*")):
        for pq in sorted(chunk_dir.glob("episode_*.parquet")):
            ep_id = int(pq.stem.split("_")[1])
            chunk_id = int(chunk_dir.name.split("-")[1])
            episodes.append((chunk_id, ep_id))
    return episodes


def load_task_instruction(task_dir):
    """Read the first task string from meta/tasks.jsonl (fallback to dir name)."""
    tasks_path = Path(task_dir) / "meta" / "tasks.jsonl"
    if tasks_path.exists():
        with open(tasks_path) as f:
            for line in f:
                obj = json.loads(line)
                if "task" in obj:
                    return obj["task"]
    return Path(task_dir).name.replace("_", " ")


def video_path(task_dir, chunk_id, view_name, episode_id):
    return (
        Path(task_dir) / "videos" / f"chunk-{chunk_id:03d}" / view_name /
        f"episode_{episode_id:06d}.mp4"
    )


def encode_view(video_file, vae, device, down_sample):
    """Load mp4 -> downsample -> resize 192x320 -> return (resized_uint8, latent)."""
    video = mediapy.read_video(str(video_file))  # (T, H, W, 3) uint8
    frames = torch.tensor(np.array(video)).permute(0, 3, 1, 2).float() / 255.0 * 2 - 1
    if down_sample > 1:
        frames = frames[::down_sample]
    x = torch.nn.functional.interpolate(
        frames, size=(TARGET_H, TARGET_W), mode="bilinear", align_corners=False
    )
    resized = ((x / 2.0 + 0.5).clamp(0, 1) * 255)
    resized = resized.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    x = x.to(device)
    with torch.no_grad():
        latents = []
        for i in range(0, len(x), 32):
            batch = x[i:i + 32]
            latent = vae.encode(batch).latent_dist.sample().mul_(vae.config.scaling_factor).cpu()
            latents.append(latent)
        latent = torch.cat(latents, dim=0)
    return resized, latent


def process_episode(task_dir, task_name, chunk_id, episode_id, instruction,
                    out_root, split, vae, device, down_sample):
    parquet_path = (
        Path(task_dir) / "data" / f"chunk-{chunk_id:03d}" /
        f"episode_{episode_id:06d}.parquet"
    )
    if not parquet_path.exists():
        return None

    df = pd.read_parquet(parquet_path)
    raw_length = len(df)

    # observation.state -> qpos (first 14 dims), downsampled to match video cadence
    state_full = np.stack(df["observation.state"].values)  # (T, 42)
    qpos = state_full[:, :QPOS_DIM]  # (T, 14)
    qpos_ds = qpos[::down_sample]  # (n, 14)

    ep_id_str = f"{task_name}__{episode_id:06d}"

    # Encode all 3 views
    resized_views = []
    latent_views = []
    for view_name in VIEW_ORDER:
        vf = video_path(task_dir, chunk_id, view_name, episode_id)
        if not vf.exists():
            print(f"  Missing video: {vf}")
            return None
        resized, latent = encode_view(vf, vae, device, down_sample)
        resized_views.append(resized)
        latent_views.append(latent)

    # Align lengths across views + state
    n_video = min(v.shape[0] for v in latent_views)
    n_frames = min(n_video, len(qpos_ds))
    qpos_ds = qpos_ds[:n_frames]

    # Save resized videos + latents
    for view_idx in range(len(VIEW_ORDER)):
        vid_dir = Path(out_root) / "videos" / split / ep_id_str
        vid_dir.mkdir(parents=True, exist_ok=True)
        mediapy.write_video(
            str(vid_dir / f"{view_idx}.mp4"),
            resized_views[view_idx][:n_frames],
            fps=max(1, int(round(30 / down_sample))),
        )
        lat_dir = Path(out_root) / "latent_videos" / split / ep_id_str
        lat_dir.mkdir(parents=True, exist_ok=True)
        torch.save(latent_views[view_idx][:n_frames], str(lat_dir / f"{view_idx}.pt"))

    # Annotation. states/qpos arrays are frame-aligned (cadence == latent frames),
    # so the training dataset uses down_sample=1 (state_id == rgb_id).
    qpos_list = qpos_ds.tolist()
    annotation = {
        "texts": [instruction],
        "episode_id": ep_id_str,
        "task_name": task_name,
        "raw_episode_id": episode_id,
        "success": True,
        "video_length": n_frames,
        "state_length": n_frames,
        "raw_length": raw_length,
        "down_sample": down_sample,
        "videos": [
            {"video_path": f"videos/{split}/{ep_id_str}/{i}.mp4"} for i in range(len(VIEW_ORDER))
        ],
        "latent_videos": [
            {"latent_video_path": f"latent_videos/{split}/{ep_id_str}/{i}.pt"} for i in range(len(VIEW_ORDER))
        ],
        # frame-aligned 14-D qpos, used both as `states` (benchmark parity) and as
        # the dedicated key the bimanual training dataset reads.
        "states": qpos_list,
        "observation.state.qpos": qpos_list,
    }

    anno_dir = Path(out_root) / "annotation" / split
    anno_dir.mkdir(parents=True, exist_ok=True)
    with open(anno_dir / f"{ep_id_str}.json", "w") as f:
        json.dump(annotation, f)

    return {"id": ep_id_str, "length": n_frames, "split": split, "instruction": instruction}


def choose_split(global_idx):
    """Deterministic ~5% val holdout (every 20th episode is val)."""
    return "val" if global_idx % 20 == 19 else "train"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=["makovian", "non_makovian", "both"], default="both")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_BASE)
    parser.add_argument("--dataset-base", type=str, default=DATASET_BASE)
    parser.add_argument("--svd-path", type=str, default=SVD_PATH)
    parser.add_argument("--down-sample", type=int, default=1,
                        help="Take every D-th frame. 1 = keep native 30fps, 6 = ~5fps.")
    parser.add_argument("--name", type=str, default="bimanual_multiview",
                        help="Output dataset name (merged across subsets).")
    parser.add_argument("--limit-episodes", type=int, default=None,
                        help="Debug: cap total episodes processed.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SVD VAE from {args.svd_path} on {device} ...")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(args.svd_path, subfolder="vae").to(device)
    vae.requires_grad_(False)

    subsets = ["makovian", "non_makovian"] if args.subset == "both" else [args.subset]
    out_root = os.path.join(args.output_dir, args.name)

    results = []
    global_idx = 0
    for subset in subsets:
        subset_dir = os.path.join(args.dataset_base, subset)
        tasks = find_tasks(subset_dir)
        print(f"\n[{subset}] {len(tasks)} tasks")
        for task_name in tasks:
            task_dir = os.path.join(subset_dir, task_name)
            instruction = load_task_instruction(task_dir)
            episodes = find_episodes(task_dir)
            for (chunk_id, ep_id) in episodes:
                if args.limit_episodes is not None and global_idx >= args.limit_episodes:
                    break
                split = choose_split(global_idx)
                out_id = f"{subset}_{task_name}__{ep_id:06d}"
                # Prefix task with subset to keep makovian/non_makovian distinct
                res = process_episode(
                    task_dir, f"{subset}_{task_name}", chunk_id, ep_id, instruction,
                    out_root, split, vae, device, args.down_sample,
                )
                if res:
                    results.append(res)
                    print(f"  [{global_idx}] {res['id']} ({split}) -> {res['length']} frames")
                else:
                    print(f"  [{global_idx}] {subset}/{task_name} ep {ep_id:06d} -> SKIPPED")
                global_idx += 1
            if args.limit_episodes is not None and global_idx >= args.limit_episodes:
                break
        if args.limit_episodes is not None and global_idx >= args.limit_episodes:
            break

    summary = {
        "name": args.name,
        "down_sample": args.down_sample,
        "views": VIEW_ORDER,
        "total_episodes": len(results),
        "n_train": sum(1 for r in results if r["split"] == "train"),
        "n_val": sum(1 for r in results if r["split"] == "val"),
        "episodes": results,
    }
    summary_path = os.path.join(out_root, "preparation_summary.json")
    os.makedirs(out_root, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone: {len(results)} episodes "
          f"(train={summary['n_train']}, val={summary['n_val']}) -> {out_root}")


if __name__ == "__main__":
    main()
