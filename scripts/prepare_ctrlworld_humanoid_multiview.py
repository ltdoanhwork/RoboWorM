"""
Prepare datasets/humanoid/multiview (AIRBOT_MMK2 LeRobot) into Ctrl-World TRAINING format.

Same pipeline as prepare_ctrlworld_bimanual_multiview.py, adapted for humanoid:
  - AIRBOT_MMK2, fps=30, 4 cameras, 480x640, observation.state = 36-D, action = 36-D.
  - Camera names differ across tasks:
        Group A: cam_high_rgb, cam_left_wrist_rgb, cam_right_wrist_rgb, cam_third_view
        Group B: cam_head_rgb, cam_left_wrist_rgb, cam_right_wrist_rgb, cam_front_rgb
    We map to a canonical 3-view set (Ctrl-World needs exactly 3, height 72 = 3*24):
        view 0 = main external  (cam_high_rgb OR cam_head_rgb)
        view 1 = cam_left_wrist_rgb
        view 2 = cam_right_wrist_rgb
    The 4th view (cam_third_view / cam_front_rgb) is dropped.
  - Action/state condition = full 36-D observation.state, normalized by a freshly
    generated 1%/99% stat.json.

Downsampling: --down-sample D (default 6 -> 30fps->5fps). Video + state downsampled
by the same factor so stored arrays are frame-aligned (dataset uses down_sample=1).

Output layout (matches DROID/bimanual dataset_example layout):
  {output_dir}/{name}/annotation/{split}/{id}.json
  {output_dir}/{name}/videos/{split}/{id}/{0,1,2}.mp4          (resized 192x320)
  {output_dir}/{name}/latent_videos/{split}/{id}/{0,1,2}.pt    (SVD-VAE latents)

{name} = humanoid_multiview, {id} = "{subset}_{task}__{episode_index:06d}".
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


DATASET_BASE = "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/datasets/humanoid/multiview"
OUTPUT_BASE = "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/models/Ctrl-World/dataset_example"
SVD_PATH = "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/checkpoints/stabilityai/stable-video-diffusion-img2vid"

# Canonical 3-view mapping. view 0 = first main-external candidate present in the task.
MAIN_VIEW_CANDIDATES = ["cam_high_rgb", "cam_head_rgb"]
WRIST_LEFT = "cam_left_wrist_rgb"
WRIST_RIGHT = "cam_right_wrist_rgb"

TARGET_H = 192
TARGET_W = 320
STATE_DIM = 36  # full observation.state


def find_tasks(subset_dir):
    tasks = []
    for d in sorted(Path(subset_dir).iterdir()):
        if d.is_dir() and (d / "data").exists():
            tasks.append(d.name)
    return tasks


def find_episodes(task_dir):
    data_dir = Path(task_dir) / "data"
    episodes = []
    for chunk_dir in sorted(data_dir.glob("chunk-*")):
        for pq in sorted(chunk_dir.glob("episode_*.parquet")):
            ep_id = int(pq.stem.split("_")[1])
            chunk_id = int(chunk_dir.name.split("-")[1])
            episodes.append((chunk_id, ep_id))
    return episodes


def available_camera_dirs(task_dir):
    """Return set of camera key names (e.g. 'cam_high_rgb') that have a video dir."""
    vid_root = Path(task_dir) / "videos"
    cams = set()
    for chunk_dir in vid_root.glob("chunk-*"):
        for cam_dir in chunk_dir.iterdir():
            if cam_dir.is_dir() and cam_dir.name.startswith("observation.images."):
                cams.add(cam_dir.name.split("observation.images.")[-1])
    return cams


def resolve_views(task_dir):
    """Return the ordered list of 3 camera KEYS [main, left_wrist, right_wrist]."""
    cams = available_camera_dirs(task_dir)
    main = next((c for c in MAIN_VIEW_CANDIDATES if c in cams), None)
    if main is None or WRIST_LEFT not in cams or WRIST_RIGHT not in cams:
        return None
    return [main, WRIST_LEFT, WRIST_RIGHT]


def load_episode_instructions(task_dir):
    """Map episode_index -> instruction from meta/episodes.jsonl."""
    out = {}
    p = Path(task_dir) / "meta" / "episodes.jsonl"
    if p.exists():
        with open(p) as f:
            for line in f:
                obj = json.loads(line)
                tasks = obj.get("tasks") or []
                out[obj["episode_index"]] = tasks[0] if tasks else ""
    return out


def video_path(task_dir, chunk_id, cam_key, episode_id):
    return (
        Path(task_dir) / "videos" / f"chunk-{chunk_id:03d}" /
        f"observation.images.{cam_key}" / f"episode_{episode_id:06d}.mp4"
    )


def encode_view(video_file, vae, device, down_sample):
    video = mediapy.read_video(str(video_file))
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


def process_episode(task_dir, task_name, view_keys, chunk_id, episode_id, instruction,
                    out_root, split, vae, device, down_sample):
    parquet_path = (
        Path(task_dir) / "data" / f"chunk-{chunk_id:03d}" /
        f"episode_{episode_id:06d}.parquet"
    )
    if not parquet_path.exists():
        return None

    df = pd.read_parquet(parquet_path)
    raw_length = len(df)

    state_full = np.stack(df["observation.state"].values)  # (T, 36)
    state_ds = state_full[::down_sample]  # (n, 36)

    ep_id_str = f"{task_name}__{episode_id:06d}"

    resized_views = []
    latent_views = []
    for cam_key in view_keys:
        vf = video_path(task_dir, chunk_id, cam_key, episode_id)
        if not vf.exists():
            print(f"  Missing video: {vf}")
            return None
        resized, latent = encode_view(vf, vae, device, down_sample)
        resized_views.append(resized)
        latent_views.append(latent)

    n_video = min(v.shape[0] for v in latent_views)
    n_frames = min(n_video, len(state_ds))
    state_ds = state_ds[:n_frames]

    for view_idx in range(3):
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

    state_list = state_ds.tolist()
    annotation = {
        "texts": [instruction or "robot manipulation task"],
        "episode_id": ep_id_str,
        "task_name": task_name,
        "raw_episode_id": episode_id,
        "view_keys": view_keys,
        "success": True,
        "video_length": n_frames,
        "state_length": n_frames,
        "raw_length": raw_length,
        "down_sample": down_sample,
        "videos": [{"video_path": f"videos/{split}/{ep_id_str}/{i}.mp4"} for i in range(3)],
        "latent_videos": [{"latent_video_path": f"latent_videos/{split}/{ep_id_str}/{i}.pt"} for i in range(3)],
        "states": state_list,
        "observation.state.qpos": state_list,  # 36-D full state used by the training dataset
    }

    anno_dir = Path(out_root) / "annotation" / split
    anno_dir.mkdir(parents=True, exist_ok=True)
    with open(anno_dir / f"{ep_id_str}.json", "w") as f:
        json.dump(annotation, f)

    return {"id": ep_id_str, "length": n_frames, "split": split, "instruction": instruction}


def choose_split(global_idx):
    return "val" if global_idx % 20 == 19 else "train"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=["makovian", "non_makovian", "both"], default="both")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_BASE)
    parser.add_argument("--dataset-base", type=str, default=DATASET_BASE)
    parser.add_argument("--svd-path", type=str, default=SVD_PATH)
    parser.add_argument("--down-sample", type=int, default=6,
                        help="Take every D-th frame. 6 = 30fps->5fps (default), 1 = keep 30fps.")
    parser.add_argument("--name", type=str, default="humanoid_multiview")
    parser.add_argument("--limit-episodes", type=int, default=None)
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
            view_keys = resolve_views(task_dir)
            if view_keys is None:
                print(f"  SKIP task {task_name}: missing canonical views "
                      f"(need one of {MAIN_VIEW_CANDIDATES} + both wrists)")
                continue
            instr_map = load_episode_instructions(task_dir)
            episodes = find_episodes(task_dir)
            for (chunk_id, ep_id) in episodes:
                if args.limit_episodes is not None and global_idx >= args.limit_episodes:
                    break
                split = choose_split(global_idx)
                res = process_episode(
                    task_dir, f"{subset}_{task_name}", view_keys, chunk_id, ep_id,
                    instr_map.get(ep_id, ""), out_root, split, vae, device, args.down_sample,
                )
                if res:
                    results.append(res)
                    print(f"  [{global_idx}] {res['id']} ({split}) views={view_keys[0]} -> {res['length']} frames")
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
        "view_mapping": {"view0": MAIN_VIEW_CANDIDATES, "view1": WRIST_LEFT, "view2": WRIST_RIGHT},
        "total_episodes": len(results),
        "n_train": sum(1 for r in results if r["split"] == "train"),
        "n_val": sum(1 for r in results if r["split"] == "val"),
        "episodes": results,
    }
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "preparation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone: {len(results)} episodes "
          f"(train={summary['n_train']}, val={summary['n_val']}) -> {out_root}")


if __name__ == "__main__":
    main()
