from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKLTemporalDecoder
from tqdm import tqdm

from finetuning.config import FinetuneConfig, add_config_arguments, config_from_args
from finetuning.data.robo_multiview_manifest import build_manifest

try:
    from decord import VideoReader, cpu
except ModuleNotFoundError:  # pragma: no cover - depends on local env
    VideoReader = None
    cpu = None


def load_or_build_manifest(cfg: FinetuneConfig) -> dict[str, Any]:
    path = Path(cfg.manifest_path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    manifest = build_manifest(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def frame_ids_for_episode(length: int, stride: int) -> np.ndarray:
    ids = np.arange(0, length, stride, dtype=np.int64)
    if ids.size == 0:
        ids = np.array([0], dtype=np.int64)
    return ids


def load_action_sequence(parquet_path: Path, action_key: str, frame_ids: np.ndarray) -> np.ndarray:
    df = pd.read_parquet(parquet_path, columns=[action_key])
    max_idx = len(df) - 1
    clipped = np.clip(frame_ids, 0, max_idx)
    actions = np.stack(df[action_key].iloc[clipped].to_numpy()).astype(np.float32)
    return actions


def read_video_frames(video_path: Path, frame_ids: np.ndarray) -> np.ndarray:
    if VideoReader is not None:
        try:
            vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=2)
            max_idx = len(vr) - 1
            clipped = np.clip(frame_ids, 0, max_idx)
            return vr.get_batch(clipped.tolist()).asnumpy()
        except Exception:
            pass

    reader = imageio.get_reader(video_path)
    try:
        n_frames = reader.count_frames()
    except Exception:
        n_frames = int(reader.get_meta_data().get("nframes", 0))
    if n_frames <= 0:
        frames = [frame for frame in reader]
        max_idx = len(frames) - 1
        clipped = np.clip(frame_ids, 0, max_idx)
        return np.stack([frames[int(i)] for i in clipped])
    max_idx = n_frames - 1
    clipped = np.clip(frame_ids, 0, max_idx)
    return np.stack([reader.get_data(int(i)) for i in clipped])


def resize_frames_uint8(frames: np.ndarray, height: int, width: int, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(frames).to(device=device, dtype=torch.float32)
    tensor = tensor.permute(0, 3, 1, 2) / 255.0
    tensor = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return (tensor.clamp(0, 1) * 255).to(torch.uint8)


def encode_frames(
    vae: AutoencoderKLTemporalDecoder,
    frames_uint8: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    frames = frames_uint8.to(dtype=torch.float32) / 255.0 * 2.0 - 1.0
    latents = []
    with torch.no_grad():
        for i in range(0, frames.shape[0], batch_size):
            batch = frames[i : i + batch_size]
            latent = vae.encode(batch).latent_dist.sample().mul_(vae.config.scaling_factor)
            latents.append(latent.detach().cpu())
    return torch.cat(latents, dim=0)


def write_video(path: Path, frames_uint8: torch.Tensor, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = frames_uint8.detach().cpu().permute(0, 2, 3, 1).numpy()
    imageio.mimsave(path, arr, fps=fps, macro_block_size=1)


def sample_windows(action_len: int, sequence_len: int, interval: int) -> list[int]:
    if action_len < sequence_len:
        return []
    return list(range(0, action_len - sequence_len + 1, max(1, interval)))


def process_record(
    record: dict[str, Any],
    cfg: FinetuneConfig,
    vae: AutoencoderKLTemporalDecoder,
    device: torch.device,
    encode_batch_size: int,
    write_resized_videos: bool,
) -> tuple[list[dict[str, Any]], np.ndarray | None]:
    processed_root = Path(cfg.processed_root)
    dataset_name = record["dataset_name"]
    split = record["split"]
    ep = int(record["episode_index"])
    ep_name = f"episode_{ep:06d}"

    ann_path = processed_root / "datasets" / dataset_name / "annotation" / split / f"{ep_name}.json"
    latent_dir = processed_root / "datasets" / dataset_name / "latent_videos" / split / ep_name
    video_dir = processed_root / "datasets" / dataset_name / "videos" / split / ep_name

    if cfg.skip_existing and ann_path.exists() and all((latent_dir / f"{view_idx}.pt").exists() for view_idx in range(cfg.n_views)):
        annotation = json.loads(ann_path.read_text(encoding="utf-8"))
        starts = sample_windows(len(annotation["actions"]), cfg.sequence_length, cfg.start_frame_interval)
        return [
            {
                "dataset_name": dataset_name,
                "episode_id": ep_name,
                "split": split,
                "start_idx": start,
                "embodiment": record["embodiment"],
                "embodiment_id": record["embodiment_id"],
            }
            for start in starts
        ], np.asarray(annotation["actions"], dtype=np.float32)

    frame_ids = frame_ids_for_episode(int(record["length"]), cfg.downsample_stride)
    actions = load_action_sequence(Path(record["parquet_path"]), cfg.action_key, frame_ids)
    if actions.shape[-1] != cfg.action_dim:
        raise ValueError(f"{dataset_name}/{ep_name}: expected action_dim={cfg.action_dim}, got {actions.shape[-1]}")

    latent_paths: list[str] = []
    video_paths_out: list[str | None] = []
    for view_idx, video_path in enumerate(record["video_paths"][: cfg.n_views]):
        latent_path = latent_dir / f"{view_idx}.pt"
        video_out = video_dir / f"{view_idx}.mp4"
        latent_path.parent.mkdir(parents=True, exist_ok=True)

        if video_path is None:
            latent = torch.zeros((len(frame_ids), 4, cfg.latent_slot_height, cfg.latent_slot_width), dtype=torch.float32)
            torch.save(latent, latent_path)
            video_paths_out.append(None)
        else:
            frames = read_video_frames(Path(video_path), frame_ids)
            resized = resize_frames_uint8(frames, cfg.image_height, cfg.image_width, device)
            latent = encode_frames(vae, resized, encode_batch_size)
            torch.save(latent, latent_path)
            if write_resized_videos:
                write_video(video_out, resized, fps=cfg.fps)
                video_paths_out.append(str(video_out.relative_to(processed_root / "datasets" / dataset_name)))
            else:
                video_paths_out.append(None)
        latent_paths.append(str(latent_path.relative_to(processed_root / "datasets" / dataset_name)))

    annotation = {
        "dataset_name": dataset_name,
        "episode_id": ep_name,
        "source_episode_index": ep,
        "split": split,
        "texts": [record.get("task", "")],
        "embodiment": record["embodiment"],
        "embodiment_id": record["embodiment_id"],
        "fps": cfg.fps,
        "source_fps": record.get("fps", cfg.source_fps),
        "frame_ids": frame_ids.tolist(),
        "actions": actions.tolist(),
        "latent_videos": [{"latent_video_path": p} for p in latent_paths],
        "videos": [{"video_path": p} for p in video_paths_out],
        "camera_keys": record["camera_keys"][: cfg.n_views],
        "action_key": cfg.action_key,
        "state_key": cfg.state_key,
    }
    ann_path.parent.mkdir(parents=True, exist_ok=True)
    ann_path.write_text(json.dumps(annotation, indent=2), encoding="utf-8")

    starts = sample_windows(len(actions), cfg.sequence_length, cfg.start_frame_interval)
    samples = [
        {
            "dataset_name": dataset_name,
            "episode_id": ep_name,
            "split": split,
            "start_idx": start,
            "embodiment": record["embodiment"],
            "embodiment_id": record["embodiment_id"],
        }
        for start in starts
    ]
    return samples, actions


def percentile_stats(actions: np.ndarray) -> dict[str, list[float]]:
    return {
        "action_01": np.percentile(actions, 1, axis=0).astype(float).tolist(),
        "action_99": np.percentile(actions, 99, axis=0).astype(float).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute SVD latents for RoboCOIN multiview Ctrl-World finetuning.")
    add_config_arguments(parser)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--limit-records", type=int, default=None)
    parser.add_argument("--write-resized-videos", action="store_true")
    args = parser.parse_args()
    cfg = config_from_args(args)

    manifest = load_or_build_manifest(cfg)
    records = manifest["records"]
    if args.limit_records is not None:
        records = records[: args.limit_records]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(cfg.svd_model_path, subfolder="vae").to(device)
    vae.eval()

    samples_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    all_actions = []
    actions_by_embodiment: dict[str, list[np.ndarray]] = {}

    for record in tqdm(records, desc="prepare latents"):
        samples, actions = process_record(
            record,
            cfg,
            vae,
            device,
            encode_batch_size=args.encode_batch_size,
            write_resized_videos=args.write_resized_videos,
        )
        if samples:
            samples_by_split[record["split"]].extend(samples)
        if actions is not None and len(actions):
            all_actions.append(actions)
            actions_by_embodiment.setdefault(record["embodiment"], []).append(actions)

    meta_dir = Path(cfg.processed_root) / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    for split, samples in samples_by_split.items():
        (meta_dir / f"{split}_sample.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
        print(f"[prepare] {split}: {len(samples)} windows")

    if all_actions:
        all_cat = np.concatenate(all_actions, axis=0)
        stats = percentile_stats(all_cat)
        stats["action_dim"] = cfg.action_dim
        stats["action_key"] = cfg.action_key
        stats["embodiments"] = {
            emb: percentile_stats(np.concatenate(values, axis=0)) for emb, values in actions_by_embodiment.items()
        }
        (meta_dir / "stat.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"[prepare] wrote stats to {meta_dir / 'stat.json'}")


if __name__ == "__main__":
    main()
