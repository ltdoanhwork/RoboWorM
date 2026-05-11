import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from .camera_pose_stability import compute_pose_stability_metric
from .geometric_consistency import (
    geometric_calculation,
    geometric_consistency_metric,
    render_depth_from_point_map,
)
from .utils import intersect_frames


@dataclass(frozen=True)
class ModelContext:
    model: VGGT
    device: str
    autocast_dtype: torch.dtype


def default_device_dtype() -> tuple[str, torch.dtype]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16 if device == "cuda" else torch.float32
    return device, dtype


def init_model_context(
    *,
    device: str | None = None,
    autocast_dtype: torch.dtype | None = None,
    model_name: str = "facebook/VGGT-1B",
) -> ModelContext:
    if device is None or autocast_dtype is None:
        d, dt = default_device_dtype()
        device = d if device is None else device
        autocast_dtype = dt if autocast_dtype is None else autocast_dtype

    model = VGGT.from_pretrained(model_name).to(device)
    model.eval()
    return ModelContext(model=model, device=device, autocast_dtype=autocast_dtype)


@torch.no_grad()
def _run_vggt(
    ctx: ModelContext,
    image_paths: Iterable[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      extrinsic: (1,S,3,4)
      intrinsic: (1,S,3,3)
      depth_map: (1,S,H,W,1)
      depth_conf: (1,S,H,W,1)
    """
    images = load_and_preprocess_images(list(image_paths)).to(ctx.device)[None]
    if ctx.device == "cuda":
        with torch.cuda.amp.autocast(dtype=ctx.autocast_dtype):
            aggregated_tokens_list, ps_idx = ctx.model.aggregator(images)
            pose_enc = ctx.model.camera_head(aggregated_tokens_list)[-1]
            depth_map, depth_conf = ctx.model.depth_head(aggregated_tokens_list, images, ps_idx)
    else:
        aggregated_tokens_list, ps_idx = ctx.model.aggregator(images)
        pose_enc = ctx.model.camera_head(aggregated_tokens_list)[-1]
        depth_map, depth_conf = ctx.model.depth_head(aggregated_tokens_list, images, ps_idx)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    return extrinsic, intrinsic, depth_map, depth_conf


def _select_frame_ids(
    episode_dir: str,
    views: list[str],
    *,
    stride: int = 1,
    max_frames: int | None = None,
) -> list[int]:
    frame_ids = intersect_frames(list(views), episode_dir)
    if not frame_ids:
        raise ValueError(f"No common frames found in {episode_dir} for views={views}")
    frame_ids = frame_ids[:: max(1, int(stride))]
    if max_frames is not None:
        frame_ids = frame_ids[: int(max_frames)]
    return frame_ids


@torch.no_grad()
def compute_pose_stability_for_episode(
    ctx: ModelContext,
    *,
    episode_dir: str,
    views: list[str],
    stride: int = 1,
    max_frames: int | None = None,
    pose_std_rot_scale_deg: float = 5.0,
    pose_std_trans_scale: float = 0.20,
    pose_std_intr_scale: float = 0.05,
    pose_std_view_agg: str = "mean",
) -> dict:
    frame_ids = _select_frame_ids(episode_dir, views, stride=stride, max_frames=max_frames)

    extrinsics_all: list[torch.Tensor] = []
    intrinsics_all: list[torch.Tensor] = []

    for fi in frame_ids:
        paths = [os.path.join(episode_dir, v, f"frame_{fi:04d}.png") for v in views]
        extrinsic, intrinsic, _depth, _conf = _run_vggt(ctx, paths)
        extrinsics_all.append(extrinsic.detach().float().cpu())
        intrinsics_all.append(intrinsic.detach().float().cpu())

    score = compute_pose_stability_metric(
        extrinsics_all,
        intrinsics_all,
        pose_std_rot_scale_deg=pose_std_rot_scale_deg,
        pose_std_trans_scale=pose_std_trans_scale,
        pose_std_intr_scale=pose_std_intr_scale,
        pose_std_view_agg=pose_std_view_agg,
    )
    return {
        "pose_stability": float(score),
        "num_frames": int(len(frame_ids)),
    }


@torch.no_grad()
def compute_geometric_consistency_for_episode(
    ctx: ModelContext,
    *,
    episode_dir: str,
    views: list[str],
    stride: int = 1,
    max_frames: int | None = None,
    conf_thres_percentile: float = 20.0,
    residual_key: str = "residual_mean",
    residual_scale: float = 0.5,
    corr_power: float = 2.0,
    agg_views: str = "min",
    agg_frames: str = "mean",  # "mean" | "min"
) -> dict:
    frame_ids = _select_frame_ids(episode_dir, views, stride=stride, max_frames=max_frames)

    per_frame: list[dict] = []
    scores: list[float] = []

    for fi in frame_ids:
        paths = [os.path.join(episode_dir, v, f"frame_{fi:04d}.png") for v in views]
        extrinsic, intrinsic, depth_map, depth_conf = _run_vggt(ctx, paths)

        # Build point map from depth (independent prediction), then render depth into each camera
        point_map = unproject_depth_map_to_point_map(
            depth_map.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0)
        )  # (S,H,W,3) numpy

        depth_rendered_np, rendered_valid_mask = render_depth_from_point_map(
            point_map_by_unprojection=point_map,
            depth_map=depth_map,
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            eps=1e-6,
            confidence=depth_conf,
            conf_thres=float(conf_thres_percentile),
        )

        depth_rendered = torch.from_numpy(depth_rendered_np).to(device=ctx.device, dtype=torch.float32)
        if depth_rendered.ndim == 3:
            depth_rendered = depth_rendered.unsqueeze(0)  # (1,S,H,W)
        depth_rendered = depth_rendered.unsqueeze(-1)  # (1,S,H,W,1)

        valid_for_metric = rendered_valid_mask
        if isinstance(valid_for_metric, np.ndarray) and valid_for_metric.ndim == 3:
            valid_for_metric = valid_for_metric[np.newaxis, ...]  # (1,S,H,W)

        res = geometric_calculation(
            depth_joint=depth_rendered,
            depth_indep=depth_map,
            valid_mask=valid_for_metric,
        )
        score = geometric_consistency_metric(
            res,
            residual_key=residual_key,
            residual_scale=float(residual_scale),
            corr_power=float(corr_power),
            agg=agg_views,
        )

        scores.append(float(score))
        per_frame.append({"frame": int(fi), "geometric_consistency": float(score), "details": res})

    if not scores:
        overall = 0.0
    elif agg_frames == "min":
        overall = float(min(scores))
    elif agg_frames == "mean":
        overall = float(sum(scores) / len(scores))
    else:
        raise ValueError(f"Unknown agg_frames={agg_frames!r}")

    return {
        "geometric_consistency": float(overall),
        "geometric_consistency_per_frame": per_frame,
        "num_frames": int(len(frame_ids)),
        "agg_frames": str(agg_frames),
    }


@torch.no_grad()
def compute_video_metrics(
    ctx: ModelContext,
    *,
    episode_dir: str,
    views: list[str],
    stride: int = 1,
    max_frames: int | None = None,
    include_per_frame: bool = False,
) -> dict:
    pose = compute_pose_stability_for_episode(
        ctx,
        episode_dir=episode_dir,
        views=views,
        stride=stride,
        max_frames=max_frames,
    )
    geo = compute_geometric_consistency_for_episode(
        ctx,
        episode_dir=episode_dir,
        views=views,
        stride=stride,
        max_frames=max_frames,
    )

    out = {
        "episode_dir": str(episode_dir),
        "views": list(views),
        "stride": int(stride),
        "max_frames": None if max_frames is None else int(max_frames),
        "pose_stability": pose["pose_stability"],
        "geometric_consistency": geo["geometric_consistency"],
        "num_frames": int(min(pose["num_frames"], geo["num_frames"])),
    }
    if include_per_frame:
        out["geometric_consistency_per_frame"] = geo["geometric_consistency_per_frame"]
    return out

