import os
from collections.abc import Iterator, Sequence
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
    multiview_disjoint_bbox,
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


def _frame_image_paths(episode_dir: str, views: list[str], fi: int) -> list[str]:
    return [os.path.join(episode_dir, v, f"frame_{fi:04d}.png") for v in views]


def _chunk_list(items: list[int], chunk_size: int) -> list[list[int]]:
    size = max(1, int(chunk_size))
    return [items[i : i + size] for i in range(0, len(items), size)]


def _load_images_batched(
    ctx: ModelContext,
    episode_dir: str,
    views: list[str],
    frame_ids: list[int],
) -> torch.Tensor:
    """Load frames as (B, S, 3, H, W) where B=len(frame_ids), S=len(views)."""
    paths: list[str] = []
    for fi in frame_ids:
        paths.extend(_frame_image_paths(episode_dir, views, fi))
    flat = load_and_preprocess_images(paths).to(ctx.device)
    b, s = len(frame_ids), len(views)
    return flat.view(b, s, *flat.shape[1:])


@torch.no_grad()
def _forward_vggt(
    ctx: ModelContext,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
      images: (B, S, 3, H, W)

    Returns:
      extrinsic: (B,S,3,4)
      intrinsic: (B,S,3,3)
      depth_map: (B,S,H,W,1)
      depth_conf: (B,S,H,W,1)
    """
    if images.ndim != 5:
        raise ValueError(f"Expected images (B,S,3,H,W), got {tuple(images.shape)}")

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


@torch.no_grad()
def _run_vggt(
    ctx: ModelContext,
    image_paths: Iterable[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single timestep (B=1). Returns (1,S,...) tensors.
    """
    images = load_and_preprocess_images(list(image_paths)).to(ctx.device).unsqueeze(0)
    extrinsic, intrinsic, depth_map, depth_conf = _forward_vggt(ctx, images)
    return extrinsic, intrinsic, depth_map, depth_conf


@torch.no_grad()
def _iter_vggt_frames(
    ctx: ModelContext,
    *,
    episode_dir: str,
    views: list[str],
    frame_ids: list[int],
    frame_batch_size: int = 4,
) -> Iterator[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Yield per-frame VGGT outputs using batch dimension B for multiple timesteps.

    Each yielded tensor has batch dim 1: (1,S,...).
    """
    for chunk in _chunk_list(frame_ids, frame_batch_size):
        images = _load_images_batched(ctx, episode_dir, views, chunk)
        extrinsic, intrinsic, depth_map, depth_conf = _forward_vggt(ctx, images)
        for b, fi in enumerate(chunk):
            yield (
                int(fi),
                extrinsic[b : b + 1],
                intrinsic[b : b + 1],
                depth_map[b : b + 1],
                depth_conf[b : b + 1],
            )


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

def _depth_pair_dissimilar_fraction(
    depth_map: torch.Tensor,
    *,
    eps: float = 1e-6,
    relative_diff_threshold: float = 0.25,
) -> float:
    """
    Trên mask chung (finite, depth > eps), tỷ lệ pixel có
    |d0-d1| / (0.5*(d0+d1)+eps) > relative_diff_threshold.

    Dùng để bắt geometric_consistency ~1 giả khi hai view depth thực tế lệch nhiều.
    Ít hơn 2 view → trả về 0.0 (không kích hoạt check).
    """
    d = depth_map.detach().float().cpu().numpy()
    if d.ndim == 5:
        d = d[0, :, :, :, 0]
    elif d.ndim == 4 and d.shape[-1] == 1:
        d = d[0, :, :, :, 0]
    elif d.ndim == 4:
        d = d[0]
    else:
        raise ValueError(f"depth_map must be (1,S,H,W,1)-like, got {tuple(depth_map.shape)}")
    if d.shape[0] < 2:
        return 0.0
    d0, d1 = d[0], d[1]
    m = np.isfinite(d0) & np.isfinite(d1) & (d0 > eps) & (d1 > eps)
    if not np.any(m):
        return 0.0
    a = d0[m].astype(np.float64)
    b = d1[m].astype(np.float64)
    rel = np.abs(a - b) / (0.5 * (a + b) + eps)
    return float(np.mean(rel > float(relative_diff_threshold)))


@dataclass(frozen=True)
class _GeometricFrameResult:
    frame: int
    geometric_consistency: float
    disjoint: bool
    details: dict | None
    include_in_overall: bool


@torch.no_grad()
def _process_geometric_frame(
    ctx: ModelContext,
    fi: int,
    *,
    extrinsic: torch.Tensor,
    intrinsic: torch.Tensor,
    depth_map: torch.Tensor,
    depth_conf: torch.Tensor,
    conf_thres_percentile: float = 50.0,
    high_geo_score_disjoint_threshold: float = 0.99,
    depth_pair_dissimilar_fraction_min: float = 0.5,
    depth_pair_relative_diff_threshold: float = 0.25,
) -> _GeometricFrameResult:
    point_map = unproject_depth_map_to_point_map(
        depth_map.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0)
    )

    disjoint = multiview_disjoint_bbox(point_map, min_overlap_ratio=0.05)
    is_disjoint = bool(disjoint["disjoint"])

    if is_disjoint:
        return _GeometricFrameResult(
            frame=int(fi),
            geometric_consistency=0.0,
            disjoint=True,
            details=None,
            include_in_overall=False,
        )

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
        depth_rendered = depth_rendered.unsqueeze(0)
    depth_rendered = depth_rendered.unsqueeze(-1)

    valid_for_metric = rendered_valid_mask
    if isinstance(valid_for_metric, np.ndarray) and valid_for_metric.ndim == 3:
        valid_for_metric = valid_for_metric[np.newaxis, ...]

    res = geometric_calculation(
        depth=depth_map,
        depth_rendered=depth_rendered,
        valid_mask=valid_for_metric,
    )
    score = float(geometric_consistency_metric(res))
    depth_pair_dissimilar_fraction = _depth_pair_dissimilar_fraction(
        depth_map,
        relative_diff_threshold=float(depth_pair_relative_diff_threshold),
    )
    if score > float(high_geo_score_disjoint_threshold) and depth_pair_dissimilar_fraction > float(
        depth_pair_dissimilar_fraction_min
    ):
        return _GeometricFrameResult(
            frame=int(fi),
            geometric_consistency=0.0,
            disjoint=True,
            details=res,
            include_in_overall=False,
        )

    return _GeometricFrameResult(
        frame=int(fi),
        geometric_consistency=score,
        disjoint=False,
        details=res,
        include_in_overall=True,
    )


def _aggregate_geometric_results(per_frame: list[dict]) -> dict:
    scores = [float(pf["geometric_consistency"]) for pf in per_frame if not pf["disjoint"]]
    total_frames = len(per_frame)
    num_disjoint_frames = sum(1 for pf in per_frame if pf["disjoint"])

    def _agg_frame_scores(frame_scores: list[float]) -> float:
        if not frame_scores:
            return 0.0
        return float(sum(frame_scores) / len(frame_scores))

    disjoint_frames = [int(pf["frame"]) for pf in per_frame if pf["disjoint"]]
    return {
        "geometric_consistency": float(_agg_frame_scores(scores)),
        "geometric_consistency_per_frame": per_frame,
        "num_frames": int(total_frames),
        "num_disjoint_frames": int(num_disjoint_frames),
        "num_non_disjoint_frames": int(total_frames - num_disjoint_frames),
        "disjoint_frame_ratio": float(num_disjoint_frames / total_frames) if total_frames else 0.0,
        "disjoint_frames": disjoint_frames,
    }


def _pose_stability_from_tensors(
    extrinsics_all: list[torch.Tensor],
    intrinsics_all: list[torch.Tensor],
    used_frame_ids: list[int],
) -> float:
    if used_frame_ids == [0] or not used_frame_ids:
        return 0.0
    return float(
        compute_pose_stability_metric(extrinsics_all, intrinsics_all)
    )


@torch.no_grad()
def compute_pose_stability_for_episode(
    ctx: ModelContext,
    *,
    episode_dir: str,
    views: list[str],
    stride: int = 1,
    max_frames: int | None = None,
    exclude_frame_ids: Sequence[int] | None = None,
    frame_batch_size: int = 4,
) -> dict:
    frame_ids = _select_frame_ids(episode_dir, views, stride=stride, max_frames=max_frames)
    exclude = set(int(fi) for fi in (exclude_frame_ids or ()))
    pose_frame_ids = [fi for fi in frame_ids if fi not in exclude]

    extrinsics_all: list[torch.Tensor] = []
    intrinsics_all: list[torch.Tensor] = []
    used_frame_ids: list[int] = []

    for fi, extrinsic, intrinsic, _depth, _conf in _iter_vggt_frames(
        ctx,
        episode_dir=episode_dir,
        views=views,
        frame_ids=pose_frame_ids,
        frame_batch_size=frame_batch_size,
    ):
        extrinsics_all.append(extrinsic.detach().float().cpu())
        intrinsics_all.append(intrinsic.detach().float().cpu())
        used_frame_ids.append(int(fi))

    score = _pose_stability_from_tensors(extrinsics_all, intrinsics_all, used_frame_ids)
    return {
        "pose_stability": float(score),
        "num_frames": int(len(frame_ids)),
        "num_frames_used": int(len(used_frame_ids)),
        "num_frames_excluded": int(len(frame_ids) - len(used_frame_ids)),
        "excluded_frame_ids": sorted(exclude & set(frame_ids)),
        "used_frame_ids": used_frame_ids,
    }

@torch.no_grad()
def compute_geometric_consistency_for_episode(
    ctx: ModelContext,
    *,
    episode_dir: str,
    views: list[str],
    stride: int = 1,
    max_frames: int | None = None,
    conf_thres_percentile: float = 50.0,

    high_geo_score_disjoint_threshold: float = 0.99,
    depth_pair_dissimilar_fraction_min: float = 0.5,
    depth_pair_relative_diff_threshold: float = 0.25,
    frame_batch_size: int = 4,
) -> dict:
    frame_ids = _select_frame_ids(episode_dir, views, stride=stride, max_frames=max_frames)
    frame_ids = [fi for fi in frame_ids if fi > 0]

    per_frame: list[dict] = []

    for fi, extrinsic, intrinsic, depth_map, depth_conf in _iter_vggt_frames(
        ctx,
        episode_dir=episode_dir,
        views=views,
        frame_ids=frame_ids,
        frame_batch_size=frame_batch_size,
    ):
        gf = _process_geometric_frame(
            ctx,
            fi,
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            depth_map=depth_map,
            depth_conf=depth_conf,
            conf_thres_percentile=conf_thres_percentile,
            high_geo_score_disjoint_threshold=high_geo_score_disjoint_threshold,
            depth_pair_dissimilar_fraction_min=depth_pair_dissimilar_fraction_min,
            depth_pair_relative_diff_threshold=depth_pair_relative_diff_threshold,
        )
        per_frame.append({
            "frame": gf.frame,
            "geometric_consistency": gf.geometric_consistency,
            "disjoint": gf.disjoint,
            "details": gf.details,
        })

    return _aggregate_geometric_results(per_frame)


@torch.no_grad()
def compute_video_metrics(
    ctx: ModelContext,
    *,
    episode_dir: str,
    views: list[str],
    stride: int = 1,
    max_frames: int | None = None,
    include_per_frame: bool = False,
    conf_thres_percentile: float = 50.0,
    high_geo_score_disjoint_threshold: float = 0.99,
    depth_pair_dissimilar_fraction_min: float = 0.5,
    depth_pair_relative_diff_threshold: float = 0.25,
    frame_batch_size: int = 4,
) -> dict:
    frame_ids = _select_frame_ids(episode_dir, views, stride=stride, max_frames=max_frames)
    geo_frame_ids = [fi for fi in frame_ids if fi > 0]

    per_frame: list[dict] = []
    disjoint_frame_ids: set[int] = set()
    extrinsics_all: list[torch.Tensor] = []
    intrinsics_all: list[torch.Tensor] = []
    used_frame_ids: list[int] = []

    for fi, extrinsic, intrinsic, depth_map, depth_conf in _iter_vggt_frames(
        ctx,
        episode_dir=episode_dir,
        views=views,
        frame_ids=frame_ids,
        frame_batch_size=frame_batch_size,
    ):
        if fi > 0:
            gf = _process_geometric_frame(
                ctx,
                fi,
                extrinsic=extrinsic,
                intrinsic=intrinsic,
                depth_map=depth_map,
                depth_conf=depth_conf,
                conf_thres_percentile=conf_thres_percentile,
                high_geo_score_disjoint_threshold=high_geo_score_disjoint_threshold,
                depth_pair_dissimilar_fraction_min=depth_pair_dissimilar_fraction_min,
                depth_pair_relative_diff_threshold=depth_pair_relative_diff_threshold,
            )
            per_frame.append({
                "frame": gf.frame,
                "geometric_consistency": gf.geometric_consistency,
                "disjoint": gf.disjoint,
                "details": gf.details,
            })
            if gf.disjoint:
                disjoint_frame_ids.add(int(fi))

        if fi not in disjoint_frame_ids:
            extrinsics_all.append(extrinsic.detach().float().cpu())
            intrinsics_all.append(intrinsic.detach().float().cpu())
            used_frame_ids.append(int(fi))

    geo = _aggregate_geometric_results(per_frame)
    pose_stability = _pose_stability_from_tensors(extrinsics_all, intrinsics_all, used_frame_ids)

    out = {
        "episode_dir": str(episode_dir),
        "views": list(views),
        "stride": int(stride),
        "max_frames": None if max_frames is None else int(max_frames),
        "pose_stability": pose_stability,
        "geometric_consistency": geo["geometric_consistency"],
        "num_frames": int(min(len(frame_ids), len(geo_frame_ids))),
        "num_disjoint_frames": geo["num_disjoint_frames"],
        "disjoint_frame_ratio": geo["disjoint_frame_ratio"],
    }
    if include_per_frame:
        out["geometric_consistency_per_frame"] = geo["geometric_consistency_per_frame"]
    return out

