import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from vggt.utils.geometry import unproject_depth_map_to_point_map
from .camera_pose_stability import compute_pose_stability_metric
from .geometric_consistency import (
    geometric_calculation,
    geometric_consistency_metric,
    render_depth_from_point_map,
)
from .utils import intersect_frames

@dataclass(frozen=True)
class ModelContext:
    # VGGT (legacy) or VGGT-Omega (new).
    model: object
    device: str
    autocast_dtype: torch.dtype
    backend: str = "vggt"  # "vggt" | "vggt_omega"
    omega_checkpoint: str | None = None
    omega_image_resolution: int = 512
    omega_patch_size: int = 16
    # Cached callables from ``_load_vggt_omega_modules`` (avoid re-import per frame).
    omega_load_fn: object | None = None
    omega_encoding_to_camera: object | None = None


def _load_vggt_omega_modules() -> tuple[object, object, object]:
    """
    Import VGGT-Omega modules without requiring it to be installed as a package.

    Returns:
      (VGGTOmega, load_and_preprocess_images_omega, encoding_to_camera)
    """
    import sys

    this_dir = os.path.dirname(os.path.abspath(__file__))
    wmb_vggt_root = os.path.abspath(os.path.join(this_dir, ".."))
    wmb_root = os.path.abspath(os.path.join(wmb_vggt_root, ".."))
    # Prefer in-tree ``vggt_omega`` under video_gen_physics(_backup), then repo ``vggt-omega/``.
    for omega_root in (wmb_vggt_root, os.path.join(wmb_root, "vggt-omega")):
        if os.path.isdir(os.path.join(omega_root, "vggt_omega")) and omega_root not in sys.path:
            sys.path.insert(0, omega_root)
            break

    from vggt_omega.models import VGGTOmega  # type: ignore
    from vggt_omega.utils.load_fn import load_and_preprocess_images as load_and_preprocess_images_omega  # type: ignore
    from vggt_omega.utils.pose_enc import encoding_to_camera  # type: ignore

    return VGGTOmega, load_and_preprocess_images_omega, encoding_to_camera


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
    backend: str = "vggt",
    omega_checkpoint: str | None = None,
    omega_image_resolution: int = 512,
    omega_patch_size: int = 16,
) -> ModelContext:
    if device is None or autocast_dtype is None:
        d, dt = default_device_dtype()
        device = d if device is None else device
        autocast_dtype = dt if autocast_dtype is None else autocast_dtype

    backend = str(backend).strip().lower()
    if backend not in {"vggt", "vggt_omega"}:
        raise ValueError(f"Unknown backend={backend!r}; expected 'vggt' or 'vggt_omega'")

    if backend == "vggt":
        from vggt.models.vggt import VGGT  # local import: avoids huggingface_hub dependency when using omega
        model = VGGT.from_pretrained(model_name).to(device)
        model.eval()
        return ModelContext(model=model, device=device, autocast_dtype=autocast_dtype, backend="vggt")

    # VGGT-Omega: load local checkpoint path.
    if omega_checkpoint is None:
        raise ValueError("omega_checkpoint is required when backend='vggt_omega'")
    VGGTOmega, load_fn, encoding_to_camera = _load_vggt_omega_modules()

    model = VGGTOmega(patch_size=int(omega_patch_size)).to(device).eval()
    state = torch.load(str(omega_checkpoint), map_location="cpu")
    model.load_state_dict(state)

    return ModelContext(
        model=model,
        device=device,
        autocast_dtype=autocast_dtype,
        backend="vggt_omega",
        omega_checkpoint=str(omega_checkpoint),
        omega_image_resolution=int(omega_image_resolution),
        omega_patch_size=int(omega_patch_size),
        omega_load_fn=load_fn,
        omega_encoding_to_camera=encoding_to_camera,
    )


def _frame_image_paths(episode_dir: str, views: list[str], fi: int) -> list[str]:
    return [os.path.join(episode_dir, v, f"frame_{fi:04d}.png") for v in views]


def _chunk_list(items: list[int], chunk_size: int) -> list[list[int]]:
    size = max(1, int(chunk_size))
    return [items[i : i + size] for i in range(0, len(items), size)]


def _load_images_chunk_cpu(
    ctx: ModelContext,
    episode_dir: str,
    views: list[str],
    frame_ids: list[int],
    *,
    parallel_frames: bool = True,
    load_workers: int = 4,
) -> torch.Tensor:
    """
    Load a chunk on CPU as (B, S, 3, H, W). Uses ``load_and_preprocess_images`` unchanged.

    With ``parallel_frames``, each timestep calls ``load_and_preprocess_images`` on its
    view paths in a thread pool (helps NFS). Otherwise one batched call for all paths.
    """
    if not frame_ids:
        raise ValueError("frame_ids must not be empty")

    s = len(views)
    if ctx.backend == "vggt_omega":
        load_fn_omega = ctx.omega_load_fn
        if load_fn_omega is None:
            _, load_fn_omega, _ = _load_vggt_omega_modules()
        preprocess = lambda paths: load_fn_omega(  # noqa: E731
            paths,
            image_resolution=int(ctx.omega_image_resolution),
            patch_size=int(ctx.omega_patch_size),
        )
    else:
        from vggt.utils.load_fn import load_and_preprocess_images  # local import
        preprocess = load_and_preprocess_images

    if parallel_frames and len(frame_ids) > 1:
        workers = max(1, min(int(load_workers), len(frame_ids)))

        def _load_one_frame(fi: int) -> torch.Tensor:
            paths = _frame_image_paths(episode_dir, views, fi)
            return preprocess(paths)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            frame_tensors = list(pool.map(_load_one_frame, frame_ids))
        return torch.stack(frame_tensors, dim=0)

    paths: list[str] = []
    for fi in frame_ids:
        paths.extend(_frame_image_paths(episode_dir, views, fi))
    flat = preprocess(paths)
    b = len(frame_ids)
    return flat.view(b, s, *flat.shape[1:])


def _images_cpu_to_device(images_cpu: torch.Tensor, device: str) -> torch.Tensor:
    if device == "cuda" and torch.cuda.is_available():
        return images_cpu.pin_memory().to(device, non_blocking=True)
    return images_cpu.to(device)

@torch.no_grad()
def _forward_vggt(
    ctx: ModelContext,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
      images: (B, S, 3, H, W)

    Returns:
      extrinsic: (B,S,3,4)
      intrinsic: (B,S,3,3)
      depth_map: (B,S,H,W,1)
    """
    if images.ndim != 5:
        raise ValueError(f"Expected images (B,S,3,H,W), got {tuple(images.shape)}")

    if ctx.backend == "vggt_omega":
        encoding_to_camera = ctx.omega_encoding_to_camera
        if encoding_to_camera is None:
            _, _, encoding_to_camera = _load_vggt_omega_modules()
        preds = ctx.model(images)  # type: ignore[misc]
        pose_enc = preds["pose_enc"]
        depth_map = preds["depth"]
        extrinsic, intrinsic = encoding_to_camera(pose_enc, preds["images"].shape[-2:])  # type: ignore[operator]
        return extrinsic, intrinsic, depth_map

    # legacy VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # local import
    if ctx.device == "cuda":
        with torch.cuda.amp.autocast(dtype=ctx.autocast_dtype):
            aggregated_tokens_list, ps_idx = ctx.model.aggregator(images)  # type: ignore[assignment]
            pose_enc = ctx.model.camera_head(aggregated_tokens_list)[-1]  # type: ignore[union-attr]
            depth_map, _ = ctx.model.depth_head(aggregated_tokens_list, images, ps_idx)  # type: ignore[union-attr]
    else:
        aggregated_tokens_list, ps_idx = ctx.model.aggregator(images)  # type: ignore[assignment]
        pose_enc = ctx.model.camera_head(aggregated_tokens_list)[-1]  # type: ignore[union-attr]
        depth_map, _ = ctx.model.depth_head(aggregated_tokens_list, images, ps_idx)  # type: ignore[union-attr]

    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    return extrinsic, intrinsic, depth_map

@torch.no_grad()
def _iter_vggt_frames(
    ctx: ModelContext,
    *,
    episode_dir: str,
    views: list[str],
    frame_ids: list[int],
    frame_batch_size: int = 4,
    prefetch_chunks: bool = True,
    parallel_frame_load: bool = True,
    load_workers: int = 4,
) -> Iterator[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Yield per-frame VGGT outputs using batch dimension B for multiple timesteps.

    Each yielded tensor has batch dim 1: (1,S,...).

    When ``prefetch_chunks`` is True, the next chunk is loaded on a background thread
    while the current chunk is forwarded on GPU (overlaps I/O with compute).
    """
    chunks = _chunk_list(frame_ids, frame_batch_size)
    if not chunks:
        return

    load_executor: ThreadPoolExecutor | None = None
    pending_load: Future[torch.Tensor] | None = None

    try:
        if prefetch_chunks:
            load_executor = ThreadPoolExecutor(max_workers=1)
            pending_load = load_executor.submit(
                _load_images_chunk_cpu,
                ctx,
                episode_dir,
                views,
                chunks[0],
                parallel_frames=parallel_frame_load,
                load_workers=load_workers,
            )

        for chunk_idx, chunk in enumerate(chunks):
            if prefetch_chunks and pending_load is not None:
                images_cpu = pending_load.result()
                if chunk_idx + 1 < len(chunks):
                    pending_load = load_executor.submit(  # type: ignore[union-attr]
                        _load_images_chunk_cpu,
                        ctx,
                        episode_dir,
                        views,
                        chunks[chunk_idx + 1],
                        parallel_frames=parallel_frame_load,
                        load_workers=load_workers,
                    )
                else:
                    pending_load = None
            else:
                images_cpu = _load_images_chunk_cpu(
                    ctx,
                    episode_dir,
                    views,
                    chunk,
                    parallel_frames=parallel_frame_load,
                    load_workers=load_workers,
                )

            images = _images_cpu_to_device(images_cpu, ctx.device)
            extrinsic, intrinsic, depth_map = _forward_vggt(ctx, images)

            for b, fi in enumerate(chunk):
                yield (
                    int(fi),
                    extrinsic[b : b + 1],
                    intrinsic[b : b + 1],
                    depth_map[b : b + 1]
                )
    finally:
        if load_executor is not None:
            load_executor.shutdown(wait=True)


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

@dataclass(frozen=True)
class _GeometricFrameResult:
    frame: int
    geometric_consistency: float
    disjoint: bool
    details: dict | None

@torch.no_grad()
def _process_geometric_frame(
    fi: int,
    *,
    extrinsic: torch.Tensor,
    intrinsic: torch.Tensor,
    depth_map: torch.Tensor,
) -> _GeometricFrameResult:
    # Keep the entire geometric post-process on CPU (avoid GPU round-trips).
    depth_cpu = depth_map.detach().float().cpu()
    extrinsic_cpu = extrinsic.detach().float().cpu()
    intrinsic_cpu = intrinsic.detach().float().cpu()

    point_map = unproject_depth_map_to_point_map(
        depth_cpu.squeeze(0), extrinsic_cpu.squeeze(0), intrinsic_cpu.squeeze(0)
    )
    depth_rendered_np, rendered_valid_mask, depth_spread = render_depth_from_point_map(
        point_map_by_unprojection=point_map,
        depth_map=depth_cpu,
        extrinsic=extrinsic_cpu,
        intrinsic=intrinsic_cpu,
    )

    if depth_rendered_np.ndim == 3:
        depth_rendered_np = depth_rendered_np[np.newaxis, ...]
    if isinstance(depth_spread, np.ndarray) and depth_spread.ndim == 3:
        depth_spread = depth_spread[np.newaxis, ...]

    valid_for_metric = rendered_valid_mask
    if isinstance(valid_for_metric, np.ndarray) and valid_for_metric.ndim == 3:
        valid_for_metric = valid_for_metric[np.newaxis, ...]

    res = geometric_calculation(
        depth=depth_cpu,
        depth_rendered=depth_rendered_np,
        valid_mask=valid_for_metric,
        depth_spread=depth_spread,
    )
    score = float(geometric_consistency_metric(res))
    if res['view0_shape']['valid_ratio'] < 0.02 or res['view1_shape']['valid_ratio'] < 0.02:
        return _GeometricFrameResult(
            frame=int(fi),
            geometric_consistency=0.0,
            disjoint=True,
            details=res,
        )

    return _GeometricFrameResult(
        frame=int(fi),
        geometric_consistency=score,
        disjoint=False,
        details=res,
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

@torch.no_grad()
def compute_video_metrics(
    ctx: ModelContext,
    *,
    episode_dir: str,
    views: list[str],
    stride: int = 1,
    max_frames: int | None = None,
    include_per_frame: bool = False,
    frame_batch_size: int = 4,
    prefetch_chunks: bool = True,
    parallel_frame_load: bool = True,
    load_workers: int = 1,
) -> dict:
    frame_ids = _select_frame_ids(episode_dir, views, stride=stride, max_frames=max_frames)
    geo_frame_ids = [fi for fi in frame_ids if fi > 0]

    per_frame: list[dict] = []
    disjoint_frame_ids: set[int] = set()
    extrinsics_all: list[torch.Tensor] = []
    intrinsics_all: list[torch.Tensor] = []
    used_frame_ids: list[int] = []

    for fi, extrinsic, intrinsic, depth_map in _iter_vggt_frames(
        ctx,
        episode_dir=episode_dir,
        views=views,
        frame_ids=frame_ids,
        frame_batch_size=frame_batch_size,
        prefetch_chunks=prefetch_chunks,
        parallel_frame_load=parallel_frame_load,
        load_workers=load_workers,
    ):
        if fi > 0:
            gf = _process_geometric_frame(
                fi,
                extrinsic=extrinsic,
                intrinsic=intrinsic,
                depth_map=depth_map,
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
    pose_stability = 0.0 if (not used_frame_ids or used_frame_ids == [0]) else float(compute_pose_stability_metric(extrinsics_all, intrinsics_all))
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