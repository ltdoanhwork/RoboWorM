import os
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterator

import torch
from vggt_omega.utils.geometry import unproject_depth_map_to_point_map
from .camera_pose_stability import compute_pose_stability_metric
from .geometric_consistency import (
    geometric_calculation,
    geometric_consistency_metric,
    render_depth_from_point_map,
    align_depth_scale,
)
from .utils import intersect_frames

def _configure_torch_for_inference() -> None:
    """Reduce inductor compile cost and enable TF32 matmul on CUDA."""
    os.environ.setdefault("CFLAGS", "-Wno-macro-redefined")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    try:
        import torch._inductor.config as inductor_config

        inductor_config.max_autotune = False
        inductor_config.max_autotune_gemm = False
    except Exception:
        pass

def _compile_model(model: torch.nn.Module, *, enabled: bool = True) -> torch.nn.Module:
    _configure_torch_for_inference()
    if not enabled:
        print("[video_metrics] torch.compile disabled", file=sys.stderr, flush=True)
        return model

    aggregator = getattr(model, "aggregator", None)
    if aggregator is None:
        return model

    try:
        model.aggregator = torch.compile(aggregator, mode="reduce-overhead")  # type: ignore[method-assign]
        print(
            "[video_metrics] torch.compile enabled on aggregator (mode=reduce-overhead)",
            file=sys.stderr,
            flush=True,
        )
    except Exception as exc:
        print(f"[video_metrics] torch.compile skipped: {exc}", file=sys.stderr, flush=True)
    return model

@dataclass(frozen=True)
class ModelContext:
    model: torch.nn.Module
    device: str = "cuda"
    autocast_dtype: torch.dtype | None = None
    omega_checkpoint: str | None = None
    omega_image_resolution: int = 512
    omega_patch_size: int = 16
    omega_load_fn: object | None = None
    omega_encoding_to_camera: object | None = None

def _load_vggt_omega_modules() -> tuple[object, object, object]:
    """
    Import in-tree VGGT-Omega from this package root only
    (``video_gen_physics_backup/vggt_omega``), never from repo-level ``vggt-omega/``.
    Returns:
      (VGGTOmega, load_and_preprocess_images_omega, encoding_to_camera)
    """
    import sys
    this_dir = os.path.dirname(os.path.abspath(__file__))
    package_root = os.path.abspath(os.path.join(this_dir, ".."))  # video_gen_physics_backup

    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    from vggt_omega.models import VGGTOmega  # type: ignore
    from vggt_omega.utils.load_fn import load_and_preprocess_images as load_and_preprocess_images_omega  # type: ignore
    from vggt_omega.utils.pose_enc import encoding_to_camera  # type: ignore
    return VGGTOmega, load_and_preprocess_images_omega, encoding_to_camera

def init_model_context(
    *,
    autocast_dtype: torch.dtype | None = None,
    omega_checkpoint: str | None = None,
    omega_image_resolution: int = 512,
    omega_patch_size: int = 16,
    enable_compile: bool = True,
) -> ModelContext:
    device = "cuda"
    if omega_checkpoint is None:
        raise ValueError("omega_checkpoint is required when backend='vggt_omega'")
    VGGTOmega, load_fn, encoding_to_camera = _load_vggt_omega_modules()

    model = VGGTOmega(patch_size=int(omega_patch_size)).to(device).eval()
    state = torch.load(str(omega_checkpoint), map_location="cpu")
    model.load_state_dict(state)
    model = _compile_model(model, enabled=bool(enable_compile))

    return ModelContext(
        model=model,
        device=device,
        autocast_dtype=autocast_dtype,
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
    
    load_fn_omega = ctx.omega_load_fn
    if load_fn_omega is None:
        _, load_fn_omega, _ = _load_vggt_omega_modules()
    preprocess = lambda paths: load_fn_omega(  # noqa: E731
        paths,
        image_resolution=int(ctx.omega_image_resolution),
        patch_size=int(ctx.omega_patch_size),
    )

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
      depth_map_indep: (B,S,H,W,1)
    """
    if images.ndim != 5:
        raise ValueError(f"Expected images (B,S,3,H,W), got {tuple(images.shape)}")

    encoding_to_camera = ctx.omega_encoding_to_camera
    if encoding_to_camera is None:
        _, _, encoding_to_camera = _load_vggt_omega_modules()
    preds = ctx.model(images)  # type: ignore[misc]
    pose_enc = preds["pose_enc"]
    depth_map = preds["depth"]
    depth_map_indep = preds["depth_single"]
    extrinsic, intrinsic = encoding_to_camera(pose_enc, preds["images"].shape[-2:])  # type: ignore[operator]
    return extrinsic, intrinsic, depth_map, depth_map_indep


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
    load_workers: int = 4
) -> Iterator[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Yield per-frame VGGT outputs using batch dimension B for multiple timesteps.

    Each yielded tensor has batch dim 1: (1,S,...).
    Yields: (frame_id, extrinsic, intrinsic, depth_map, images).

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

            images = images_cpu.pin_memory().to(ctx.device, non_blocking=True)


            extrinsic, intrinsic, depth_map, depth_map_indep = _forward_vggt(ctx, images)
            torch.cuda.synchronize()


            for b, fi in enumerate(chunk):
                yield (
                    int(fi),
                    extrinsic[b : b + 1],
                    intrinsic[b : b + 1],
                    depth_map[b : b + 1],
                    depth_map_indep[b : b + 1],
                    images[b : b + 1],
                )
    finally:
        if load_executor is not None:
            load_executor.shutdown(wait=True)


@dataclass(frozen=True)
class _GeometricFrameResult:
    frame: int
    geometric_consistency: float
    disjoint: bool
    details: dict | None


def _snapshot_frame_tensors(
    extrinsic: torch.Tensor,
    intrinsic: torch.Tensor,
    depth_map: torch.Tensor,
    depth_map_indep: torch.Tensor,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """CPU copies for async geometric post-processing (GPU tensors may be reused)."""
    return (
        extrinsic.detach().float().cpu(),
        intrinsic.detach().float().cpu(),
        depth_map.detach().float().cpu(),
        depth_map_indep.detach().float().cpu(),
        images.detach().float().cpu(),
    )

def _collect_geometric_result(
    fi: int,
    gf: _GeometricFrameResult,
    extrinsic_cpu: torch.Tensor,
    intrinsic_cpu: torch.Tensor,
    *,
    per_frame: list[dict],
    pose_buffer: list[tuple[int, torch.Tensor, torch.Tensor]],
) -> None:
    per_frame.append({
        "frame": gf.frame,
        "geometric_consistency": gf.geometric_consistency,
        "disjoint": gf.disjoint,
        "details": gf.details,
    })
    if not gf.disjoint:
        pose_buffer.append((int(fi), extrinsic_cpu, intrinsic_cpu))


def _drain_geometric_futures(
    pending: list[tuple[int, Future, torch.Tensor, torch.Tensor]],
    *,
    per_frame: list[dict],
    pose_buffer: list[tuple[int, torch.Tensor, torch.Tensor]],
    block: bool,
) -> list[tuple[int, Future, torch.Tensor, torch.Tensor]]:
    remaining: list[tuple[int, Future, torch.Tensor, torch.Tensor]] = []
    for fi, fut, extrinsic_cpu, intrinsic_cpu in pending:
        if block or fut.done():
            gf = fut.result()
            _collect_geometric_result(
                fi, gf, extrinsic_cpu, intrinsic_cpu,
                per_frame=per_frame, pose_buffer=pose_buffer,
            )
        else:
            remaining.append((fi, fut, extrinsic_cpu, intrinsic_cpu))
    return remaining


@torch.no_grad()
def _process_geometric_frame(
    fi: int,
    *,
    extrinsic: torch.Tensor,
    intrinsic: torch.Tensor,
    depth_map: torch.Tensor,
    depth_map_indep: torch.Tensor,
    images: torch.Tensor
) -> _GeometricFrameResult:
    """
    Fixed layouts:
      images/depth/extrinsic/intrinsic: (1, S, ...)
      point_map: (S, H, W, 3)
      depth_rendered / valid_mask: (1, S, H, W)
      rgb_rendered: (1, S, H, W, 3)
    """
    if depth_map.is_cuda:
        depth_cpu = depth_map.detach().float().cpu()
        depth_cpu_indep = depth_map_indep.detach().float().cpu()
        extrinsic_cpu = extrinsic.detach().float().cpu()
        intrinsic_cpu = intrinsic.detach().float().cpu()
        images_cpu = images.detach().float().cpu()
    else:
        depth_cpu = depth_map.detach().float()
        depth_cpu_indep = depth_map_indep.detach().float()
        extrinsic_cpu = extrinsic.detach().float()
        intrinsic_cpu = intrinsic.detach().float()
        images_cpu = images.detach().float()

    depth_map_indep_aligned = align_depth_scale(depth_cpu_indep, depth_cpu, max_depth=1.0)

    point_map = unproject_depth_map_to_point_map(
        depth_map_indep_aligned[0], extrinsic_cpu.squeeze(0), intrinsic_cpu.squeeze(0)
    )

    depth_rendered_np, rendered_valid_mask, rgb_rendered_np = render_depth_from_point_map(
        point_map_by_unprojection=point_map,
        depth_map=depth_map_indep_aligned,
        extrinsic=extrinsic_cpu,
        intrinsic=intrinsic_cpu,
        images=images_cpu,
    )
    res = geometric_calculation(
        depth=depth_map_indep_aligned,
        depth_rendered=depth_rendered_np,
        images=images_cpu,
        rgb_rendered=rgb_rendered_np,
        valid_mask=rendered_valid_mask,
    )
    score = float(geometric_consistency_metric(res))
    if res["view0_shape"]["valid_ratio"] < 0.02 or res["view1_shape"]["valid_ratio"] < 0.02:
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
    # scores = [float(pf["geometric_consistency"]) for pf in per_frame if not pf["disjoint"]]
    scores = [float(pf["geometric_consistency"]) for pf in per_frame]
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
    frame_batch_size: int = 4,
    prefetch_chunks: bool = True,
    parallel_frame_load: bool = True,
    load_workers: int = 4
) -> dict:
    frame_ids = intersect_frames(list(views), episode_dir)
    if not frame_ids:
        raise ValueError(f"No common frames found in {episode_dir} for views={views}")
    geo_frame_ids = [fi for fi in frame_ids if fi > 0]

    per_frame: list[dict] = []
    pose_buffer: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    pending_geo: list[tuple[int, Future, torch.Tensor, torch.Tensor]] = []

    geo_executor = ThreadPoolExecutor(max_workers=2)
    try:
        for fi, extrinsic, intrinsic, depth_map, depth_map_indep, images in _iter_vggt_frames(
            ctx,
            episode_dir=episode_dir,
            views=views,
            frame_ids=frame_ids,
            frame_batch_size=frame_batch_size,
            prefetch_chunks=prefetch_chunks,
            parallel_frame_load=parallel_frame_load,
            load_workers=load_workers,
        ):
            pending_geo = _drain_geometric_futures(
                pending_geo,
                per_frame=per_frame,
                pose_buffer=pose_buffer,
                block=False,
            )

            if fi == 0:
                ext0, intr0, _, _, _ = _snapshot_frame_tensors(
                    extrinsic, intrinsic, depth_map, depth_map_indep, images,
                )
                pose_buffer.append((0, ext0, intr0))
            else:
                ext_cpu, intr_cpu, d_cpu, d_ind_cpu, img_cpu = _snapshot_frame_tensors(
                    extrinsic, intrinsic, depth_map, depth_map_indep, images,
                )
                pending_geo.append((
                    int(fi),
                    geo_executor.submit(
                        _process_geometric_frame,
                        fi,
                        extrinsic=ext_cpu,
                        intrinsic=intr_cpu,
                        depth_map=d_cpu,
                        depth_map_indep=d_ind_cpu,
                        images=img_cpu,
                    ),
                    ext_cpu,
                    intr_cpu,
                ))

        while pending_geo:
            pending_geo = _drain_geometric_futures(
                pending_geo,
                per_frame=per_frame,
                pose_buffer=pose_buffer,
                block=True,
            )
    finally:
        geo_executor.shutdown(wait=True)

    pose_buffer.sort(key=lambda item: item[0])
    extrinsics_all = [ext for _, ext, _ in pose_buffer]
    intrinsics_all = [intr for _, _, intr in pose_buffer]
    used_frame_ids = [fi for fi, _, _ in pose_buffer]

    geo = _aggregate_geometric_results(per_frame)
    pose_stability = 0.0 if (not used_frame_ids or used_frame_ids == [0]) else float(compute_pose_stability_metric(extrinsics_all, intrinsics_all))

    out = {
        "episode_dir": str(episode_dir),
        "views": list(views),
        "pose_stability": pose_stability,
        "geometric_consistency": geo["geometric_consistency"],
        "num_frames": int(min(len(frame_ids), len(geo_frame_ids))),
        "num_disjoint_frames": geo["num_disjoint_frames"],
        "disjoint_frame_ratio": geo["disjoint_frame_ratio"],
    }
    return out

