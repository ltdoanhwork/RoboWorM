import torch
import math                        # (H,W)
import numpy as np


def _images_to_rgb_maps(images: torch.Tensor | np.ndarray) -> np.ndarray:
    """Normalize images to (S, H, W, 3) float in [0, 1]."""
    if isinstance(images, torch.Tensor):
        img = images.detach().cpu().float().numpy()
    else:
        img = np.asarray(images, dtype=np.float32)
    if img.ndim == 5:  # (B,S,3,H,W)
        img = img[0]
    if img.ndim == 4 and img.shape[1] == 3:  # (S,3,H,W)
        img = np.transpose(img, (0, 2, 3, 1))
    if img.ndim != 4 or img.shape[-1] != 3:
        raise ValueError(f"images must become (S,H,W,3), got {img.shape}")
    if img.max() > 1.5:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0).astype(np.float32)


@torch.no_grad()
def render_depth_from_point_map(
    point_map_by_unprojection: np.ndarray,
    depth_map: torch.Tensor | np.ndarray,
    extrinsic: torch.Tensor | np.ndarray,
    intrinsic: torch.Tensor | np.ndarray,
    images: torch.Tensor | np.ndarray | None = None,
) -> tuple:
    """
    Project multi-view unprojected points into each camera (z-buffer).

    Same idea as demo_gradio / visual_util coloring points from `images`, then
    splat into each target view keeping the nearest point's depth (and RGB).

    Returns:
      depth_rendered: (B,S,H,W) or (S,H,W) float32
      fov_overlap_mask: same layout, bool
      rgb_rendered (optional): (B,S,H,W,3) or (S,H,W,3) float32 in [0,1]
        only when ``images`` is provided.
    """
    if point_map_by_unprojection.ndim != 4 or point_map_by_unprojection.shape[-1] != 3:
        raise ValueError(f"point_map_by_unprojection must be (S,H,W,3), got {point_map_by_unprojection.shape}")

    S_pm, H, W, _ = point_map_by_unprojection.shape

    if isinstance(depth_map, torch.Tensor):
        d = depth_map.detach().cpu().float()
        if d.ndim == 5:
            d = d.squeeze(-1)
        d0 = d[0].numpy() if d.ndim == 4 else d.numpy()
    else:
        d0 = depth_map
        if d0.ndim == 4:
            d0 = d0[0]
        if d0.ndim == 4 and d0.shape[-1] == 1:
            d0 = d0[..., 0]
        if d0.ndim == 3 and d0.shape[-1] == 1:
            d0 = d0[..., 0]

    if d0.shape[:3] != (S_pm, H, W):
        raise ValueError(f"depth_map (first batch) must match (S,H,W)={(S_pm,H,W)}, got {d0.shape}")

    rgb_maps = None
    if images is not None:
        rgb_maps = _images_to_rgb_maps(images)
        if rgb_maps.shape[:3] != (S_pm, H, W):
            raise ValueError(
                f"images spatial shape must match depth (S,H,W)={(S_pm, H, W)}, got {rgb_maps.shape}"
            )

    def _to_np(x):
        return x.detach().cpu().float().numpy() if isinstance(x, torch.Tensor) else x

    E = _to_np(extrinsic)
    K = _to_np(intrinsic)
    batched = (E.ndim == 4)
    if E.ndim == 3:
        E = E[None]
    if K.ndim == 3:
        K = K[None]
    B = E.shape[0]
    S_cam = E.shape[1]

    pts = []
    cols = []
    for s in range(S_pm):
        ds_ = d0[s]
        m = np.isfinite(ds_) & (ds_ > 1e-6)
        if np.any(m):
            pts.append(point_map_by_unprojection[s][m])
            if rgb_maps is not None:
                cols.append(rgb_maps[s][m])

    out = np.zeros((B, S_cam, H, W), dtype=np.float32)
    fov_overlap_geom = np.zeros((B, S_cam, H, W), dtype=bool)
    rgb_out = None
    if rgb_maps is not None:
        rgb_out = np.zeros((B, S_cam, H, W, 3), dtype=np.float32)

    if len(pts) > 0:
        P = np.concatenate(pts, axis=0).astype(np.float32)
        Pwh = np.concatenate([P, np.ones((P.shape[0], 1), dtype=np.float32)], axis=1)
        C = np.concatenate(cols, axis=0).astype(np.float32) if cols else None

        for b in range(B):
            for s in range(S_cam):
                Eb = E[b, s].astype(np.float32)
                Kb = K[b, s].astype(np.float32)

                Pc = (Eb @ Pwh.T).T
                z = Pc[:, 2]
                valid = z > 0.01
                Pc_v = Pc[valid]
                z_v = z[valid]
                C_v = C[valid] if C is not None else None

                if z_v.size > 0:
                    pix = (Kb @ Pc_v.T).T
                    u = pix[:, 0] / (pix[:, 2] + 1e-5)
                    v = pix[:, 1] / (pix[:, 2] + 1e-5)

                    ui_all = np.rint(u)
                    vi_all = np.rint(v)
                    inb = (ui_all >= 0) & (ui_all < W) & (vi_all >= 0) & (vi_all < H) & np.isfinite(ui_all) & np.isfinite(vi_all)

                    ui_b = ui_all[inb].astype(np.int32)
                    vi_b = vi_all[inb].astype(np.int32)
                    z_b = z_v[inb].astype(np.float32)
                    C_b = C_v[inb] if C_v is not None else None

                    if z_b.size > 0:
                        depth_min_buf = np.full((H, W), np.inf, dtype=np.float32)
                        np.minimum.at(depth_min_buf, (vi_b, ui_b), z_b)

                        hit_count = np.zeros((H, W), dtype=np.int32)
                        np.add.at(hit_count, (vi_b, ui_b), 1)

                        hit = hit_count > 0
                        depth_min_buf[~hit] = 0.0
                        out[b, s] = depth_min_buf

                        if C_b is not None and rgb_out is not None:
                            order = np.argsort(-z_b)
                            flat = vi_b[order].astype(np.int64) * W + ui_b[order].astype(np.int64)
                            rgb_flat = rgb_out[b, s].reshape(-1, 3)
                            rgb_flat[flat] = C_b[order]

                pm_s = point_map_by_unprojection[s].reshape(-1, 3).astype(np.float32)
                Pwh_s = np.concatenate(
                    [pm_s, np.ones((pm_s.shape[0], 1), dtype=np.float32)], axis=1
                )
                acc_geom = np.zeros(H * W, dtype=bool)
                for src in range(S_cam):
                    if src == s:
                        continue
                    Esrc = E[b, src].astype(np.float32)
                    Ksrc = K[b, src].astype(np.float32)

                    Pc_g = (Esrc @ Pwh_s.T).T
                    z_g = Pc_g[:, 2]
                    pix_g = (Ksrc @ Pc_g.T).T
                    u_g = pix_g[:, 0] / (pix_g[:, 2] + 1e-6)
                    v_g = pix_g[:, 1] / (pix_g[:, 2] + 1e-6)

                    in_front = z_g > 0.01
                    in_bounds = (u_g >= 0) & (u_g < W) & (v_g >= 0) & (v_g < H)
                    overlap_flat = in_front & in_bounds & np.isfinite(u_g) & np.isfinite(v_g)
                    acc_geom |= overlap_flat

                fov_overlap_geom[b, s] = acc_geom.reshape(H, W)

    if not batched:
        out, fov_overlap_geom = out[0], fov_overlap_geom[0]
        if rgb_out is not None:
            rgb_out = rgb_out[0]

    if rgb_out is not None:
        return out, fov_overlap_geom, rgb_out
    return out, fov_overlap_geom

def _to_rgb_bshw3(rgb: torch.Tensor | np.ndarray, *, name: str) -> np.ndarray:
    """Normalize RGB to (B,S,H,W,3) float in [0, 1]."""
    if isinstance(rgb, torch.Tensor):
        arr = rgb.detach().cpu().float().numpy()
    else:
        arr = np.asarray(rgb, dtype=np.float32)

    if arr.ndim == 4 and arr.shape[-1] == 3:
        # (S,H,W,3) or (B,H,W,3) — treat as S views, add batch
        arr = arr[None, ...]
    elif arr.ndim == 4 and arr.shape[1] == 3:
        # (S,3,H,W)
        arr = np.transpose(arr, (0, 2, 3, 1))[None, ...]
    elif arr.ndim == 5 and arr.shape[2] == 3:
        # (B,S,3,H,W)
        arr = np.transpose(arr, (0, 1, 3, 4, 2))
    elif arr.ndim == 5 and arr.shape[-1] == 3:
        # (B,S,H,W,3)
        pass
    else:
        raise ValueError(f"{name} must become (B,S,H,W,3), got {arr.shape}")

    if arr.max() > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def geometric_calculation(
    depth: torch.Tensor | np.ndarray,
    depth_rendered: torch.Tensor | np.ndarray,
    *,
    eps: float = 1e-6,
    valid_mask: torch.Tensor | np.ndarray | None = None,
    images: torch.Tensor | np.ndarray | None = None,
    rgb_rendered: torch.Tensor | np.ndarray | None = None,
) -> dict:
    """
    Inputs:
      depth_joint, depth_indep:
        - (B,S,H,W,1) or (B,S,H,W) or (S,H,W,1) or (S,H,W)
        - Typically B==1, S==2 in your pipeline.
      valid_mask:
        - Optional; same rank/layout as depths after batching: (B,S,H,W) or (S,H,W), bool or 0/1.
        - Non-boolean arrays are cast to bool (non-zero is True).
      images, rgb_rendered:
        - Optional RGB pair for photometric consistency (source vs cross-view rendered).
        - Layouts: (B,S,3,H,W), (B,S,H,W,3), (S,3,H,W), or (S,H,W,3); values in [0,1] or [0,255].

    Returns:
      {
        "view0_shape": {...},  # may include "rgb_rmse"
        "view1_shape": {...},
        "rgb_rmse": float,     # only when images & rgb_rendered are provided
      }
      If S>2, keys are view{idx}_shape / valid_ratio_view{idx}.
    """
    def _to_np_mask(m):
        if isinstance(m, torch.Tensor):
            m = m.detach().cpu().numpy()
        m = np.asarray(m)
        if m.ndim == 5:
            m = m[..., 0]
        elif m.ndim == 4 and m.shape[-1] == 1:
            m = m[..., 0]
        if m.dtype != np.bool_:
            m = m.astype(bool)
        return m
    
    def _to_np_depth(d):
        if isinstance(d, torch.Tensor):
            d = d.detach().cpu().float().numpy()
        d = np.asarray(d)
        if d.ndim == 5:
            d = d[..., 0]  # (B,S,H,W)
        elif d.ndim == 4 and d.shape[-1] == 1:
            d = d[..., 0]  # (S,H,W) case stored as (S,H,W,1)
        return d

    dj = _to_np_depth(depth)
    di = _to_np_depth(depth_rendered)

    vm = None
    if valid_mask is not None:
        vm = _to_np_mask(valid_mask)

    # Normalize to (B,S,H,W)
    if dj.ndim == 3:
        dj = dj[None, ...]
    if di.ndim == 3:
        di = di[None, ...]
    if dj.ndim != 4 or di.ndim != 4:
        raise ValueError(f"Expected depth tensors to become (B,S,H,W); got {dj.shape} and {di.shape}")
    if dj.shape != di.shape:
        raise ValueError(f"depth and depth_rendered shape mismatch: {dj.shape} vs {di.shape}")

    if vm is not None:
        if vm.ndim == 3:
            vm = vm[None, ...]
        if vm.ndim != 4:
            raise ValueError(f"valid_mask must become (B,S,H,W); got {vm.shape} after batching")
        if vm.shape != dj.shape:
            raise ValueError(f"valid_mask shape {vm.shape} must match depth shape {dj.shape}")

    B, S, H, W = dj.shape

    rgb_src = None
    rgb_ren = None
    if images is not None or rgb_rendered is not None:
        if images is None or rgb_rendered is None:
            raise ValueError("images and rgb_rendered must be provided together")
        rgb_src = _to_rgb_bshw3(images, name="images")
        rgb_ren = _to_rgb_bshw3(rgb_rendered, name="rgb_rendered")
        if rgb_src.shape != (B, S, H, W, 3):
            raise ValueError(f"images shape {rgb_src.shape} must match depth (B,S,H,W,3)={(B,S,H,W,3)}")
        if rgb_ren.shape != rgb_src.shape:
            raise ValueError(f"rgb_rendered shape {rgb_ren.shape} must match images {rgb_src.shape}")

    def _shape_stats_1view(
        y_bhw: np.ndarray,
        x_bhw: np.ndarray,
        m_bhw: np.ndarray | None,
        rgb_src_bhw3: np.ndarray | None = None,
        rgb_ren_bhw3: np.ndarray | None = None,
    ) -> dict | list[dict]:
        # y: joint, x: indep, both (B,H,W); optional m_bhw (B,H,W) bool
        out = []
        for b in range(B):
            y = y_bhw[b]
            x = x_bhw[b]
            roi = np.isfinite(y) & np.isfinite(x) & (y > eps) & (x > eps)
            if m_bhw is not None:
                roi = roi & m_bhw[b]
            cnt = int(roi.sum())
            if cnt == 0:
                out.append({
                    "residual": float("nan"), "count": 0, "scale_ref": float("nan"),
                    "valid_ratio": float(roi.mean()),
                    "rgb_rmse": float("nan"),
                })
                continue

            xv = x.astype(np.float64)
            yv = y.astype(np.float64)

            scale_ref = float(np.median(yv))

            r = np.abs(yv - xv)
            rgb_rmse = float("nan")
            if rgb_src_bhw3 is not None and rgb_ren_bhw3 is not None:
                src_pix = rgb_src_bhw3[b].astype(np.float64)
                ren_pix = rgb_ren_bhw3[b].astype(np.float64)
                if src_pix.size > 0:
                    rgb_rmse = float(np.sqrt(np.mean(np.square(src_pix - ren_pix))))
                else:
                    rgb_rmse = 0.0

            out.append({
                # "residual": float(r.mean()),
                # "residual": float(np.sqrt(np.mean(r * r))), # residual_rms
                # "residual": float(np.median(r)), # residual_median
                "residual": float(np.percentile(r, 90.0)), # residual_p90
                # "residual": float(np.percentile(r, 95.0)), # residual_p95
                "count": cnt,
                "scale_ref": scale_ref,
                "valid_ratio": float(roi.mean()),
                "rgb_rmse": rgb_rmse,
            })

        return out[0] if B == 1 else out

    result: dict = {}
    for s in range(S):
        m_slice = vm[:, s] if vm is not None else None
        rgb_src_s = rgb_src[:, s] if rgb_src is not None else None
        rgb_ren_s = rgb_ren[:, s] if rgb_ren is not None else None
        shape_s = _shape_stats_1view(
            dj[:, s], di[:, s], m_slice, rgb_src_s, rgb_ren_s
        )
        result[f"view{s}_shape"] = shape_s

    if rgb_src is not None and rgb_ren is not None:
        diffs = []
        for s in range(S):
            for b in range(B):
                y = dj[b, s]
                x = di[b, s]
                roi = np.isfinite(y) & np.isfinite(x) & (y > eps) & (x > eps)
                diffs.append(rgb_src[b, s][roi] - rgb_ren[b, s][roi])
        if diffs:
            all_diff = np.concatenate(diffs, axis=0).astype(np.float64)
            result["rgb_rmse"] = float(np.sqrt(np.mean(np.square(all_diff))))
        else:
            result["rgb_rmse"] = 0.0

    return result

def geometric_consistency_metric(
    res: dict,
) -> float:
    """
    Returns scalar in [0,1], higher is better.

    When per-view ``rgb_rmse`` is present, multiplies each view score by
    ``exp(-rgb_rmse)`` (photometric consistency; lower RMSE → higher score).
    """
    def _view_score(vshape: dict) -> float:
        resid = float(vshape.get("residual", float("nan")))
        scale_ref = float(vshape.get("scale_ref", float("nan")))
        if not (math.isfinite(resid) and math.isfinite(scale_ref) and scale_ref > 1e-6):
            return 0.0

        rel_resid = resid / scale_ref
        s_resid = max(0.0, 1 - rel_resid/0.5)

        rgb_rmse = vshape.get("rgb_rmse", float("nan"))
        if math.isfinite(rgb_rmse):
            s_rgb = max(0.0, 1 - rgb_rmse)
        else:
            s_rgb = 1.0
        score = (s_resid + s_rgb) / 2.0
        return score

    v0 = res.get("view0_shape", {})
    v1 = res.get("view1_shape", {})
    s0 = _view_score(v0)
    s1 = _view_score(v1)
    base = s0 * s1 

    return float(max(0.0, min(1.0, base)))

if __name__ == "__main__":
    import os
    import sys

    try:
        from .utils import save_depth_maps, save_rgb_maps
    except Exception:  # pragma: no cover
        from utils import save_depth_maps, save_rgb_maps

    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _physics_root = os.path.abspath(os.path.join(_this_dir, ".."))
    if _physics_root not in sys.path:
        sys.path.insert(0, _physics_root)

    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    device = "cuda" if torch.cuda.is_available() else "cpu"
    omega_image_resolution = 512
    omega_patch_size = 16
    omega_checkpoint = (
        "/pfss/mlde/workspaces/mlde_wsp_MGPATH/thanhnv/WorldModelBenchmark/data/checkpoint/vggt_omega_1b_512.pt"
    )

    model = VGGTOmega(patch_size=omega_patch_size).to(device).eval()
    state = torch.load(omega_checkpoint, map_location="cpu")
    model.load_state_dict(state)

    depth_output_dir = "output_depth"

    ep = 397
    frame1 = 0
    frame2 = 0

    version = "real"
    # version = "good_2fps"
    # version = "bad_2fps"

    if version == "real":
        image_names = [
            f"/pfss/mlde/workspaces/mlde_wsp_MGPATH/thanhnv/WorldModelBenchmark/data/preprocessing/single_arm/real/makovian/episode_{ep:06d}/observation.images.exterior_1_left/frame_{frame1:04d}.png",
            f"/pfss/mlde/workspaces/mlde_wsp_MGPATH/thanhnv/WorldModelBenchmark/data/preprocessing/single_arm/real/makovian/episode_{ep:06d}/observation.images.exterior_2_left/frame_{frame2:04d}.png",
        ]
    else:
        image_names = [
            f"/pfss/mlde/workspaces/mlde_wsp_MGPATH/thanhnv/WorldModelBenchmark/data/multiview/thanhnv_multiview/{version}/episode_{ep:06d}/view1/frame_{frame1:04d}.png",
            f"/pfss/mlde/workspaces/mlde_wsp_MGPATH/thanhnv/WorldModelBenchmark/data/multiview/thanhnv_multiview/{version}/episode_{ep:06d}/view2/frame_{frame2:04d}.png",
        ]

    images = load_and_preprocess_images(
        image_names,
        image_resolution=omega_image_resolution,
        patch_size=omega_patch_size,
    ).to(device)
    images = images[None]  # (1, S, 3, H, W)

    with torch.no_grad():
        preds = model(images)

        depth_map = preds["depth"]
        depth_conf = preds["depth_conf"]

        extrinsic, intrinsic = encoding_to_camera(
            preds["pose_enc"], preds["images"].shape[-2:]
        )

        point_map_by_unprojection = unproject_depth_map_to_point_map(
            depth_map.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0)
        )

        depth_rendered_np, rendered_valid_mask, rgb_rendered_np = render_depth_from_point_map(
            point_map_by_unprojection=point_map_by_unprojection,
            depth_map=depth_map,
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            images=preds["images"],
        )

        depth_rendered = torch.from_numpy(depth_rendered_np).to(
            device=device, dtype=torch.float32
        )
        if depth_rendered.ndim == 3:
            depth_rendered = depth_rendered.unsqueeze(0)  # (1,S,H,W)
        depth_rendered = depth_rendered.unsqueeze(-1)  # (B,S,H,W,1)

        if rgb_rendered_np.ndim == 4:
            rgb_to_save = rgb_rendered_np[None]
        else:
            rgb_to_save = rgb_rendered_np

    valid_for_vt = rendered_valid_mask
    if isinstance(valid_for_vt, np.ndarray) and valid_for_vt.ndim == 3:
        valid_for_vt = valid_for_vt[np.newaxis, ...]

    res = geometric_calculation(
        depth=depth_map,
        depth_rendered=depth_rendered,
        valid_mask=valid_for_vt,
        images=preds["images"],
        rgb_rendered=rgb_to_save,
    )


    print("\n Depth Joint vs Depth Rendered")
    cons = geometric_consistency_metric(res)
    if "rgb_rmse" in res:
        print(f"RGB RMSE: {res['rgb_rmse']:.6f}")

    print(f"Geometric Consistency Metric: {cons}")

