import torch
import math                        # (H,W)
import numpy as np

@torch.no_grad()
def render_depth_from_point_map(
    point_map_by_unprojection: np.ndarray,
    depth_map: torch.Tensor | np.ndarray,
    extrinsic: torch.Tensor | np.ndarray,
    intrinsic: torch.Tensor | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      depth_rendered: (B,S,H,W) float32
      fov_overlap_mask: (B,S,H,W) bool
      depth_spread: (B,S,H,W) float32
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
    for s in range(S_pm):
        ds_ = d0[s]
        m = np.isfinite(ds_)
        if np.any(m):
            pts.append(point_map_by_unprojection[s][m])

    out = np.zeros((B, S_cam, H, W), dtype=np.float32)
    fov_overlap_geom = np.zeros((B, S_cam, H, W), dtype=bool)
    depth_spread = np.zeros((B, S_cam, H, W), dtype=np.float32)

    if len(pts) > 0:
        P = np.concatenate(pts, axis=0).astype(np.float32)
        Pwh = np.concatenate([P, np.ones((P.shape[0], 1), dtype=np.float32)], axis=1)

        for b in range(B):
            for s in range(S_cam):
                Eb = E[b, s].astype(np.float32)
                Kb = K[b, s].astype(np.float32)

                Pc = (Eb @ Pwh.T).T
                z = Pc[:, 2]
                valid = z > 0.01
                Pc_v = Pc[valid]
                z_v = z[valid]

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

                    if z_b.size > 0:
                        depth_min_buf = np.full((H, W), np.inf, dtype=np.float32)
                        np.minimum.at(depth_min_buf, (vi_b, ui_b), z_b)

                        depth_max_buf = np.full((H, W), -np.inf, dtype=np.float32)
                        np.maximum.at(depth_max_buf, (vi_b, ui_b), z_b)

                        hit_count = np.zeros((H, W), dtype=np.int32)
                        np.add.at(hit_count, (vi_b, ui_b), 1)

                        hit = hit_count > 0
                        depth_min_buf[~hit] = 0.0
                        depth_max_buf[~hit] = 0.0
                        out[b, s] = depth_min_buf

                        multi_hit = hit_count >= 2
                        spread = np.zeros((H, W), dtype=np.float32)
                        spread[multi_hit] = depth_max_buf[multi_hit] - depth_min_buf[multi_hit]
                        depth_spread[b, s] = spread

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

    return (out, fov_overlap_geom, depth_spread) if batched else (out[0], fov_overlap_geom[0], depth_spread[0])

def geometric_calculation(
    depth: torch.Tensor | np.ndarray,
    depth_rendered: torch.Tensor | np.ndarray,
    *,
    eps: float = 1e-6,
    valid_mask: torch.Tensor | np.ndarray | None = None,
    depth_spread: torch.Tensor | np.ndarray | None = None,   # <-- mới
    conflict_tol: float = 0.3,
) -> dict:
    """
    Inputs:
      depth_joint, depth_indep:
        - (B,S,H,W,1) or (B,S,H,W) or (S,H,W,1) or (S,H,W)
        - Typically B==1, S==2 in your pipeline.
      valid_mask:
        - Optional; same rank/layout as depths after batching: (B,S,H,W) or (S,H,W), bool or 0/1.
        - Non-boolean arrays are cast to bool (non-zero is True).

    Returns:
      {
        "view0_shape": {...},
        "view1_shape": {...},
      }
      If S>2, keys are view{idx}_shape / valid_ratio_view{idx}.
    """
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

    ds = None
    if depth_spread is not None:
        ds = _to_np_depth(depth_spread)

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
    if ds is not None and ds.shape != dj.shape:
        raise ValueError(f"depth_spread shape {ds.shape} must match depth shape {dj.shape}")


    if vm is not None:
        if vm.ndim == 3:
            vm = vm[None, ...]
        if vm.ndim != 4:
            raise ValueError(f"valid_mask must become (B,S,H,W); got {vm.shape} after batching")
        if vm.shape != dj.shape:
            raise ValueError(f"valid_mask shape {vm.shape} must match depth shape {dj.shape}")

    B, S, _, _ = dj.shape 

    def _shape_stats_1view(y_bhw: np.ndarray, x_bhw: np.ndarray, m_bhw: np.ndarray | None, spread_bhw: np.ndarray | None) -> dict | list[dict]:
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
                    "alpha": float("nan"), "beta": float("nan"), "corr": float("nan"),
                    "residual": float("nan"), "count": 0, "scale_ref": float("nan"),
                    "valid_ratio": float(roi.mean()), "conflict_ratio": float("nan"),
                })
                continue

            xv = x[roi].astype(np.float64)
            yv = y[roi].astype(np.float64)

            scale_ref = float(np.median(yv))

            mx = float(xv.mean())
            my = float(yv.mean())
            dx = xv - mx
            dy = yv - my
            vx = float((dx * dx).mean())
            cov = float((dx * dy).mean())

            # alpha = cov / (vx + 1e-12)
            # beta = my - alpha * mx

            alpha = 1.0
            beta = float(np.median(yv - xv))

            y_hat = alpha * xv + beta
            r = np.abs(yv - y_hat)

            sx = float(np.sqrt((dx * dx).mean()))
            sy = float(np.sqrt((dy * dy).mean()))
            corr = cov / (sx * sy + 1e-12)

            conflict_ratio = float("nan")
            if spread_bhw is not None:
                spread_roi = spread_bhw[b][roi].astype(np.float64)
                rel_spread = spread_roi / (scale_ref + 1e-8)
                conflict_ratio = float(np.mean(rel_spread > conflict_tol))
            out.append({
                "alpha": float(alpha),
                "beta": float(beta),
                "corr": float(corr),
                # "residual": float(r.mean()),
                # "residual": float(np.sqrt(np.mean(r * r))), # residual_rms
                # "residual": float(np.median(r)), # residual_median
                "residual": float(np.percentile(r, 90.0)), # residual_p90
                # "residual": float(np.percentile(r, 95.0)), # residual_p95
                "count": cnt,
                "scale_ref": scale_ref,
                "valid_ratio": float(roi.mean()),
                "conflict_ratio": conflict_ratio,     
            })

        return out[0] if B == 1 else out

    result: dict = {}
    for s in range(S):
        m_slice = vm[:, s] if vm is not None else None
        spread_slice = ds[:, s] if ds is not None else None
        shape_s = _shape_stats_1view(dj[:, s], di[:, s], m_slice, spread_slice)
        result[f"view{s}_shape"] = shape_s

    return result

def geometric_consistency_metric(
    res: dict,
) -> float:
    """
    Returns scalar in [0,1], higher is better.
    """
    def _view_score(vshape: dict) -> float:
        corr = float(vshape.get("corr", float("nan")))
        resid = float(vshape.get("residual", float("nan")))
        scale_ref = float(vshape.get("scale_ref", float("nan")))
        if not (math.isfinite(corr) and math.isfinite(resid)
                and math.isfinite(scale_ref) and scale_ref > 1e-6):
            return 0.0

        s_corr = max(0.0, corr)
        rel_resid = resid / scale_ref
        s_resid = math.exp(-rel_resid / 1.0)
        score = s_corr * s_resid

        conflict_ratio = vshape.get("conflict_ratio", float("nan"))
        if math.isfinite(conflict_ratio):
            score *= math.exp(-conflict_ratio / 1.0)

        return score

    v0 = res.get("view0_shape", {})
    v1 = res.get("view1_shape", {})
    s0 = _view_score(v0)
    s1 = _view_score(v1)
    base = 2 * s0 * s1 / (s0 + s1) if (s0 + s1) > 0 else 0.0

    return float(max(0.0, min(1.0, base)))

if __name__ == "__main__":
    import os
    import sys

    from vggt.utils.geometry import unproject_depth_map_to_point_map

    try:
        from .utils import save_depth_maps  
    except Exception:  # pragma: no cover
        from utils import save_depth_maps  

    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _physics_root = os.path.abspath(os.path.join(_this_dir, ".."))
    _wmb_root = os.path.abspath(os.path.join(_physics_root, ".."))
    for _root in (_physics_root, os.path.join(_wmb_root, "vggt-omega")):
        if os.path.isdir(os.path.join(_root, "vggt_omega")) and _root not in sys.path:
            sys.path.insert(0, _root)

    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    device = "cuda" if torch.cuda.is_available() else "cpu"
    omega_image_resolution = 512
    omega_patch_size = 16
    omega_checkpoint = (
        "/home/thanhnv/.cache/huggingface/hub/models--facebook--VGGT-Omega/snapshots/05654241adc2f218dfb089c373a011f8a7040576/vggt_omega_1b_512.pt"
    )

    model = VGGTOmega(patch_size=omega_patch_size).to(device).eval()
    state = torch.load(omega_checkpoint, map_location="cpu")
    model.load_state_dict(state)
    print(f"[main] VGGT-Omega loaded from {omega_checkpoint} on {device}")

    depth_output_dir = "output_depth"

    ep = 397
    frame1 = 5
    frame2 = 5

    # version = "real"
    version = "good_2fps"
    # version = "bad_2fps"

    if version == "real":
        image_names = [
        f"/home/thanhnv/data2_disk/VT/1.VinRobotics/1.Projects/Code/VideoWordModelBenchmark/thanhnv_multiview/data/real/episode_{ep:06d}/observation.images.exterior_1_left/frame_{frame1:04d}.png",
        f"/home/thanhnv/data2_disk/VT/1.VinRobotics/1.Projects/Code/VideoWordModelBenchmark/thanhnv_multiview/data/real/episode_{ep:06d}/observation.images.exterior_2_left/frame_{frame2:04d}.png",
    ]
    else:
        image_names = [
            f"/home/thanhnv/data2_disk/VT/1.VinRobotics/1.Projects/Code/VideoWordModelBenchmark/thanhnv_multiview/data/{version}/episode_{ep:06d}/view1/frame_{frame1:04d}.png",
            f"/home/thanhnv/data2_disk/VT/1.VinRobotics/1.Projects/Code/VideoWordModelBenchmark/thanhnv_multiview/data/{version}/episode_{ep:06d}/view2/frame_{frame2:04d}.png",
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
        save_depth_maps(depth_map, depth_output_dir, "depth")

        depth_rendered_np, rendered_valid_mask, depth_spread = render_depth_from_point_map(
            point_map_by_unprojection=point_map_by_unprojection,
            depth_map=depth_map,
            extrinsic=extrinsic,
            intrinsic=intrinsic
        )

        depth_rendered = torch.from_numpy(depth_rendered_np).to(
            device=device, dtype=torch.float32
        )
        if depth_rendered.ndim == 3:
            depth_rendered = depth_rendered.unsqueeze(0)  # (1,S,H,W)
        depth_rendered = depth_rendered.unsqueeze(-1)  # (B,S,H,W,1)

        save_depth_maps(depth_rendered, depth_output_dir, "rendered")
        save_depth_maps(rendered_valid_mask.astype(float), depth_output_dir, "valid_mask")

    valid_for_vt = rendered_valid_mask
    if isinstance(valid_for_vt, np.ndarray) and valid_for_vt.ndim == 3:
        valid_for_vt = valid_for_vt[np.newaxis, ...]

    res = geometric_calculation(
        depth=depth_map,
        depth_rendered=depth_rendered,
        valid_mask=valid_for_vt,
        depth_spread=depth_spread,
    )
    print(res)

    cons = geometric_consistency_metric(res)

    print(f"Geometric Consistency Metric: {cons}")

