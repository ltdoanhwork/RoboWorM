import torch
import math                        # (H,W)
import numpy as np

@torch.no_grad()
def render_depth_from_point_map(
    point_map_by_unprojection: np.ndarray,   # (S,H,W,3) world points (as returned by unproject_depth_map_to_point_map)
    depth_map: torch.Tensor | np.ndarray,    # (B,S,H,W,1)/(B,S,H,W) or (S,H,W,1)/(S,H,W) for fusion validity / threshold
    extrinsic: torch.Tensor | np.ndarray,    # (B,S,3,4) or (S,3,4) cam_from_world
    intrinsic: torch.Tensor | np.ndarray,    # (B,S,3,3) or (S,3,3)
    *,
    eps: float = 1e-6,
    confidence: torch.Tensor | np.ndarray | None = None,
    conf_thres: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      depth_rendered: (S,H,W) float32 (if inputs are unbatched) or (B,S,H,W) if batched.
        Pixels with no raster hits are 0 (after clearing non-finite buffer entries).
      valid_depth_mask: same shape, dtype bool. True iff depth is finite, not NaN/Inf, and > ``eps``.
        Intended for restricting consistency metrics to reliable pixels.

    """
    if point_map_by_unprojection.ndim != 4 or point_map_by_unprojection.shape[-1] != 3:
        raise ValueError(f"point_map_by_unprojection must be (S,H,W,3), got {point_map_by_unprojection.shape}")

    S_pm, H, W, _ = point_map_by_unprojection.shape

    # ---- normalize depth to numpy (S,H,W) or (B,S,H,W) ----
    if isinstance(depth_map, torch.Tensor):
        d = depth_map.detach().cpu().float()
        if d.ndim == 5:
            d = d.squeeze(-1)
        if d.ndim == 4:
            d0 = d[0].numpy()  # (S,H,W) assuming B==1 typical usage here
        elif d.ndim == 3:
            d0 = d.numpy()
        else:
            raise ValueError(f"depth_map must be 3D/4D/5D, got {d.ndim}D")
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

    c0: np.ndarray | None = None
    conf_threshold_value = 0.0
    if confidence is not None:
        if isinstance(confidence, torch.Tensor):
            c = confidence.detach().cpu().float()
            if c.ndim == 5:
                c = c.squeeze(-1)
            if c.ndim == 4:
                c0 = c[0].numpy()
            elif c.ndim == 3:
                c0 = c.numpy()
            else:
                raise ValueError(f"confidence must be 3D/4D/5D, got {c.ndim}D")
        else:
            c0 = np.asarray(confidence, dtype=np.float32)
            if c0.ndim == 4:
                c0 = c0[0]
            if c0.ndim == 4 and c0.shape[-1] == 1:
                c0 = c0[..., 0]
            if c0.ndim == 3 and c0.shape[-1] == 1:
                c0 = c0[..., 0]
        if c0.shape[:3] != (S_pm, H, W):
            raise ValueError(f"confidence must match (S,H,W)={(S_pm,H,W)}, got {c0.shape}")

        thr_pct = 30.0 if conf_thres is None else float(conf_thres)

        conf_flat = np.concatenate([c0[s].ravel() for s in range(S_pm)])
        conf_flat = conf_flat[np.isfinite(conf_flat)]
        if thr_pct == 0.0:
            conf_threshold_value = 0.0
        elif conf_flat.size == 0:
            conf_threshold_value = 0.0
        else:
            conf_threshold_value = float(np.percentile(conf_flat, thr_pct))

    # ---- normalize extrinsic/intrinsic to numpy, handle optional batch ----
    def _to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().float().numpy()
        return x

    E = _to_np(extrinsic)
    K = _to_np(intrinsic)

    batched = (E.ndim == 4)
    if E.ndim == 3:
        E = E[None]  # (1,S,3,4)
    if K.ndim == 3:
        K = K[None]  # (1,S,3,3)
    B = E.shape[0]
    S_cam = E.shape[1]
    if S_cam != S_pm:
        # allow rendering onto a different set of cameras, but requires K/E S_cam
        pass

    pts = []
    for s in range(S_pm):
        ds = d0[s]
        m = np.isfinite(ds)
        if c0 is not None:
            cs = c0[s]
            m = m & np.isfinite(cs) & (cs >= conf_threshold_value) & (cs > 1e-5)
        if np.any(m):
            pts.append(point_map_by_unprojection[s][m])  # (Ns,3)
    if len(pts) == 0:
        out = np.zeros((B, S_cam, H, W), dtype=np.float32)
        valid = np.zeros((B, S_cam, H, W), dtype=bool)
        return (out, valid) if batched else (out[0], valid[0])

    P = np.concatenate(pts, axis=0).astype(np.float32)  # (N,3)
    Pwh = np.concatenate([P, np.ones((P.shape[0], 1), dtype=np.float32)], axis=1)  # (N,4)

    out = np.zeros((B, S_cam, H, W), dtype=np.float32)
    valid_mask = np.zeros((B, S_cam, H, W), dtype=bool)

    for b in range(B):
        for s in range(S_cam):
            Eb = E[b, s].astype(np.float32)  # (3,4)
            Kb = K[b, s].astype(np.float32)  # (3,3)

            Pc = (Eb @ Pwh.T).T  # (N,3)
            z = Pc[:, 2]
            valid = z > eps
            Pc = Pc[valid]
            z = z[valid]
            if z.size == 0:
                continue

            pix = (Kb @ Pc.T).T  # (M,3)
            u = pix[:, 0] / (pix[:, 2] + eps)
            v = pix[:, 1] / (pix[:, 2] + eps)

            ui = np.rint(u).astype(np.int32)
            vi = np.rint(v).astype(np.int32)

            inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            ui = ui[inb]
            vi = vi[inb]
            z = z[inb].astype(np.float32)
            if z.size == 0:
                continue

            depth = np.full((H, W), np.inf, dtype=np.float32)
            # z-buffer: min depth per pixel
            np.minimum.at(depth, (vi, ui), z)
            depth[~np.isfinite(depth)] = 0.0
            out[b, s] = depth
            valid_mask[b, s] = np.isfinite(out[b, s]) & (out[b, s] > eps)

    return (out, valid_mask) if batched else (out[0], valid_mask[0])

import numpy as np

def _valid_points_per_view(point_map: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """
    point_map: (S, H, W, 3) world coordinates from unproject_depth_map_to_point_map
    Returns P0, P1 as (N, 3) finite points with non-trivial norm (optional).
    """
    if point_map.ndim != 4 or point_map.shape[-1] != 3:
        raise ValueError(f"Expected (S,H,W,3), got {point_map.shape}")

    pts = []
    for s in range(point_map.shape[0]):
        P = point_map[s].reshape(-1, 3).astype(np.float64)
        m = np.isfinite(P).all(axis=1)
        # bỏ điểm gần origin / depth invalid (thường là 0 hoặc rất nhỏ)
        m &= np.linalg.norm(P, axis=1) > eps
        pts.append(P[m])
    if len(pts) < 2:
        raise ValueError("Need at least 2 views")
    return pts[0], pts[1]


def _bbox_3d(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return P.min(axis=0), P.max(axis=0)


def _bbox_volume(lo: np.ndarray, hi: np.ndarray) -> float:
    side = np.maximum(hi - lo, 0.0)
    return float(side[0] * side[1] * side[2])


def _bbox_intersection_volume(lo0, hi0, lo1, hi1) -> float:
    lo_i = np.maximum(lo0, lo1)
    hi_i = np.minimum(hi0, hi1)
    return _bbox_volume(lo_i, hi_i)


def multiview_disjoint_bbox(
    point_map: np.ndarray,
    *,
    eps: float = 1e-6,
    min_overlap_ratio: float = 0.05,
) -> dict:
    """
    Cách B: FAIL (disjoint) nếu IoU_bbox = vol(intersection) / vol(union) < min_overlap_ratio
    """
    P0, P1 = _valid_points_per_view(point_map, eps=eps)
    if P0.size == 0 or P1.size == 0:
        return {
            "method": "bbox",
            "disjoint": True,
            "reason": "empty_point_cloud",
            "overlap_ratio": 0.0,
            "min_overlap_ratio": min_overlap_ratio,
        }

    lo0, hi0 = _bbox_3d(P0)
    lo1, hi1 = _bbox_3d(P1)
    v0 = _bbox_volume(lo0, hi0)
    v1 = _bbox_volume(lo1, hi1)
    v_i = _bbox_intersection_volume(lo0, hi0, lo1, hi1)
    v_u = v0 + v1 - v_i
    overlap_ratio = v_i / (v_u + 1e-12)

    return {
        "method": "bbox",
        "disjoint": bool(overlap_ratio < min_overlap_ratio),
        "overlap_ratio": float(overlap_ratio),
        "vol_intersection": float(v_i),
        "vol_union": float(v_u),
        "vol_view0": float(v0),
        "vol_view1": float(v1),
        "min_overlap_ratio": min_overlap_ratio,
        "bbox0": (lo0.tolist(), hi0.tolist()),
        "bbox1": (lo1.tolist(), hi1.tolist()),
        "n0": int(P0.shape[0]),
        "n1": int(P1.shape[0]),
    }

def geometric_calculation(
    depth: torch.Tensor | np.ndarray,
    depth_rendered: torch.Tensor | np.ndarray,
    *,
    eps: float = 1e-6,
    valid_mask: torch.Tensor | np.ndarray | None = None,
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

    if vm is not None:
        if vm.ndim == 3:
            vm = vm[None, ...]
        if vm.ndim != 4:
            raise ValueError(f"valid_mask must become (B,S,H,W); got {vm.shape} after batching")
        if vm.shape != dj.shape:
            raise ValueError(f"valid_mask shape {vm.shape} must match depth shape {dj.shape}")

    B, S, _, _ = dj.shape

    def _shape_stats_1view(y_bhw: np.ndarray, x_bhw: np.ndarray, m_bhw: np.ndarray | None) -> dict | list[dict]:
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
                out.append(
                    {
                        "alpha": float("nan"),
                        "beta": float("nan"),
                        "corr": float("nan"),
                        "residual": float("nan"),
                        "count": 0,
                    }
                )
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

            alpha = cov / (vx + 1e-12)
            beta = my - alpha * mx

            y_hat = alpha * xv + beta
            r = np.abs(yv - y_hat)

            sx = float(np.sqrt((dx * dx).mean()))
            sy = float(np.sqrt((dy * dy).mean()))
            corr = cov / (sx * sy + 1e-12)

            out.append(
                {
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
                }
            )
        return out[0] if B == 1 else out

    result: dict = {}
    for s in range(S):
        m_slice = vm[:, s] if vm is not None else None
        shape_s = _shape_stats_1view(dj[:, s], di[:, s], m_slice)
        # Fraction of image pixels in the consistency ROI (depth valid [+ mask])
        if B == 1:
            roi = np.isfinite(dj[0, s]) & np.isfinite(di[0, s]) & (dj[0, s] > eps) & (di[0, s] > eps)
            if vm is not None:
                roi = roi & vm[0, s]
            vr = float(roi.mean())
        else:
            vr = []
            for b in range(B):
                roi_b = (
                    np.isfinite(dj[b, s])
                    & np.isfinite(di[b, s])
                    & (dj[b, s] > eps)
                    & (di[b, s] > eps)
                )
                if vm is not None:
                    roi_b = roi_b & vm[b, s]
                vr.append(float(roi_b.mean()))

        result[f"view{s}_shape"] = shape_s

    return result

def geometric_consistency_metric(
    res: dict,        
) -> float:
    """
    Returns scalar in [0,1], higher is better
    """
    def _view_score(vshape: dict, *, tau: float = 1.0) -> float:
        corr = float(vshape.get("corr", float("nan")))
        resid = float(vshape.get("residual", float("nan")))
        scale_ref = float(vshape.get("scale_ref", float("nan")))
        if not (math.isfinite(corr) and math.isfinite(resid)
                and math.isfinite(scale_ref) and scale_ref > 1e-6):
            return 0.0

        s_corr = max(0.0, corr)    
        rel_resid = resid / scale_ref
        s_resid = math.exp(-rel_resid / tau)
        return s_corr * s_resid     

    v0 = res.get("view0_shape", {})
    v1 = res.get("view1_shape", {})
    s0 = _view_score(v0)
    s1 = _view_score(v1)
    # base = (s0 + s1) / 2.0
    base = math.sqrt(max(s0, 0.0) * max(s1, 0.0))

    return float(max(0.0, min(1.0, base)))

if __name__ == "__main__":
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    try:
        from .utils import save_depth_maps  # pyright: ignore[reportMissingImports]
    except Exception:  # pragma: no cover
        from utils import save_depth_maps  # type: ignore

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

    depth_output_dir = "output_depth"

    ep = 255
    frame1 = 10
    frame2 = 10
    
    version = "good_2fps"
    # version = "bad_2fps"

    image_names = [f"/pfss/mlde/workspaces/mlde_wsp_MGPATH/thanhnv/WorldModelBenchmark/data/multiview/thanhnv_multiview/{version}/episode_{ep:06d}/view1/frame_{frame1:04d}.png", \
        f"/pfss/mlde/workspaces/mlde_wsp_MGPATH/thanhnv/WorldModelBenchmark/data/multiview/thanhnv_multiview/{version}/episode_{ep:06d}/view2/frame_{frame2:04d}.png"]

    print(image_names)

    images = load_and_preprocess_images(image_names).to(device)
    images = images[None]

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            aggregated_tokens_list, ps_idx = model.aggregator(images)
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

        # save_depth_maps(depth_map, depth_output_dir, "multiview")
        
        point_map_by_unprojection = unproject_depth_map_to_point_map(
            depth_map.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0)
        )
        
        depth_rendered_np, rendered_valid_mask = render_depth_from_point_map(
            point_map_by_unprojection=point_map_by_unprojection,
            depth_map=depth_map,
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            eps=1e-6,
            confidence=depth_conf,
            conf_thres=30.0,
        )  

        depth_rendered = torch.from_numpy(depth_rendered_np).to(device=device, dtype=torch.float32)
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
    )
    print(res)

    cons = geometric_consistency_metric(res)

    print(f"Geometric Consistency Metric: {cons}")
