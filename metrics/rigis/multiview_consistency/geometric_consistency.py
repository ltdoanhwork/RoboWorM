import math

import numpy as np
import torch
from numba import njit
def align_depth_scale(
    d_single: torch.Tensor,   # (B,S,H,W,1)
    d_joint: torch.Tensor,    # (B,S,H,W,1)
    max_depth: float = 0.8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Bản mở rộng đa view/đa batch của align_depth_scale_shift -- fit scale RIÊNG
    cho từng (b, s), xử lý toàn bộ trong 1 lần gọi hàm.

    Args:
      d_single, d_joint: (B,S,H,W,1)

    Returns:
      d_aligned: (B,S,H,W) -- depth_single đã được nhân với scale riêng của từng view.
      scales: (B,S) -- hệ số scale s đã dùng cho từng (b,s), để log/debug.
    """
    if d_single.shape != d_joint.shape:
        raise ValueError(f"d_single/d_joint shape mismatch: {d_single.shape} vs {d_joint.shape}")
    if d_single.ndim != 5 or d_single.shape[-1] != 1:
        raise ValueError(f"Expected (B,S,H,W,1), got {tuple(d_single.shape)}")

    ds = d_single.squeeze(-1)   # (B,S,H,W)
    dj = d_joint.squeeze(-1)    # (B,S,H,W)

    B, S, H, W = ds.shape

    ch, cw = max(1, H // 3), max(1, W // 3)
    y0 = (H - ch) // 2
    x0 = (W - cw) // 2

    ds_crop = ds[..., y0:y0 + ch, x0:x0 + cw]   # (B,S,ch,cw)
    dj_crop = dj[..., y0:y0 + ch, x0:x0 + cw]

    valid = (
        (dj_crop < max_depth) & (ds_crop < max_depth)
        & (dj_crop > 1e-4) & (ds_crop > 1e-4)
        & torch.isfinite(dj_crop) & torch.isfinite(ds_crop)
    )  # (B,S,ch,cw)

    scales = torch.ones((B, S), dtype=torch.float64, device=ds.device)

    for b in range(B):
        for s in range(S):
            m = valid[b, s]
            n_valid = int(m.sum())
            if n_valid < 10:
                continue

            x = ds_crop[b, s][m].double()
            y = dj_crop[b, s][m].double()
            log_ratio = torch.log(y) - torch.log(x)
            scales[b, s] = torch.exp(torch.median(log_ratio))

    scales_broadcast = scales.view(B, S, 1, 1).to(ds.dtype)   # (B,S,1,1)
    d_aligned = ds * scales_broadcast                          # (B,S,H,W)

    return d_aligned.unsqueeze(-1)


def _as_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x)


def _images_b_s_hwc(images: torch.Tensor | np.ndarray) -> np.ndarray:
    """(B,S,3,H,W) → (B,S,H,W,3) float32 in [0, 1]."""
    img = _as_numpy(images).astype(np.float32, copy=False)
    # (B, S, 3, H, W) → (B, S, H, W, 3)
    img = np.transpose(img, (0, 1, 3, 4, 2))
    return np.clip(img, 0.0, 1.0)

def _detect_flying_pixels(depth: np.ndarray, rel_thresh: float = 0.1) -> np.ndarray:
    d = depth
    disc = np.zeros_like(d, dtype=np.float32)
    disc = np.maximum(disc, np.abs(d - np.roll(d, 1, axis=1)))
    disc = np.maximum(disc, np.abs(d - np.roll(d, -1, axis=1)))
    disc = np.maximum(disc, np.abs(d - np.roll(d, 1, axis=0)))
    disc = np.maximum(disc, np.abs(d - np.roll(d, -1, axis=0)))

    rel_disc = disc / (d + 1e-8)
    safe_mask = rel_disc < rel_thresh
    safe_mask[0, :] = safe_mask[-1, :] = safe_mask[:, 0] = safe_mask[:, -1] = False
    return safe_mask


@njit(cache=True)
def _zbuffer_scatter_numba(
    ui: np.ndarray,
    vi: np.ndarray,
    z: np.ndarray,
    C: np.ndarray,
    H: int,
    W: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Z-buffer + RGB with closest-z wins.

    Depth matches ``np.minimum.at``. RGB matches far→near overwrite for distinct z
    (equal-z ties are vanishingly rare with float32 network depths).
    """
    depth_min = np.full((H, W), np.inf, dtype=np.float32)
    rgb_out = np.zeros((H, W, 3), dtype=np.float32)
    n = z.shape[0]
    for i in range(n):
        uii = ui[i]
        vii = vi[i]
        zf = z[i]
        if zf < depth_min[vii, uii]:
            depth_min[vii, uii] = zf
            rgb_out[vii, uii, 0] = C[i, 0]
            rgb_out[vii, uii, 1] = C[i, 1]
            rgb_out[vii, uii, 2] = C[i, 2]
    for vii in range(H):
        for uii in range(W):
            if not np.isfinite(depth_min[vii, uii]):
                depth_min[vii, uii] = 0.0
    return depth_min, rgb_out


@njit(cache=True)
def _scatter_hits_numba(u: np.ndarray, v: np.ndarray, valid: np.ndarray, H: int, W: int) -> np.ndarray:
    """other→own FOV scatter: rint + clip, same as NumPy path."""
    hit = np.zeros((H, W), dtype=np.bool_)
    n = valid.shape[0]
    for i in range(n):
        if not valid[i]:
            continue
        ui = int(np.rint(u[i]))
        vi = int(np.rint(v[i]))
        if ui < 0:
            ui = 0
        elif ui > W - 1:
            ui = W - 1
        if vi < 0:
            vi = 0
        elif vi > H - 1:
            vi = H - 1
        hit[vi, ui] = True
    return hit


def _project_points_to_camera(
    pm_hw3: np.ndarray,
    E: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project (H,W,3) world points with extrinsic E into pixel u,v and depth z."""
    H, W, _ = pm_hw3.shape
    P = np.ascontiguousarray(pm_hw3.reshape(-1, 3), dtype=np.float32)
    ones = np.ones((P.shape[0], 1), dtype=np.float32)
    Pwh = np.concatenate([P, ones], axis=1)
    Pc = (E.astype(np.float32, copy=False) @ Pwh.T).T
    z = Pc[:, 2]
    pix = (K.astype(np.float32, copy=False) @ Pc.T).T
    u = pix[:, 0] / (pix[:, 2] + 1e-6)
    v = pix[:, 1] / (pix[:, 2] + 1e-6)
    return u, v, z


def _fov_overlap(
    point_maps: np.ndarray,  # (S, H, W, 3)
    E: np.ndarray,  # (B, S, 3, 4)
    K: np.ndarray,  # (B, S, 3, 3)
    b: int,
) -> np.ndarray:
    """
    Bidirectional FOV overlap for all views.

    Same results as the nested NumPy loops, but each directed projection
    (view i → camera j) is computed once and reused.
    """
    S_cam, H, W, _ = point_maps.shape
    out = np.zeros((S_cam, H, W), dtype=bool)

    # projections[(i, j)] = (u, v, z) of view-i points in camera j
    projections: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for i in range(S_cam):
        for j in range(S_cam):
            if i == j:
                continue
            projections[(i, j)] = _project_points_to_camera(
                point_maps[i], E[b, j], K[b, j]
            )

    for s in range(S_cam):
        acc = np.zeros(H * W, dtype=bool)
        for src in range(S_cam):
            if src == s:
                continue
            # own → other: view-s points visible in camera src
            u, v, z = projections[(s, src)]
            own_to_other = (
                (z > 0.01)
                & (u >= 0) & (u < W) & (v >= 0) & (v < H)
                & np.isfinite(u) & np.isfinite(v)
            )
            acc |= own_to_other

            # other → own: view-src points that hit camera s (scatter)
            u2, v2, z2 = projections[(src, s)]
            valid2 = (
                (z2 > 0.01)
                & (u2 >= 0) & (u2 < W) & (v2 >= 0) & (v2 < H)
                & np.isfinite(u2) & np.isfinite(v2)
            )
            hit = _scatter_hits_numba(
                u2.astype(np.float32, copy=False),
                v2.astype(np.float32, copy=False),
                valid2.astype(np.bool_, copy=False),
                H,
                W,
            )
            acc |= hit.reshape(-1)
        out[s] = acc.reshape(H, W)
    return out

def _render_depth(
    point_map_by_unprojection: np.ndarray,
    d0: np.ndarray,
    rgb_maps: np.ndarray,
    E: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Same algorithm as ``_render_depth_numpy``, faster system path:
      - FOV projections computed once per directed pair (half the matmuls for S=2)
      - Numba z-buffer scatter (avoids ``argsort`` + ``minimum.at`` overhead)
      - Numba scatter for other→own FOV hits
    """
    S_pm, H, W, _ = point_map_by_unprojection.shape
    B = E.shape[0]
    S_cam = E.shape[1]

    pts = []
    cols = []
    for s in range(S_pm):
        ds_ = d0[s]
        m = np.isfinite(ds_) & (ds_ > 1e-6)
        safe = _detect_flying_pixels(ds_, rel_thresh=0.1)
        m = m & safe
        if np.any(m):
            pts.append(np.ascontiguousarray(point_map_by_unprojection[s][m], dtype=np.float32))
            cols.append(np.ascontiguousarray(rgb_maps[s][m], dtype=np.float32))

    out = np.zeros((B, S_cam, H, W), dtype=np.float32)
    fov_overlap_geom = np.zeros((B, S_cam, H, W), dtype=bool)
    rgb_out = np.zeros((B, S_cam, H, W, 3), dtype=np.float32)

    point_maps_f = np.ascontiguousarray(point_map_by_unprojection, dtype=np.float32)
    E_f = np.ascontiguousarray(E, dtype=np.float32)
    K_f = np.ascontiguousarray(K, dtype=np.float32)

    if len(pts) == 0:
        return out, fov_overlap_geom, rgb_out

    P = np.concatenate(pts, axis=0)
    Pwh = np.concatenate([P, np.ones((P.shape[0], 1), dtype=np.float32)], axis=1)
    C = np.concatenate(cols, axis=0)
    Pwh = np.ascontiguousarray(Pwh)
    C = np.ascontiguousarray(C)

    for b in range(B):
        for s in range(S_cam):
            Eb = E_f[b, s]
            Kb = K_f[b, s]
            Pc = (Eb @ Pwh.T).T
            z = Pc[:, 2]
            valid = z > 0.01
            Pc_v = Pc[valid]
            z_v = z[valid]
            C_v = C[valid]
            if z_v.size > 0:
                pix = (Kb @ Pc_v.T).T
                u = pix[:, 0] / (pix[:, 2] + 1e-5)
                v = pix[:, 1] / (pix[:, 2] + 1e-5)
                ui_all = np.rint(u)
                vi_all = np.rint(v)
                inb = (
                    (ui_all >= 0) & (ui_all < W) & (vi_all >= 0) & (vi_all < H)
                    & np.isfinite(ui_all) & np.isfinite(vi_all)
                )
                ui_b = ui_all[inb].astype(np.int32)
                vi_b = vi_all[inb].astype(np.int32)
                z_b = z_v[inb].astype(np.float32)
                C_b = C_v[inb]
                if z_b.size > 0:
                    depth_s, rgb_s = _zbuffer_scatter_numba(
                        np.ascontiguousarray(ui_b),
                        np.ascontiguousarray(vi_b),
                        np.ascontiguousarray(z_b),
                        np.ascontiguousarray(C_b),
                        H,
                        W,
                    )
                    out[b, s] = depth_s
                    rgb_out[b, s] = rgb_s
                    

        fov_overlap_geom[b] = _fov_overlap(point_maps_f, E_f, K_f, b)

    return out, fov_overlap_geom, rgb_out


@torch.no_grad()
def render_depth_from_point_map(
    point_map_by_unprojection: np.ndarray,
    depth_map: torch.Tensor | np.ndarray,
    extrinsic: torch.Tensor | np.ndarray,
    intrinsic: torch.Tensor | np.ndarray,
    images: torch.Tensor | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project multi-view unprojected points into each camera (z-buffer).

    Args:
      point_map_by_unprojection: (S, H, W, 3)
      depth_map: (B, S, H, W, 1)
      extrinsic: (B, S, 3, 4)
      intrinsic: (B, S, 3, 3)
      images: (B, S, 3, H, W) in [0, 1]

    Returns:
      depth_rendered: (B, S, H, W) float32
      fov_overlap_mask: (B, S, H, W) bool
      rgb_rendered: (B, S, H, W, 3) float32 in [0, 1]
    """
    d = _as_numpy(depth_map)[..., 0]  # (B, S, H, W)
    d0 = d[0]  # (S, H, W) — point_map is from batch 0
    rgb_maps = _images_b_s_hwc(images)[0]  # (S, H, W, 3)
    E = _as_numpy(extrinsic)  # (B, S, 3, 4)
    K = _as_numpy(intrinsic)  # (B, S, 3, 3)
    pm = np.asarray(point_map_by_unprojection)

    return _render_depth(pm, d0, rgb_maps, E, K)


def geometric_calculation(
    depth: torch.Tensor | np.ndarray,
    depth_rendered: torch.Tensor | np.ndarray,
    *,
    eps: float = 1e-6,
    valid_mask: torch.Tensor | np.ndarray,
    images: torch.Tensor | np.ndarray,
    rgb_rendered: torch.Tensor | np.ndarray,
) -> dict:
    """
    Args:
      depth: (B, S, H, W, 1)
      depth_rendered: (B, S, H, W)
      valid_mask: (B, S, H, W) bool
      images: (B, S, 3, H, W) in [0, 1]
      rgb_rendered: (B, S, H, W, 3) in [0, 1]
    """
    dj = _as_numpy(depth)[..., 0]  # (B, S, H, W)
    di = _as_numpy(depth_rendered)  # (B, S, H, W)
    vm = _as_numpy(valid_mask).astype(bool, copy=False)
    B, S, H, W = dj.shape
    rgb_src = _images_b_s_hwc(images)  # (B, S, H, W, 3)
    rgb_ren = np.clip(_as_numpy(rgb_rendered).astype(np.float32, copy=False), 0.0, 1.0)
    def _shape_stats_1view(
        y_bhw: np.ndarray,
        x_bhw: np.ndarray,
        m_bhw: np.ndarray,
        rgb_src_bhw3: np.ndarray,
        rgb_ren_bhw3: np.ndarray,
    ) -> dict | list[dict]:
        out = []
        for b in range(B):
            y = y_bhw[b]
            x = x_bhw[b]
            roi = np.isfinite(y) & np.isfinite(x) & (y > eps) & (x > eps) & m_bhw[b]
            cnt = int(roi.sum())
            if cnt == 0:
                out.append({
                    "residual": float("nan"),
                    "count": 0,
                    "scale_ref": float("nan"),
                    "valid_ratio": float(roi.mean()),
                    "rgb_rmse": float("nan"),
                })
                continue
            r_roi = np.abs(y[roi].astype(np.float64) - x[roi].astype(np.float64))
            # residual_p90 = float(np.percentile(r_roi, 90.0))
            residual_p90 = float(np.mean(r_roi))
            scale_ref = float(np.median(y.astype(np.float64)))
            src_pix = rgb_src_bhw3[b].astype(np.float64)[roi] 
            ren_pix = rgb_ren_bhw3[b].astype(np.float64)[roi] 
            rgb_rmse = float(np.sqrt(np.mean(np.square(src_pix - ren_pix))))
            out.append({
                "residual": residual_p90,
                "count": cnt,
                "scale_ref": scale_ref,
                "valid_ratio": float(roi.mean()),
                "rgb_rmse": rgb_rmse,
            })
        return out[0] if B == 1 else out
    result: dict = {}
    for s in range(S):
        result[f"view{s}_shape"] = _shape_stats_1view(
            dj[:, s], di[:, s], vm[:, s], rgb_src[:, s], rgb_ren[:, s]
        )
    diffs = []
    for s in range(S):
        for b in range(B):
            diffs.append(rgb_src[b, s] - rgb_ren[b, s])
    if diffs:
        all_diff = np.concatenate(diffs, axis=0).astype(np.float64)
        result["rgb_rmse"] = float(np.sqrt(np.mean(np.square(all_diff))))
    else:
        result["rgb_rmse"] = 0.0
    return result

def geometric_consistency_metric(res: dict) -> float:
    """Returns scalar in [0, 1]; higher is better."""

    def _view_score(vshape: dict) -> float:
        resid = float(vshape.get("residual", float("nan")))
        scale_ref = float(vshape.get("scale_ref", float("nan")))
        if not (math.isfinite(resid) and math.isfinite(scale_ref) and scale_ref > 1e-6):
            return 0.0

        rel_resid = resid / scale_ref
        s_resid = max(0.0, 1.0 - rel_resid**2)

        rgb_rmse = vshape.get("rgb_rmse", float("nan"))
        if math.isfinite(rgb_rmse):
            s_rgb = max(0.0, 1.0 - float(rgb_rmse))
        else:
            s_rgb = 1.0
        
        return (s_resid + s_rgb) / 2.0

    s0 = _view_score(res.get("view0_shape", {}))
    s1 = _view_score(res.get("view1_shape", {}))
    return float(max(0.0, min(1.0, s0 * s1)))


if __name__ == "__main__":
    """Smoke demo: geometric consistency on one episode (view1/view2 frames).

    For full-video RIGIS (geo + pose), prefer the service client:
      python -m multiview_consistency.run_video_metrics --episode_dir ... --host ... --port ...
    """
    import argparse
    import os
    import sys
    from pathlib import Path

    _this_dir = Path(__file__).resolve().parent
    _rigis_root = _this_dir.parent  # metrics/rigis
    if str(_rigis_root) not in sys.path:
        sys.path.insert(0, str(_rigis_root))

    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.geometry import unproject_depth_map_to_point_map
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    parser = argparse.ArgumentParser(
        description="Demo: per-frame geometric consistency for a multi-view episode."
    )
    parser.add_argument(
        "--episode_dir",
        type=Path,
        required=True,
        help="Episode folder with view subdirs (e.g. view1/, view2/) and frame_XXXX.png",
    )
    parser.add_argument(
        "--omega_checkpoint",
        type=Path,
        required=True,
        help="Path to vggt_omega_1b_512.pt",
    )
    parser.add_argument("--views", default="view1,view2", help="Comma-separated view folder names")
    parser.add_argument("--image_resolution", type=int, default=512)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--max_frames", type=int, default=0, help="0 = all matched frames")
    parser.add_argument("--max_depth", type=float, default=1.0, help="Depth align crop threshold")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    view_names = [v.strip() for v in args.views.split(",") if v.strip()]
    if len(view_names) < 2:
        raise SystemExit("--views needs at least two names")

    episode_dir = args.episode_dir.expanduser().resolve()
    view_dirs = [episode_dir / v for v in view_names]
    for vd in view_dirs:
        if not vd.is_dir():
            raise SystemExit(f"Missing view folder: {vd}")

    # Matched frame stems across views
    stem_sets = []
    for vd in view_dirs:
        stems = {
            p.stem
            for p in vd.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        }
        stem_sets.append(stems)
    stems = sorted(set.intersection(*stem_sets))
    if not stems:
        raise SystemExit(f"No matched frames under {episode_dir} for views={view_names}")
    if args.max_frames > 0:
        stems = stems[: args.max_frames]

    ckpt = args.omega_checkpoint.expanduser().resolve()
    if not ckpt.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    device = args.device
    model = VGGTOmega(patch_size=args.patch_size).to(device).eval()
    state = torch.load(str(ckpt), map_location="cpu")
    model.load_state_dict(state)

    scores: list[float] = []
    print(f"episode={episode_dir}")
    print(f"views={view_names}  frames={len(stems)}  device={device}")

    for stem in stems:
        image_names = [str(vd / f"{stem}.png") for vd in view_dirs]
        # fallback other extensions
        for i, p in enumerate(image_names):
            if not os.path.isfile(p):
                for ext in (".jpg", ".jpeg", ".webp"):
                    alt = str(Path(p).with_suffix(ext))
                    if os.path.isfile(alt):
                        image_names[i] = alt
                        break

        images = load_and_preprocess_images(
            image_names,
            image_resolution=args.image_resolution,
            patch_size=args.patch_size,
        ).to(device)
        images = images[None]  # (1, S, 3, H, W)

        with torch.no_grad():
            preds = model(images)
            depth_map = preds["depth"]
            depth_map_single = preds["depth_single"]
            extrinsic, intrinsic = encoding_to_camera(
                preds["pose_enc"], preds["images"].shape[-2:]
            )

            depth_map_single_aligned = align_depth_scale(
                depth_map_single, depth_map, max_depth=args.max_depth
            )
            # align_depth_scale may return Tensor or (Tensor, scales)
            if isinstance(depth_map_single_aligned, tuple):
                depth_map_single_aligned = depth_map_single_aligned[0]

            point_map = unproject_depth_map_to_point_map(
                depth_map_single_aligned[0],
                extrinsic.squeeze(0),
                intrinsic.squeeze(0),
            )
            depth_rendered_np, valid_mask, rgb_rendered_np = render_depth_from_point_map(
                point_map_by_unprojection=point_map,
                depth_map=depth_map_single_aligned,
                extrinsic=extrinsic,
                intrinsic=intrinsic,
                images=preds["images"],
            )
            depth_rendered = torch.from_numpy(depth_rendered_np).to(
                device=device, dtype=torch.float32
            )

        res = geometric_calculation(
            depth=depth_map_single_aligned,
            depth_rendered=depth_rendered,
            valid_mask=valid_mask,
            images=preds["images"],
            rgb_rendered=rgb_rendered_np,
        )
        cons = geometric_consistency_metric(res)
        scores.append(cons)
        print(f"  {stem}: geometric_consistency={cons:.4f}")

    mean_score = float(sum(scores) / len(scores)) if scores else float("nan")
    print(f"mean geometric_consistency over {len(scores)} frames: {mean_score:.4f}")