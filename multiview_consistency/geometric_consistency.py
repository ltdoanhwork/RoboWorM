import math

import numpy as np
import torch

# Canonical layouts for this pipeline (B may be 1):
#   images          (B, S, 3, H, W) float [0, 1]
#   depth           (B, S, H, W, 1)
#   extrinsic       (B, S, 3, 4)
#   intrinsic       (B, S, 3, 3)
#   point_map       (S, H, W, 3)   # after unproject on batch item 0
#   depth_rendered  (B, S, H, W)
#   valid_mask      (B, S, H, W) bool
#   rgb_rendered    (B, S, H, W, 3) float [0, 1]

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
                print(f"⚠️ Cảnh báo: (b={b}, s={s}) chỉ có {n_valid} pixels hợp lệ trong vùng center. Giữ scale=1.0.")
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


# @torch.no_grad()
# def render_depth_from_point_map(
#     point_map_by_unprojection: np.ndarray,
#     depth_map: torch.Tensor | np.ndarray,
#     extrinsic: torch.Tensor | np.ndarray,
#     intrinsic: torch.Tensor | np.ndarray,
#     images: torch.Tensor | np.ndarray,
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Project multi-view unprojected points into each camera (z-buffer).

#     Args:
#       point_map_by_unprojection: (S, H, W, 3)
#       depth_map: (B, S, H, W, 1)
#       extrinsic: (B, S, 3, 4)
#       intrinsic: (B, S, 3, 3)
#       images: (B, S, 3, H, W) in [0, 1]

#     Returns:
#       depth_rendered: (B, S, H, W) float32
#       fov_overlap_mask: (B, S, H, W) bool
#       rgb_rendered: (B, S, H, W, 3) float32 in [0, 1]
#     """
#     S_pm, H, W, _ = point_map_by_unprojection.shape

#     d = _as_numpy(depth_map)[..., 0]  # (B, S, H, W)
#     d0 = d[0]  # (S, H, W) — point_map is from batch 0

#     rgb_maps = _images_b_s_hwc(images)[0]  # (S, H, W, 3)

#     E = _as_numpy(extrinsic)  # (B, S, 3, 4)
#     K = _as_numpy(intrinsic)  # (B, S, 3, 3)
#     B = E.shape[0]
#     S_cam = E.shape[1]

#     pts = []
#     cols = []
#     for s in range(S_pm):
#         ds_ = d0[s]
#         m = np.isfinite(ds_) & (ds_ > 1e-6)
#         if np.any(m):
#             pts.append(point_map_by_unprojection[s][m])
#             cols.append(rgb_maps[s][m])

#     out = np.zeros((B, S_cam, H, W), dtype=np.float32)
#     fov_overlap_geom = np.zeros((B, S_cam, H, W), dtype=bool)
#     rgb_out = np.zeros((B, S_cam, H, W, 3), dtype=np.float32)

#     if len(pts) > 0:
#         P = np.concatenate(pts, axis=0).astype(np.float32)
#         Pwh = np.concatenate([P, np.ones((P.shape[0], 1), dtype=np.float32)], axis=1)
#         C = np.concatenate(cols, axis=0).astype(np.float32)

#         for b in range(B):
#             for s in range(S_cam):
#                 Eb = E[b, s].astype(np.float32)
#                 Kb = K[b, s].astype(np.float32)

#                 Pc = (Eb @ Pwh.T).T
#                 z = Pc[:, 2]
#                 valid = z > 0.01
#                 Pc_v = Pc[valid]
#                 z_v = z[valid]
#                 C_v = C[valid]

#                 if z_v.size > 0:
#                     pix = (Kb @ Pc_v.T).T
#                     u = pix[:, 0] / (pix[:, 2] + 1e-5)
#                     v = pix[:, 1] / (pix[:, 2] + 1e-5)

#                     ui_all = np.rint(u)
#                     vi_all = np.rint(v)
#                     inb = (
#                         (ui_all >= 0)
#                         & (ui_all < W)
#                         & (vi_all >= 0)
#                         & (vi_all < H)
#                         & np.isfinite(ui_all)
#                         & np.isfinite(vi_all)
#                     )

#                     ui_b = ui_all[inb].astype(np.int32)
#                     vi_b = vi_all[inb].astype(np.int32)
#                     z_b = z_v[inb].astype(np.float32)
#                     C_b = C_v[inb]

#                     if z_b.size > 0:
#                         depth_min_buf = np.full((H, W), np.inf, dtype=np.float32)
#                         np.minimum.at(depth_min_buf, (vi_b, ui_b), z_b)

#                         hit_count = np.zeros((H, W), dtype=np.int32)
#                         np.add.at(hit_count, (vi_b, ui_b), 1)

#                         hit = hit_count > 0
#                         depth_min_buf[~hit] = 0.0
#                         out[b, s] = depth_min_buf

#                         # Far → near so closer points overwrite RGB.
#                         order = np.argsort(-z_b)
#                         rgb_out[b, s, vi_b[order], ui_b[order]] = C_b[order]

#                 pm_s = point_map_by_unprojection[s].reshape(-1, 3).astype(np.float32)
#                 Pwh_s = np.concatenate(
#                     [pm_s, np.ones((pm_s.shape[0], 1), dtype=np.float32)], axis=1
#                 )
#                 acc_geom = np.zeros(H * W, dtype=bool)
#                 for src in range(S_cam):
#                     if src == s:
#                         continue
#                     Esrc = E[b, src].astype(np.float32)
#                     Ksrc = K[b, src].astype(np.float32)

#                     Pc_g = (Esrc @ Pwh_s.T).T
#                     z_g = Pc_g[:, 2]
#                     pix_g = (Ksrc @ Pc_g.T).T
#                     u_g = pix_g[:, 0] / (pix_g[:, 2] + 1e-6)
#                     v_g = pix_g[:, 1] / (pix_g[:, 2] + 1e-6)

#                     in_front = z_g > 0.01
#                     in_bounds = (u_g >= 0) & (u_g < W) & (v_g >= 0) & (v_g < H)
#                     overlap_flat = in_front & in_bounds & np.isfinite(u_g) & np.isfinite(v_g)
#                     acc_geom |= overlap_flat

#                 fov_overlap_geom[b, s] = acc_geom.reshape(H, W)

#     return out, fov_overlap_geom, rgb_out

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

# @torch.no_grad()
# def render_depth_from_point_map(
#     point_map_by_unprojection: np.ndarray,
#     depth_map: torch.Tensor | np.ndarray,
#     extrinsic: torch.Tensor | np.ndarray,
#     intrinsic: torch.Tensor | np.ndarray,
#     images: torch.Tensor | np.ndarray,
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Project multi-view unprojected points into each camera (z-buffer).

#     Args:
#       point_map_by_unprojection: (S, H, W, 3)
#       depth_map: (B, S, H, W, 1)
#       extrinsic: (B, S, 3, 4)
#       intrinsic: (B, S, 3, 3)
#       images: (B, S, 3, H, W) in [0, 1]

#     Returns:
#       depth_rendered: (B, S, H, W) float32
#       fov_overlap_mask: (B, S, H, W) bool
#       rgb_rendered: (B, S, H, W, 3) float32 in [0, 1]
#     """
#     S_pm, H, W, _ = point_map_by_unprojection.shape

#     d = _as_numpy(depth_map)[..., 0]  # (B, S, H, W)
#     d0 = d[0]  # (S, H, W) — point_map is from batch 0

#     rgb_maps = _images_b_s_hwc(images)[0]  # (S, H, W, 3)

#     E = _as_numpy(extrinsic)  # (B, S, 3, 4)
#     K = _as_numpy(intrinsic)  # (B, S, 3, 3)
#     B = E.shape[0]
#     S_cam = E.shape[1]

#     pts = []
#     cols = []
#     for s in range(S_pm):
#         ds_ = d0[s]
#         m = np.isfinite(ds_) & (ds_ > 1e-6)
#         safe = _detect_flying_pixels(ds_, rel_thresh=0.1)
#         m = m & safe
#         if np.any(m):
#             pts.append(point_map_by_unprojection[s][m])
#             cols.append(rgb_maps[s][m])

#     out = np.zeros((B, S_cam, H, W), dtype=np.float32)
#     fov_overlap_geom = np.zeros((B, S_cam, H, W), dtype=bool)
#     rgb_out = np.zeros((B, S_cam, H, W, 3), dtype=np.float32)

#     if len(pts) > 0:
#         P = np.concatenate(pts, axis=0).astype(np.float32)
#         Pwh = np.concatenate([P, np.ones((P.shape[0], 1), dtype=np.float32)], axis=1)
#         C = np.concatenate(cols, axis=0).astype(np.float32)

#         for b in range(B):
#             for s in range(S_cam):
#                 Eb = E[b, s].astype(np.float32)
#                 Kb = K[b, s].astype(np.float32)

#                 Pc = (Eb @ Pwh.T).T
#                 z = Pc[:, 2]
#                 valid = z > 0.01
#                 Pc_v = Pc[valid]
#                 z_v = z[valid]
#                 C_v = C[valid]

#                 if z_v.size > 0:
#                     pix = (Kb @ Pc_v.T).T
#                     u = pix[:, 0] / (pix[:, 2] + 1e-5)
#                     v = pix[:, 1] / (pix[:, 2] + 1e-5)

#                     ui_all = np.rint(u)
#                     vi_all = np.rint(v)
#                     inb = (
#                         (ui_all >= 0)
#                         & (ui_all < W)
#                         & (vi_all >= 0)
#                         & (vi_all < H)
#                         & np.isfinite(ui_all)
#                         & np.isfinite(vi_all)
#                     )

#                     ui_b = ui_all[inb].astype(np.int32)
#                     vi_b = vi_all[inb].astype(np.int32)
#                     z_b = z_v[inb].astype(np.float32)
#                     C_b = C_v[inb]

#                     if z_b.size > 0:
#                         depth_min_buf = np.full((H, W), np.inf, dtype=np.float32)
#                         np.minimum.at(depth_min_buf, (vi_b, ui_b), z_b)

#                         hit_count = np.zeros((H, W), dtype=np.int32)
#                         np.add.at(hit_count, (vi_b, ui_b), 1)

#                         hit = hit_count > 0
#                         depth_min_buf[~hit] = 0.0
#                         out[b, s] = depth_min_buf

#                         order = np.argsort(-z_b)
#                         rgb_out[b, s, vi_b[order], ui_b[order]] = C_b[order]

#                 pm_s = point_map_by_unprojection[s].reshape(-1, 3).astype(np.float32)
#                 Pwh_s = np.concatenate(
#                     [pm_s, np.ones((pm_s.shape[0], 1), dtype=np.float32)], axis=1
#                 )
#                 acc_geom = np.zeros(H * W, dtype=bool)
#                 for src in range(S_cam):
#                     if src == s:
#                         continue
#                     Esrc = E[b, src].astype(np.float32)
#                     Ksrc = K[b, src].astype(np.float32)

#                     Pc_g = (Esrc @ Pwh_s.T).T
#                     z_g = Pc_g[:, 2]
#                     pix_g = (Ksrc @ Pc_g.T).T
#                     u_g = pix_g[:, 0] / (pix_g[:, 2] + 1e-6)
#                     v_g = pix_g[:, 1] / (pix_g[:, 2] + 1e-6)

#                     in_front = z_g > 0.01
#                     in_bounds = (u_g >= 0) & (u_g < W) & (v_g >= 0) & (v_g < H)
#                     overlap_flat = in_front & in_bounds & np.isfinite(u_g) & np.isfinite(v_g)
#                     acc_geom |= overlap_flat

#                 fov_overlap_geom[b, s] = acc_geom.reshape(H, W)

#     return out, fov_overlap_geom, rgb_out

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
    S_pm, H, W, _ = point_map_by_unprojection.shape

    d = _as_numpy(depth_map)[..., 0]  # (B, S, H, W)
    d0 = d[0]  # (S, H, W) — point_map is from batch 0

    rgb_maps = _images_b_s_hwc(images)[0]  # (S, H, W, 3)

    E = _as_numpy(extrinsic)  # (B, S, 3, 4)
    K = _as_numpy(intrinsic)  # (B, S, 3, 3)
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
            pts.append(point_map_by_unprojection[s][m])
            cols.append(rgb_maps[s][m])

    out = np.zeros((B, S_cam, H, W), dtype=np.float32)
    fov_overlap_geom = np.zeros((B, S_cam, H, W), dtype=bool)
    rgb_out = np.zeros((B, S_cam, H, W, 3), dtype=np.float32)

    if len(pts) > 0:
        P = np.concatenate(pts, axis=0).astype(np.float32)
        Pwh = np.concatenate([P, np.ones((P.shape[0], 1), dtype=np.float32)], axis=1)
        C = np.concatenate(cols, axis=0).astype(np.float32)

        for b in range(B):
            for s in range(S_cam):
                Eb = E[b, s].astype(np.float32)
                Kb = K[b, s].astype(np.float32)

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
                        (ui_all >= 0)
                        & (ui_all < W)
                        & (vi_all >= 0)
                        & (vi_all < H)
                        & np.isfinite(ui_all)
                        & np.isfinite(vi_all)
                    )

                    ui_b = ui_all[inb].astype(np.int32)
                    vi_b = vi_all[inb].astype(np.int32)
                    z_b = z_v[inb].astype(np.float32)
                    C_b = C_v[inb]

                    if z_b.size > 0:
                        depth_min_buf = np.full((H, W), np.inf, dtype=np.float32)
                        np.minimum.at(depth_min_buf, (vi_b, ui_b), z_b)

                        hit_count = np.zeros((H, W), dtype=np.int32)
                        np.add.at(hit_count, (vi_b, ui_b), 1)

                        hit = hit_count > 0
                        depth_min_buf[~hit] = 0.0
                        out[b, s] = depth_min_buf

                        order = np.argsort(-z_b)
                        rgb_out[b, s, vi_b[order], ui_b[order]] = C_b[order]

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
                    Es = E[b, s].astype(np.float32)
                    Ks = K[b, s].astype(np.float32)

                    Pc_g = (Esrc @ Pwh_s.T).T
                    z_g = Pc_g[:, 2]
                    pix_g = (Ksrc @ Pc_g.T).T
                    u_g = pix_g[:, 0] / (pix_g[:, 2] + 1e-6)
                    v_g = pix_g[:, 1] / (pix_g[:, 2] + 1e-6)
                    in_front = z_g > 0.01
                    in_bounds = (u_g >= 0) & (u_g < W) & (v_g >= 0) & (v_g < H)
                    overlap_own_to_other = in_front & in_bounds & np.isfinite(u_g) & np.isfinite(v_g)

                    pm_src = point_map_by_unprojection[src].reshape(-1, 3).astype(np.float32)
                    Pwh_src = np.concatenate(
                        [pm_src, np.ones((pm_src.shape[0], 1), dtype=np.float32)], axis=1
                    )
                    Pc_g2 = (Es @ Pwh_src.T).T
                    z_g2 = Pc_g2[:, 2]
                    pix_g2 = (Ks @ Pc_g2.T).T
                    u_g2 = pix_g2[:, 0] / (pix_g2[:, 2] + 1e-6)
                    v_g2 = pix_g2[:, 1] / (pix_g2[:, 2] + 1e-6)
                    in_front2 = z_g2 > 0.01
                    in_bounds2 = (u_g2 >= 0) & (u_g2 < W) & (v_g2 >= 0) & (v_g2 < H)
                    valid2 = in_front2 & in_bounds2 & np.isfinite(u_g2) & np.isfinite(v_g2)

                    if np.any(valid2):
                        ui2 = np.clip(np.rint(u_g2[valid2]), 0, W - 1).astype(np.int32)
                        vi2 = np.clip(np.rint(v_g2[valid2]), 0, H - 1).astype(np.int32)
                        hit_from_src = np.zeros((H, W), dtype=bool)
                        hit_from_src[vi2, ui2] = True
                        overlap_other_to_own = hit_from_src.reshape(-1)
                    else:
                        overlap_other_to_own = np.zeros(H * W, dtype=bool)
                    acc_geom |= overlap_own_to_other | overlap_other_to_own

                fov_overlap_geom[b, s] = acc_geom.reshape(H, W)

    return out, fov_overlap_geom, rgb_out

# def geometric_calculation(
#     depth: torch.Tensor | np.ndarray,
#     depth_rendered: torch.Tensor | np.ndarray,
#     *,
#     eps: float = 1e-6,
#     valid_mask: torch.Tensor | np.ndarray,
#     images: torch.Tensor | np.ndarray,
#     rgb_rendered: torch.Tensor | np.ndarray,
# ) -> dict:
#     """
#     Args:
#       depth: (B, S, H, W, 1)
#       depth_rendered: (B, S, H, W)
#       valid_mask: (B, S, H, W) bool
#       images: (B, S, 3, H, W) in [0, 1]
#       rgb_rendered: (B, S, H, W, 3) in [0, 1]
#     """
#     dj = _as_numpy(depth)[..., 0]  # (B, S, H, W)
#     di = _as_numpy(depth_rendered)  # (B, S, H, W)

#     vm = _as_numpy(valid_mask).astype(bool, copy=False)

#     B, S, H, W = dj.shape

#     rgb_src = _images_b_s_hwc(images)  # (B, S, H, W, 3)
#     rgb_ren = np.clip(_as_numpy(rgb_rendered).astype(np.float32, copy=False), 0.0, 1.0)

#     def _shape_stats_1view(
#         y_bhw: np.ndarray,
#         x_bhw: np.ndarray,
#         m_bhw: np.ndarray,
#         rgb_src_bhw3: np.ndarray,
#         rgb_ren_bhw3: np.ndarray,
#     ) -> dict | list[dict]:
#         out = []
#         for b in range(B):
#             y = y_bhw[b]
#             x = x_bhw[b]
#             roi = np.isfinite(y) & np.isfinite(x) & (y > eps) & (x > eps) & m_bhw[b]
#             cnt = int(roi.sum())
#             if cnt == 0:
#                 out.append({
#                     "residual": float("nan"),
#                     "count": 0,
#                     "scale_ref": float("nan"),
#                     "valid_ratio": float(roi.mean()),
#                     "rgb_rmse": float("nan"),
#                 })
#                 continue

#             xv = x.astype(np.float64)
#             yv = y.astype(np.float64)
#             scale_ref = float(np.median(yv))
#             r = np.abs(yv - xv)

#             src_pix = rgb_src_bhw3[b].astype(np.float64)
#             ren_pix = rgb_ren_bhw3[b].astype(np.float64)
#             rgb_rmse = float(np.sqrt(np.mean(np.square(src_pix - ren_pix))))

#             out.append({
#                 "residual": float(np.percentile(r, 90.0)), 
#                 "count": cnt,
#                 "scale_ref": scale_ref,
#                 "valid_ratio": float(roi.mean()),
#                 "rgb_rmse": rgb_rmse,
#             })

#         return out[0] if B == 1 else out

#     result: dict = {}
#     for s in range(S):
#         result[f"view{s}_shape"] = _shape_stats_1view(
#             dj[:, s], di[:, s], vm[:, s], rgb_src[:, s], rgb_ren[:, s]
#         )

#     diffs = []
#     for s in range(S):
#         for b in range(B):
#             diffs.append(rgb_src[b, s] - rgb_ren[b, s])
#     if diffs:
#         all_diff = np.concatenate(diffs, axis=0).astype(np.float64)
#         result["rgb_rmse"] = float(np.sqrt(np.mean(np.square(all_diff))))
#     else:
#         result["rgb_rmse"] = 0.0

#     return result

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

    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    device = "cuda" if torch.cuda.is_available() else "cpu"
    omega_image_resolution = 512
    omega_patch_size = 16
    omega_checkpoint = (
        "/home/thanhnv/data2_disk/VT/1.VinRobotics/1.Projects/Code/VideoWordModelBenchmark/thanhnv_multiview/vggt_omega_checkpoint/vggt_omega_1b_512.pt"
    )

    model = VGGTOmega(patch_size=omega_patch_size).to(device).eval()
    state = torch.load(omega_checkpoint, map_location="cpu")
    model.load_state_dict(state)

    depth_output_dir = "output_depth"

    ep = 255
    frame1 = 5
    frame2 = 5

    # version = "real"
    # version = "good_2fps"
    version = "bad_2fps"
    for frame1, frame2 in zip(range(35), range(35)):
        if version == "real":
            image_names = [
                f"/home/thanhnv/data2_disk/VT/1.VinRobotics/1.Projects/Code/VideoWordModelBenchmark/thanhnv_multiview/data/{version}/episode_{ep:06d}/observation.images.exterior_1_left/frame_{frame1:04d}.png",
                f"/home/thanhnv/data2_disk/VT/1.VinRobotics/1.Projects/Code/VideoWordModelBenchmark/thanhnv_multiview/data/{version}/episode_{ep:06d}/observation.images.exterior_2_left/frame_{frame2:04d}.png",
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
            depth_map_single = preds["depth_single"]

            extrinsic, intrinsic = encoding_to_camera(
                preds["pose_enc"], preds["images"].shape[-2:]
            )

            depth_map_single_aligned = align_depth_scale(depth_map_single, depth_map, max_depth=1.0)

            point_map_by_unprojection_aligned = unproject_depth_map_to_point_map(
                depth_map_single_aligned[0], extrinsic.squeeze(0), intrinsic.squeeze(0)
            )

            depth_rendered_np_single, rendered_valid_mask_single, rgb_rendered_np_single = render_depth_from_point_map(
                point_map_by_unprojection=point_map_by_unprojection_aligned,
                depth_map=depth_map_single_aligned,
                extrinsic=extrinsic,
                intrinsic=intrinsic,
                images=preds["images"]
            )
            depth_rendered_single = torch.from_numpy(depth_rendered_np_single).to(
                device=device, dtype=torch.float32
            )

        res = geometric_calculation2(
            depth=depth_map_single_aligned,
            depth_rendered=depth_rendered_single,
            valid_mask=rendered_valid_mask_single,
            images=preds["images"],
            rgb_rendered=rgb_rendered_np_single,
        )

        cons = geometric_consistency_metric(res)
        print(f"Geometric Consistency Metric of frame {frame1}: {cons}")
