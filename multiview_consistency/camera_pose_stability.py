import os
import csv
from pathlib import Path
import torch
try:
    # When imported as a package module
    from .utils import intersect_frames  # pyright: ignore[reportMissingImports]
except Exception:  # pragma: no cover
    # When executed as a standalone script
    from utils import intersect_frames  # type: ignore

def _rot_angle_deg(Rrel: torch.Tensor) -> torch.Tensor:
    # Rrel: (...,3,3)
    tr = Rrel[..., 0, 0] + Rrel[..., 1, 1] + Rrel[..., 2, 2]
    c = 0.5 * (tr - 1.0)
    c = torch.clamp(c, -1.0, 1.0)
    return torch.rad2deg(torch.acos(c))


def compute_pose_stability_metric(
    extrinsics_all: list[torch.Tensor],
    intrinsics_all: list[torch.Tensor],
    *,
    pose_std_rot_scale_deg: float = 5.0,
    pose_std_trans_scale: float = 0.20,
    pose_std_intr_scale: float = 0.05,
    w_rot: float = 0.6,
    w_trans: float = 0.2,
    w_intr: float = 0.2,
    pose_std_view_agg: str = "mean", # "min" | "harmonic" | "mean" (min encourages stability in all views)
) -> float:
    """
    Compute camera-pose consistency over a video from per-frame extrinsics and intrinsics.

    Each item in ``extrinsics_all`` can be ``(1,S,3,4)`` or ``(S,3,4)``.
    Each item in ``intrinsics_all`` can be ``(1,S,3,3)`` or ``(S,3,3)``.
    The score is in [0,1], higher means camera extrinsic and intrinsic are more stable.
    """
    if not extrinsics_all or not intrinsics_all:
        return 0.0
    if len(extrinsics_all) != len(intrinsics_all):
        raise ValueError(f"extrinsics_all and intrinsics_all length mismatch: {len(extrinsics_all)} vs {len(intrinsics_all)}")

    normalized_E = []
    for E in extrinsics_all:
        if E.ndim == 3:
            E = E.unsqueeze(0)
        if E.ndim != 4 or E.shape[-2:] != (3, 4):
            raise ValueError(f"Expected extrinsic shape (1,S,3,4) or (S,3,4), got {tuple(E.shape)}")
        normalized_E.append(E.detach().float().cpu())

    normalized_K = []
    for K in intrinsics_all:
        if K.ndim == 3:
            K = K.unsqueeze(0)
        if K.ndim != 4 or K.shape[-2:] != (3, 3):
            raise ValueError(f"Expected intrinsic shape (1,S,3,3) or (S,3,3), got {tuple(K.shape)}")
        normalized_K.append(K.detach().float().cpu())

    E = torch.cat(normalized_E, dim=0)  # (T,S,3,4)
    K = torch.cat(normalized_K, dim=0)  # (T,S,3,3)
    if E.shape[:2] != K.shape[:2]:
        raise ValueError(f"extrinsic/intrinsic shape mismatch: {tuple(E.shape[:2])} vs {tuple(K.shape[:2])}")
    _, S = E.shape[:2]

    view_scores = []
    for s in range(S):
        R = E[:, s, :, :3]  # (T,3,3)
        t = E[:, s, :, 3]   # (T,3)
        K_view = K[:, s]     # (T,3,3)

        sigma_t = t.std(dim=0, unbiased=False).mean()

        Rref = R[0]
        Rrel = R @ Rref.T
        thetas = _rot_angle_deg(Rrel)  # (T,)
        sigma_R = thetas.std(unbiased=False)

        sR = float(torch.exp(-sigma_R / max(pose_std_rot_scale_deg, 1e-12)).item())
        st = float(torch.exp(-sigma_t / max(pose_std_trans_scale, 1e-12)).item())

        k_vals = torch.stack(
            [
                K_view[:, 0, 0],
                K_view[:, 1, 1],
                K_view[:, 0, 2],
                K_view[:, 1, 2],
            ],
            dim=-1,
        )
        k_ref = k_vals[0].abs().clamp_min(1e-12)
        sigma_K = (k_vals / k_ref).std(dim=0, unbiased=False).mean()
        sK = float(torch.exp(-sigma_K / max(pose_std_intr_scale, 1e-12)).item())

        wsum = w_rot + w_trans + w_intr
        sv = (w_rot * sR + w_trans * st + w_intr * sK) / (wsum if wsum > 0 else 1.0)
        view_scores.append(float(max(0.0, min(1.0, sv))))

    if pose_std_view_agg == "min":
        return float(min(view_scores)) if view_scores else 0.0
    if pose_std_view_agg == "mean":
        return float(sum(view_scores) / max(1, len(view_scores)))
    if pose_std_view_agg == "harmonic":
        if not view_scores:
            return 0.0
        if len(view_scores) == 1:
            return float(view_scores[0])
        a, b = view_scores[0], view_scores[1]
        return float(2.0 * a * b / (a + b + 1e-12))

    raise ValueError(f"Unknown pose_std_view_agg={pose_std_view_agg!r}")


@torch.no_grad()
def main() -> None:
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images

    ep = 8
    episode_dir = f"/pfss/mlde/workspaces/mlde_wsp_MGPATH/thanhnv/data/multiview/thanhnv_multiview/episode_{ep:06d}_generated"
    views = ("view1", "view2")
    max_frames = None
    stride = 1
    pose_std_rot_scale_deg = 5.0
    pose_std_trans_scale = 0.05
    pose_std_intr_scale = 0.01
    pose_std_view_agg = "min"
    out_csv = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()

    frame_ids = intersect_frames(list(views), episode_dir)
    if not frame_ids:
        raise ValueError(f"No common frames found in {episode_dir} for views={views}")

    frame_ids = frame_ids[:: max(1, int(stride))]
    if max_frames is not None:
        frame_ids = frame_ids[: int(max_frames)]

    rows: list[dict] = []
    extrinsics_all: list[torch.Tensor] = []   # each: (1,S,3,4) on CPU
    intrinsics_all: list[torch.Tensor] = []   # each: (1,S,3,3) on CPU

    for fi in frame_ids:
        paths = [os.path.join(episode_dir, v, f"frame_{fi:04d}.png") for v in views]
        images = load_and_preprocess_images(paths).to(device)[None]

        with torch.cuda.amp.autocast(dtype=dtype):
            aggregated_tokens_list, _ = model.aggregator(images)
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        print("fi: ", fi)
        print("extrinsic: ", extrinsic)
        print("intrinsic: ", intrinsic)
        print("--------------------------------")

        extrinsics_all.append(extrinsic.detach().float().cpu())
        intrinsics_all.append(intrinsic.detach().float().cpu())
        rows.append({"frame": int(fi)})

    pose_stability = compute_pose_stability_metric(
        extrinsics_all,
        intrinsics_all,
        pose_std_rot_scale_deg=pose_std_rot_scale_deg,
        pose_std_trans_scale=pose_std_trans_scale,
        pose_std_intr_scale=pose_std_intr_scale,
        pose_std_view_agg=pose_std_view_agg,
    )

    for r in rows:
        r["pose_stability"] = float(pose_stability)

    if out_csv is not None:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["frame", "pose_stability"])
            w.writeheader()
            w.writerows(rows)

    print(rows)
    print(f"Processed {len(rows)} frames from {episode_dir}")
    print(f"Camera pose consistency (extrinsic + intrinsic): {pose_stability:.6f}")


if __name__ == "__main__":
    main()

