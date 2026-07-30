import math
import os
import csv
from collections.abc import Sequence
from pathlib import Path

import torch
try:
    from .utils import intersect_frames  # pyright: ignore[reportMissingImports]
except Exception:  # pragma: no cover
    from utils import intersect_frames  # type: ignore

def _rot_angle_deg(Rrel: torch.Tensor) -> torch.Tensor:
    # Rrel: (...,3,3)
    tr = Rrel[..., 0, 0] + Rrel[..., 1, 1] + Rrel[..., 2, 2]
    c = 0.5 * (tr - 1.0)
    c = torch.clamp(c, -1.0, 1.0)
    return torch.rad2deg(torch.acos(c))


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def _intrinsic_stability_score(K_view: torch.Tensor) -> float:
    """Temporal stability of fx, fy, cx, cy for one view. K_view: (T,3,3)."""
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
    return math.exp(-float(sigma_K.item())) 


def _extrinsic_stability_score(
    E_view: torch.Tensor,
) -> float:
    """Temporal stability of rotation and translation for one view. E_view: (T,3,4)."""
    R = E_view[:, :, :3]
    t = E_view[:, :, 3]

    sigma_t = float(t.std(dim=0, unbiased=False).mean().item())
    Rref = R[0]
    Rrel = R @ Rref.T
    sigma_R = float(_rot_angle_deg(Rrel).std(unbiased=False).item())

    sR = _clamp01(math.exp(-sigma_R / 5.0))
    st = _clamp01(math.exp(-sigma_t / 1.0))
    return (sR + st) / 2.0


def compute_pose_stability_metric(
    extrinsics_all: list[torch.Tensor],
    intrinsics_all: list[torch.Tensor],
) -> float:
    """
    Camera pose stability over a video.

    1. Intrinsic score: mean temporal stability of view0 and view1 intrinsics.
    2. Extrinsic score: temporal stability of view1 (relocating camera) rotation + translation.
    3. Final score: (1 - w_intrinsic) * extrinsic + w_intrinsic * intrinsic.

    Each item in ``extrinsics_all`` can be ``(1,S,3,4)`` or ``(S,3,4)``.
    Each item in ``intrinsics_all`` can be ``(1,S,3,3)`` or ``(S,3,3)``.
    Returns a scalar in [0, 1]; higher is more stable.
    """
    if not extrinsics_all or not intrinsics_all:
        return 0.0
    if len(extrinsics_all) != len(intrinsics_all):
        raise ValueError(
            f"extrinsics_all and intrinsics_all length mismatch: {len(extrinsics_all)} vs {len(intrinsics_all)}"
        )

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
    if S < 2:
        raise ValueError(f"Expected at least 2 views, got S={S}")

    intr_view0 = _intrinsic_stability_score(K[:, 0])
    intr_view1 = _intrinsic_stability_score(K[:, 1])
    intrinsic_score = (intr_view0 + intr_view1) / 2.0

    extrinsic_score = _extrinsic_stability_score(E[:, 1])

    return _clamp01((extrinsic_score + intrinsic_score)/2.0)


@torch.no_grad()
def main() -> None:
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images

    ep = 1
    episode_dir = f"/pfss/mlde/workspaces/mlde_wsp_MGPATH/thanhnv/data/multiview/thanhnv_multiview/episode_{ep:06d}_generated"
    # episode_dir = "/pfss/mlde/workspaces/mlde_wsp_MGPATH/thanhnv/data/multiview/thanhnv_multiview/episode_000255"
    views = ("view1", "view2")
    max_frames = None
    stride = 1
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

    print(f"Processed {len(rows)} frames from {episode_dir}")
    print(f"Camera pose consistency (extrinsic + intrinsic): {pose_stability:.6f}")


if __name__ == "__main__":
    main()

