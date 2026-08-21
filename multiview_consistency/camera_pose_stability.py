import math
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

    sR = math.exp(-sigma_R / 5.0)
    st = math.exp(-sigma_t / 1.0)
    return (sR + st) / 2.0

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

def compute_pose_stability_metric(
    extrinsics_all: list[torch.Tensor],
    intrinsics_all: list[torch.Tensor],
) -> float:
    """
    Camera pose stability over a video.

    Each item in ``extrinsics_all``: (1, S, 3, 4).
    Each item in ``intrinsics_all``: (1, S, 3, 3).
    Returns a scalar in [0, 1]; higher is more stable.
    """
    if not extrinsics_all or not intrinsics_all:
        return 0.0

    E = torch.cat([e.detach().float().cpu() for e in extrinsics_all], dim=0)  # (T,S,3,4)
    K = torch.cat([k.detach().float().cpu() for k in intrinsics_all], dim=0)  # (T,S,3,3)

    intr_view0 = _intrinsic_stability_score(K[:, 0])
    intr_view1 = _intrinsic_stability_score(K[:, 1])
    intrinsic_score = (intr_view0 + intr_view1) / 2.0
    extrinsic_score = _extrinsic_stability_score(E[:, 1])
    return (extrinsic_score + intrinsic_score) / 2.0

@torch.no_grad()
def main() -> None:
    import os
    import sys
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _physics_root = os.path.abspath(os.path.join(_this_dir, ".."))
    if _physics_root not in sys.path:
        sys.path.insert(0, _physics_root)

    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    device = "cuda" if torch.cuda.is_available() else "cpu"
    omega_patch_size = 16
    omega_image_resolution = 512
    omega_checkpoint = (
        "/home/thanhnv/data2_disk/VT/1.VinRobotics/1.Projects/Code/VideoWordModelBenchmark/thanhnv_multiview/vggt_omega_checkpoint/vggt_omega_1b_512.pt"
    )

    model = VGGTOmega(patch_size=omega_patch_size).to(device).eval()
    state = torch.load(omega_checkpoint, map_location="cpu")
    model.load_state_dict(state)

    ep = 397

    version = "real"
    # version = "good_2fps"
    # version = "bad_2fps"

    if version == "real":
        episode_dir = f"/home/thanhnv/data2_disk/VT/1.VinRobotics/1.Projects/Code/VideoWordModelBenchmark/thanhnv_multiview/data/real/episode_{ep:06d}"
        views = ("observation.images.exterior_1_left", "observation.images.exterior_2_left")
    else:
        episode_dir  = f"/home/thanhnv/data2_disk/VT/1.VinRobotics/1.Projects/Code/VideoWordModelBenchmark/thanhnv_multiview/data/good_2fps/{version}/episode_{ep:06d}"
        views = ("view1", "view2")

    max_frames = None
    stride = 1
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
        image_names = [os.path.join(episode_dir, v, f"frame_{fi:04d}.png") for v in views]
        images = load_and_preprocess_images(
            image_names,
            image_resolution=omega_image_resolution,
            patch_size=omega_patch_size,
        ).to(device)

        preds = model(images)
        extrinsic, intrinsic = encoding_to_camera(
            preds["pose_enc"], preds["images"].shape[-2:]
        )
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

    print(f"Processed {len(rows)} frames from {episode_dir}")
    print(f"Camera pose consistency (extrinsic + intrinsic): {pose_stability:.6f}")


if __name__ == "__main__":
    main()

