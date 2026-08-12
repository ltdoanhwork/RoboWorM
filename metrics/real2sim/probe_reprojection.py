"""Is real2sim actually blocked, or was the calibration always there? A reprojection test.

Every metric-scale estimator tried so far was an *indirect* ruler: match some statistic of the
vision-derived arm (path length, radius of gyration, static extent) against the same statistic
computed from FK, and read off metres per unit. All of them failed differently, and
`probe_arm_split` recorded strategy 3 -- prompting SAM with FK-projected hand points -- as
impossible because "it needs the ego camera's intrinsics and its pose in the robot frame, and this
dataset records neither".

**That claim is wrong, and this probe tests it.** The pieces are present, just not in the dataset
metadata:

* the GR1 MJCF (`checkpoints/mjcf_robosuite/gr1/robot.xml`) defines camera `robotview` mounted on
  body `head_pitch` at `pos=[0.13, 0, 0.09]` with `fovy=50` — that is an extrinsic *and* an
  intrinsic;
* `modality.json` puts `neck` at state dims **19:22**, i.e. the three head joints
  (`head_yaw/head_roll/head_pitch`) are recorded per frame, so the camera's pose is not assumed,
  it is driven by GT.

If this chain is right, scale stops being something to estimate: a calibrated camera plus FK in
metres gives absolute geometry directly, and every indirect ruler becomes unnecessary. If it is
wrong, that is worth knowing before more rulers are invented.

The test needs no ground-truth annotation. Project the FK hand position through `robotview` and
compare it with the *hand mask's* centroid, which was obtained independently by detection and
tracking. Two independent pipelines agreeing on where the hand is in the image is strong evidence;
the null hypothesis is easy to state and easy to beat: reprojection error should be far below what
a fixed guess at image centre would give.

Run:
    models/dreamgen/.venv/bin/python -m metrics.real2sim.probe_reprojection
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np


REPO = os.environ.get("REPO", os.getcwd())
ROOT = os.environ.get(
    "ROOT", "metrics/sim2real_out/dense/humanoid_singleview_dreamgen/makovian")

# From modality.json, not guessed.
NECK = slice(19, 22)
NECK_JOINTS = ("head_yaw", "head_roll", "head_pitch")


def camera_matrix(model, cam_id: int, W: int, H: int) -> np.ndarray:
    """Pinhole K from the MJCF camera's vertical FOV.

    MuJoCo's `fovy` is the full vertical field of view in degrees, so the focal length in pixels
    follows from the image height alone; the horizontal focal length is the same under square
    pixels, which is what an unmodified pinhole render assumes.
    """
    fovy = float(model.cam_fovy[cam_id])
    f = 0.5 * H / np.tan(np.deg2rad(fovy) * 0.5)
    return np.array([[f, 0.0, W / 2.0], [0.0, f, H / 2.0], [0.0, 0.0, 1.0]])


def fk_and_project(K, state: np.ndarray, W: int, H: int, cam_name: str = "robotview",
                   bodies: tuple[str, ...] = ("left_eef", "right_eef")) -> dict:
    """Forward-kinematics the whole body, then project chosen bodies through the head camera."""
    mujoco = K.mujoco
    m, d = K.model, K.data

    d.qpos[:] = 0.0
    # Arm joints via the spec's resolved mapping (the same path the FK metrics already trust).
    d.qpos[K.qpos_index] = state[K.state_dims]
    # Neck, so the camera pose comes from GT rather than a default pose.
    for name, val in zip(NECK_JOINTS, state[NECK]):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            d.qpos[m.jnt_qposadr[jid]] = float(val)
    mujoco.mj_forward(m, d)

    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cid < 0:
        raise ValueError(f"no camera {cam_name!r}")
    cam_pos = d.cam_xpos[cid].copy()
    cam_mat = d.cam_xmat[cid].reshape(3, 3).copy()
    # MuJoCo cameras look down -Z with +Y up; the pinhole convention looks down +Z with +Y down.
    R_wc = cam_mat @ np.diag([1.0, -1.0, -1.0])
    Kmat = camera_matrix(m, cid, W, H)

    out = {"cam_pos": cam_pos}
    for b in bodies:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
        if bid < 0:
            out[b] = None
            continue
        p_w = d.xpos[bid].copy()
        p_c = R_wc.T @ (p_w - cam_pos)
        if p_c[2] <= 1e-6:
            out[b] = None  # behind the camera
            continue
        uv = Kmat @ p_c
        out[b] = (float(uv[0] / uv[2]), float(uv[1] / uv[2]), float(p_c[2]))
    return out


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask)
    if len(xs) < 10:
        return None
    return float(xs.mean()), float(ys.mean())


def main() -> None:
    import pandas as pd

    from metrics.embodied_exec.kinematics import Kinematics
    from metrics.force_feasibility.lift_to_3d import unpack_masks

    K = Kinematics.for_group("humanoid", "singleview")
    if K is None:
        raise SystemExit("no kinematics for humanoid/singleview")
    print(f"MJCF: {K.mjcf_path}")
    print(f"arm state dims -> qpos: {list(zip(K.state_dims, K.qpos_index))[:4]} ...")

    eps = sorted(d for d in glob.glob(f"{ROOT}/episode_*")
                 if os.path.exists(f"{d}/tracks.npz"))
    print(f"\n{'episode':16s} {'n':>4s} {'err_px':>8s} {'err_%diag':>10s} "
          f"{'null_px':>8s} {'depth_m':>8s}  beats null?")
    all_err, all_null = [], []
    for ep_dir in eps:
        ep = os.path.basename(ep_dir)
        z = np.load(f"{ep_dir}/tracks.npz", allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        hw = tuple(meta["mask_hw"])
        T = int(meta["num_frames"])
        mg = unpack_masks(z, "v0_gt_gripper", hw, T)
        H, W = hw

        eid = int(ep.replace("episode_", ""))
        hits = sorted(glob.glob("datasets/humanoid/singleview/makovian/data/chunk-*/"
                                f"episode_{eid:06d}.parquet"))
        if not hits:
            continue
        S = np.stack(pd.read_parquet(hits[0])["observation.state"].values)

        errs, depths = [], []
        for t in range(0, min(T, len(mg)), 3):
            c = mask_centroid(mg[t])
            if c is None:
                continue
            # Video and parquet have different lengths, so map proportionally -- the same
            # convention the grasp-window screening uses.
            row = int(round(t / max(min(T, len(mg)) - 1, 1) * (len(S) - 1)))
            pr = fk_and_project(K, S[row], W, H)
            pts = [pr[b] for b in ("left_eef", "right_eef") if pr.get(b)]
            if not pts:
                continue
            # The mask is the *arm*, so compare against whichever hand projects closest: the
            # test is whether the calibration lands the hand in the right place at all, not
            # which hand the tracker happened to follow.
            d_best = min(np.hypot(u - c[0], v - c[1]) for u, v, _ in pts)
            errs.append(d_best)
            depths.append(min(p[2] for p in pts))
        if not errs:
            print(f"{ep:16s} {'-':>4s}   (no measurable frames)")
            continue
        diag = float(np.hypot(H, W))
        # Null model: predict the image centre every frame. Any real calibration must beat it.
        nulls = []
        for t in range(0, min(T, len(mg)), 3):
            c = mask_centroid(mg[t])
            if c is not None:
                nulls.append(np.hypot(c[0] - W / 2.0, c[1] - H / 2.0))
        e, nl = float(np.median(errs)), float(np.median(nulls))
        all_err += errs
        all_null += nulls
        print(f"{ep:16s} {len(errs):4d} {e:8.1f} {e / diag * 100:9.1f}% {nl:8.1f} "
              f"{np.median(depths):8.2f}  {'YES' if e < nl else 'no'}")

    if all_err:
        e, nl = float(np.median(all_err)), float(np.median(all_null))
        print(f"\npooled: median reprojection error {e:.1f} px vs null (image centre) "
              f"{nl:.1f} px -> ratio {e / max(nl, 1e-9):.2f}")
        print("A ratio well below 1 means the MJCF camera + GT neck angles really do describe "
              "this view, i.e. the calibration exists and metric scale is directly available.")


if __name__ == "__main__":
    main()
