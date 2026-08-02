"""Render the *real* robot in the fitted scene, and measure whether the proxy agrees with it.

The proxy scene represents the gripper as two boxes because that is all the *prediction*
provides: a generated video has no joint angles, only pixels. Ground truth does have them, so
for the GT side the actual GR1 can be posed from `observation.state` and drawn.

That is worth more than a nicer picture. The two boxes come from the vision bridge (mask ->
MoGe -> metric scale); the real hand comes from parquet joints through validated FK. They are
independent estimates of the same physical thing, so comparing them **measures the vision
bridge's error** -- a number the tier did not previously have.

The two live in different frames (MoGe camera vs robot base) and no extrinsics exist, so they
are registered by Procrustes on the trajectories. Two things make this a test rather than a
fit: the alignment has 7 parameters against ~123x3 observations, and the recovered *scale*
should come out near 1.0 if the independently-derived metric scale was right. Both the
residual and that scale are reported.

Usage:
    MUJOCO_GL=egl python -m metrics.force_feasibility.render_robot \
        --tracks .../episode_002346/tracks.npz --lift /tmp/sim_vis/ep002346_gt_lift.npz \
        --out /tmp/sim_vis/ep002346_robot
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def procrustes(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Similarity transform taking ``A`` onto ``B``: returns ``R, t, s, rms``.

    Rotation is forced to be proper (det = +1) rather than allowing a reflection, because a
    reflected robot is not a registration failure this should hide.
    """
    ca, cb = A.mean(0), B.mean(0)
    A0, B0 = A - ca, B - cb
    U, S, Vt = np.linalg.svd(A0.T @ B0)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    varA = float((A0 ** 2).sum())
    s = float((S * [1.0, 1.0, d]).sum() / varA) if varA > 1e-12 else 1.0
    t = cb - s * (R @ ca)
    rms = float(np.sqrt(((s * (A0 @ R.T) - B0) ** 2).sum(1).mean()))
    return R, t, s, rms


def _R_to_quat(R: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(1.0 + np.trace(R), 1e-12)) / 2.0
    if w > 1e-6:
        return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w),
                         (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])
    k = int(np.argmax(np.diag(R)))
    v = np.zeros(3)
    v[k] = np.sqrt(max((R[k, k] + 1.0) / 2.0, 1e-12))
    return np.array([0.0, *v])


def build_scene_xml(gr1_path: str, obj_half, support_z: float) -> str:
    """GR1 plus the manipulated object and its support plane, in the lifted frame."""
    ox, oy, oz = obj_half
    return f"""<mujoco model="gr1_scene">
  <include file="{gr1_path}"/>
  <visual>
    <headlight ambient="0.5 0.5 0.5" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/>
  </visual>
  <asset>
    <texture name="ff_grid" type="2d" builtin="checker" rgb1="0.80 0.82 0.85"
             rgb2="0.92 0.94 0.96" width="300" height="300"/>
    <material name="ff_grid_mat" texture="ff_grid" texrepeat="18 18" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light pos="0.5 -0.5 {support_z + 1.5:.4f}" dir="-0.3 0.3 -1" diffuse="0.7 0.7 0.7"/>
    <geom name="support" type="plane" pos="0 0 {support_z:.6f}" size="2 2 0.02"
          material="ff_grid_mat"/>
    <body name="vis_object" mocap="true" pos="0 0 {support_z + oz:.6f}">
      <geom name="vis_obj" type="box" size="{ox:.6f} {oy:.6f} {oz:.6f}"
            rgba="0.95 0.55 0.15 1" contype="0" conaffinity="0"/>
    </body>
    <body name="proxy0" mocap="true" pos="0 -0.2 {support_z:.6f}">
      <geom type="box" size="{max(ox,1e-3)*0.8:.6f} {max(oy,1e-3)*0.5:.6f} {max(oz,1e-3)*0.8:.6f}"
            rgba="0.20 0.45 0.85 0.45" contype="0" conaffinity="0"/>
    </body>
    <body name="proxy1" mocap="true" pos="0 0.2 {support_z:.6f}">
      <geom type="box" size="{max(ox,1e-3)*0.8:.6f} {max(oy,1e-3)*0.5:.6f} {max(oz,1e-3)*0.8:.6f}"
            rgba="0.20 0.45 0.85 0.45" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--lift", required=True, help="cached *_lift.npz from render_sim")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=560)
    ap.add_argument("--height", type=int, default=420)
    a = ap.parse_args()

    import mujoco
    import pandas as pd

    from metrics.embodied_exec.kinematics import Kinematics
    from metrics.force_feasibility.metric_scale import resolve_from_tracks

    sc = resolve_from_tracks(a.tracks)
    K = Kinematics.for_group(sc["embodiment"], sc["view_type"])
    df = pd.read_parquet(sc["parquet"])
    state = np.stack(df["observation.state"].to_numpy())
    Q = K.select_state(state)

    z = np.load(a.lift, allow_pickle=True)
    finger = z["finger_pos"]
    obj_pos = z["obj_pos"]
    T = len(obj_pos)
    # Compare FK against the gripper *body centroid* when available. Comparing it to the
    # contact point measures the wrong thing: that point is chosen per frame as the gripper
    # pixels nearest the object, so it slides over the arm and does not track a material point.
    if "grip_body_pos" in z and z["grip_body_pos"].ndim == 2:
        grip_track = z["grip_body_pos"]
        track_name = "gripper body centroid"
    else:
        grip_track = finger.mean(axis=1)
        track_name = "contact-point midpoint (no grip_body_pos in cache)"
    print(f"[robot] comparing FK to: {track_name}")

    # GT covers the full episode, the mask window may be shorter; the cheap tier matched them
    # by proportion of the episode, so the same assumption is used here.
    idx = np.linspace(0, len(Q) - 1, T).round().astype(int)
    ee_all = K.fk_frames(Q[idx])[0] if hasattr(K, "fk_frames") else K.fk(Q[idx])
    ee = np.asarray(ee_all).reshape(T, -1)[:, :3]

    ok = np.isfinite(grip_track).all(1) & np.isfinite(ee).all(1)
    R, t, s, rms = procrustes(ee[ok], grip_track[ok])
    # Rigid-only alignment: trust the independently-derived metric scale rather than letting
    # the fit absorb a scale. Reported alongside the free-scale value because the gap between
    # them is itself the diagnostic (a free scale collapsing toward 0 means the two shapes are
    # poorly correlated, not that the scale is small).
    A0, B0 = ee[ok] - ee[ok].mean(0), grip_track[ok] - grip_track[ok].mean(0)
    U, S, Vt = np.linalg.svd(A0.T @ B0)
    dsign = np.sign(np.linalg.det(Vt.T @ U.T))
    R_rigid = Vt.T @ np.diag([1.0, 1.0, dsign]) @ U.T
    rms_rigid = float(np.sqrt(((A0 @ R_rigid.T - B0) ** 2).sum(1).mean()))
    t_rigid = grip_track[ok].mean(0) - (R_rigid @ ee[ok].mean(0))

    # Shuffle control: the same fit against a time-permuted version of the same trajectory.
    # Without it, "rms = 34 mm" reads as agreement when it is in fact barely better than
    # matching two unrelated paths of similar length.
    rng = np.random.default_rng(0)
    shuf = []
    for _ in range(200):
        fake = B0[rng.permutation(len(B0))]
        U2, S2, V2 = np.linalg.svd(A0.T @ fake)
        d2 = np.sign(np.linalg.det(V2.T @ U2.T))
        va = float((A0 ** 2).sum())
        s2 = float((S2 * [1.0, 1.0, d2]).sum() / va) if va > 1e-12 else 1.0
        R2 = V2.T @ np.diag([1.0, 1.0, d2]) @ U2.T
        shuf.append(float(np.sqrt(((s2 * (A0 @ R2.T) - fake) ** 2).sum(1).mean())))
    sh_m, sh_s = float(np.mean(shuf)), float(np.std(shuf))

    obj_span = 2.0 * float(np.max(z["obj_half"]))
    print(f"[robot] Procrustes FK->vision: free-scale rms={rms * 1000:.1f} mm (s={s:.3f}), "
          f"rigid rms={rms_rigid * 1000:.1f} mm, n={int(ok.sum())}/{T}")
    print(f"[robot] shuffle control: {sh_m * 1000:.1f} +- {sh_s * 1000:.1f} mm "
          f"-> real fit is only {(1 - rms / sh_m) * 100:.0f}% better than a permuted path")
    print(f"[robot] object size {obj_span * 100:.1f} cm; residual = "
          f"{rms / max(obj_span, 1e-9):.2f}x the object")
    R, t = R_rigid, t_rigid

    gr1 = os.path.abspath(K.mjcf_path)
    model = mujoco.MjModel.from_xml_string(
        build_scene_xml(gr1, tuple(z["obj_half"]), float(z["support_z"])))
    data = mujoco.MjData(model)
    mid = {n: model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)]
           for n in ("vis_object", "proxy0", "proxy1")}
    base_q = _R_to_quat(R)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    # Frame the object and the hand that is near it, not the whole humanoid: the point of the
    # picture is whether the proxy contacts land where the real fingers are.
    cam.lookat[:] = np.nanmean(obj_pos, axis=0)
    cam.distance = 0.55
    cam.azimuth, cam.elevation = 160.0, -22.0

    frames = []
    with mujoco.Renderer(model, a.height, a.width) as r:
        for k in range(T):
            mujoco.mj_resetData(model, data)
            # Floating base carries the registration; joints carry the real GT pose.
            data.qpos[0:3] = t
            data.qpos[3:7] = base_q
            data.qpos[K.qpos_index] = Q[idx[k]]
            data.mocap_pos[mid["vis_object"]] = obj_pos[k]
            data.mocap_pos[mid["proxy0"]] = finger[k, 0]
            data.mocap_pos[mid["proxy1"]] = finger[k, 1]
            mujoco.mj_forward(model, data)
            r.update_scene(data, camera=cam)
            frames.append(r.render())

    import imageio.v3 as iio
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    iio.imwrite(f"{a.out}.mp4", np.stack(frames), fps=16)
    iio.imwrite(f"{a.out}_t0.png", frames[0])
    iio.imwrite(f"{a.out}_mid.png", frames[T // 2])
    print(f"[robot] {a.out}.mp4  {a.out}_t0.png  {a.out}_mid.png")
    os._exit(0)


if __name__ == "__main__":
    main()
