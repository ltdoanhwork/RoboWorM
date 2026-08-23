"""Embodiment checkers: is the behaviour in this video *mechanically* possible?

`metrics/sim2real` asks whether the hand-object geometry is plausible.
These checkers ask a different question, on the robot rather than the object:
does the trajectory implied by the video correspond to a configuration the robot
could physically have been in, and to the motion the GT episode actually performed?

Everything here is pure geometry over a joint trajectory plus a `Kinematics` model.
No simulation rollout, no scene reconstruction, no object meshes — the reasons are in
`README.md`.

Three families, and they answer different questions:

`fk_consistency`
    The paper's headline robot failure. Given a joint trajectory and where the video
    puts the end effector, does forward kinematics agree? A generated arm can look
    like it reaches the target while the implied joint configuration puts the hand
    somewhere else. Validated to 3.3e-08 m on GT, so any residual is signal.

`kinematic_feasibility`
    Joint limits, joint velocity, and joint acceleration against what the real robot
    does. GT violates joint limits 0.000% of the time, so a violation is a property
    of the prediction. Velocity/acceleration ceilings are calibrated *from the GT of
    the same split* rather than from a spec sheet, so they measure "faster than this
    robot ever moves in this data" and need no datasheet per embodiment.

`trajectory_agreement`
    Paired-GT comparison of the recovered trajectory: end-effector path error in
    metres, path-length ratio (frozen vs hallucinated motion), and gripper-signal
    agreement. Metres, not pixels — the whole point of going through kinematics.

Being differenced against the paired GT episode is what makes these robust, exactly
as in `metrics/sim2real`: `episode_index.py` resolves 99.9% of the benchmark to its
GT episode, so every number below has a same-episode reference.
"""

from __future__ import annotations

import numpy as np

from .kinematics import Kinematics


def _finite_diff(x: np.ndarray, dt: float) -> np.ndarray:
    if len(x) < 2:
        return np.zeros((0,) + x.shape[1:])
    return np.diff(x, axis=0) / dt


def fk_consistency(kin: Kinematics, q: np.ndarray, ee_xyz: np.ndarray) -> dict:
    """Does FK of the joint trajectory land where the end effector is claimed to be?

    ``q``: (T,n) joint angles recovered from (or recorded for) the video.
    ``ee_xyz``: (T,3) end-effector positions from an independent source.

    On GT these agree to 3.3e-08 m (measured, 40 DROID episodes), so this is a pure
    error signal rather than a calibration offset.
    """
    r = kin.fk_residual(q, ee_xyz)
    return {
        "fk_residual_mean_m": r["mean_m"],
        "fk_residual_median_m": r["median_m"],
        "fk_residual_p95_m": r["p95_m"],
        "fk_residual_max_m": r["max_m"],
        "n_frames": r["n_frames"],
    }


def kinematic_feasibility(kin: Kinematics, q: np.ndarray, fps: float,
                          vel_ceiling: np.ndarray | None = None,
                          acc_ceiling: np.ndarray | None = None) -> dict:
    """Joint-limit, velocity and acceleration violations of a joint trajectory.

    ``vel_ceiling`` / ``acc_ceiling`` are per-joint ceilings, normally produced by
    :func:`calibrate_ceilings` from the GT of the same split. Calibrating from data
    rather than a datasheet is what keeps this portable across embodiments.
    """
    q = np.atleast_2d(np.asarray(q, dtype=np.float64))[:, : kin.spec.n_arm_joints]
    dt = 1.0 / float(fps)
    out = {"limits": kin.limit_violation(q)}

    vel = _finite_diff(q, dt)
    acc = _finite_diff(vel, dt)
    out["joint_vel_abs_max"] = float(np.abs(vel).max()) if len(vel) else 0.0
    out["joint_acc_abs_max"] = float(np.abs(acc).max()) if len(acc) else 0.0

    for name, arr, ceil in (("vel", vel, vel_ceiling), ("acc", acc, acc_ceiling)):
        if ceil is None or not len(arr):
            out[f"{name}_exceed_frac"] = None
            out[f"{name}_exceed_ratio_max"] = None
            continue
        c = np.asarray(ceil, dtype=np.float64)[: arr.shape[1]]
        ratio = np.abs(arr) / np.maximum(c, 1e-9)
        bad = ratio > 1.0
        out[f"{name}_exceed_frac"] = float(bad.any(1).mean())
        out[f"{name}_exceed_ratio_max"] = float(ratio.max())
    return out


def calibrate_ceilings(q_list: list[np.ndarray], fps: float,
                       percentile: float = 99.5) -> dict:
    """Per-joint velocity/acceleration ceilings from ground-truth trajectories.

    A high percentile rather than the max, so one noisy GT frame does not raise the
    ceiling past anything a prediction could violate.
    """
    dt = 1.0 / float(fps)
    vels, accs = [], []
    for q in q_list:
        q = np.atleast_2d(np.asarray(q, dtype=np.float64))
        v = _finite_diff(q, dt)
        if len(v):
            vels.append(np.abs(v))
        a = _finite_diff(v, dt)
        if len(a):
            accs.append(np.abs(a))
    if not vels:
        return {"error": "no usable trajectories"}
    V, A = np.concatenate(vels, 0), np.concatenate(accs, 0)
    return {
        "fps": float(fps), "percentile": percentile,
        "n_episodes": len(q_list), "n_vel_samples": int(len(V)),
        "vel_ceiling": [float(x) for x in np.percentile(V, percentile, axis=0)],
        "acc_ceiling": [float(x) for x in np.percentile(A, percentile, axis=0)],
    }


def trajectory_agreement(ee_pred: np.ndarray, ee_gt: np.ndarray,
                         grip_pred: np.ndarray | None = None,
                         grip_gt: np.ndarray | None = None) -> dict:
    """Paired-GT comparison of end-effector paths, in metres.

    ``path_len_ratio`` separates the two failure modes a single error number confuses:
    a frozen arm and a wildly over-moving one can share a mean error, but the ratio
    goes to 0 for one and above 1 for the other. ``abs_log_path_ratio`` penalises both
    directions symmetrically, following the same reasoning as the motion-magnitude
    ratio in `metrics/dynamics/README.md`.
    """
    a = np.asarray(ee_pred, dtype=np.float64)[:, :3]
    b = np.asarray(ee_gt, dtype=np.float64)[:, :3]
    n = min(len(a), len(b))
    if n < 2:
        return {"error": f"too few frames ({n})"}
    a, b = a[:n], b[:n]

    err = np.linalg.norm(a - b, axis=1)
    len_pred = float(np.linalg.norm(np.diff(a, axis=0), axis=1).sum())
    len_gt = float(np.linalg.norm(np.diff(b, axis=0), axis=1).sum())
    ratio = len_pred / len_gt if len_gt > 1e-9 else None

    out = {
        "ee_err_mean_m": float(err.mean()),
        "ee_err_median_m": float(np.median(err)),
        "ee_err_p95_m": float(np.percentile(err, 95)),
        "ee_err_max_m": float(err.max()),
        "path_len_pred_m": len_pred,
        "path_len_gt_m": len_gt,
        "path_len_ratio": ratio,
        "abs_log_path_ratio": float(abs(np.log(max(ratio, 1e-6)))) if ratio else None,
        "n_frames": int(n),
    }

    if grip_pred is not None and grip_gt is not None:
        gp = np.asarray(grip_pred, dtype=np.float64).ravel()[:n]
        gg = np.asarray(grip_gt, dtype=np.float64).ravel()[:n]
        if len(gp) == len(gg) == n:
            out["gripper_mae"] = float(np.abs(gp - gg).mean())
            # correlation is undefined when a signal never changes (gripper held shut)
            if gp.std() > 1e-9 and gg.std() > 1e-9:
                out["gripper_corr"] = float(np.corrcoef(gp, gg)[0, 1])
            else:
                out["gripper_corr"] = None
    return out
