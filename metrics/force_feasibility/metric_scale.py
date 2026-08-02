"""Metric scale from robot kinematics, not from a guessed object size.

MoGe's point maps are scale-ambiguous, and this tier measures forces, so the scale has
to come from somewhere. The fallback in ``lift_to_3d`` assumes the manipulated object is
~8 cm across -- honest but crude, and it makes the scale a property of the guess rather
than of the episode.

There is a better ruler already in the repo. ``metrics/embodied_exec`` maps any generated
mp4 to its action-labelled GT episode and has validated forward kinematics per embodiment
(Panda reproduces DROID's own end-effector column to 3.3e-08 m). Running FK on the GT
joint trajectory gives the end-effector's path length **in metres**, which is exactly the
quantity the tracked gripper's path length has to be matched against.

This closes the last hand-set constant in the vision bridge: one scalar, fitted per
episode from measured kinematics.

Which hand
----------
For a two-armed robot most of the motion belongs to one arm. Measured on GR1
``episode_003151``: the right hand travels 0.4987 m while the left travels 0.0661 m. The
manipulating hand is picked by path length rather than named, so no per-episode
annotation is needed.

Coverage note: this works wherever ``embodied_exec`` has a validated robot, which after
adding GR1 from robosuite includes ``humanoid/singleview``. Groups without a robot model
(``humanoid/multiview``, Airbot MMK2) keep the object-size fallback, and
``scale_source`` records which was used.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np


def gt_ee_path_length(embodiment: str, view_type: str, split: str, episode_id: str,
                      datasets_root: str = "datasets") -> dict:
    """Forward-kinematics path length of the manipulating end effector, in metres.

    Returns a dict with ``path_len_m`` plus the diagnostics needed to decide whether to
    trust it (which body was used, both hands' lengths, frame count).
    """
    import mujoco
    import pandas as pd

    from metrics.embodied_exec.kinematics import GROUP_ROBOT, REGISTRY, Kinematics

    key = (embodiment, view_type)
    robot = GROUP_ROBOT.get(key)
    if robot is None:
        return {"error": f"no robot model registered for {key}"}
    spec = REGISTRY[robot]
    kin = Kinematics(spec)
    model = kin.model
    data = mujoco.MjData(model)

    eid = int(episode_id.replace("episode_", "")) if isinstance(episode_id, str) else int(episode_id)
    pat = f"{datasets_root}/{embodiment}/{view_type}/{split}/data/chunk-*/episode_{eid:06d}.parquet"
    hits = sorted(glob.glob(pat))
    if not hits:
        return {"error": f"no parquet matching {pat}"}
    df = pd.read_parquet(hits[0])

    col = spec.joint_source if spec.joint_source in df.columns else "observation.state"
    if col not in df.columns:
        return {"error": f"no joint column ({spec.joint_source}) in {hits[0]}"}
    S = np.stack(df[col].values)
    dims = list(spec.state_dims) if spec.state_dims else list(range(S.shape[1]))
    q = S[:, dims]

    # Candidate end-effector bodies: whatever the spec names, plus the conventional
    # left/right pair. Names differ per MJCF (GR1's is spelled `left_hand_eefd`), so
    # missing ones are skipped rather than assumed.
    cands = [spec.ee_body] + ["right_hand_eef", "left_hand_eefd", "right_eef", "left_eef"]
    lens: dict[str, float] = {}
    rgs: dict[str, float] = {}
    for nm in dict.fromkeys(c for c in cands if c):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
        if bid < 0:
            continue
        P = np.empty((len(q), 3))
        for t, qi in enumerate(q):
            data.qpos[:] = 0.0
            if model.nq > 6:
                data.qpos[3] = 1.0   # unit quaternion for a floating base
            data.qpos[kin.qpos_index] = qi
            mujoco.mj_kinematics(model, data)
            P[t] = data.xpos[bid]
        lens[nm] = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
        # Radius of gyration alongside path length. This is what the vision bridge matches
        # against, because path length counts tracker jitter and so is biased upward by it
        # (measured: 7.05x inflation under 20 mm of noise, versus 1.009x for this quantity).
        # Path length is retained for reporting and for the older scale source.
        rgs[nm] = float(np.sqrt(((P - P.mean(0)) ** 2).sum(1).mean()))
    if not lens:
        return {"error": "no end-effector body resolved in the MJCF"}

    best = max(lens, key=lens.get)

    # The vision bridge's ruler is the *segmentation mask's* centroid, and open-vocabulary
    # detection returns the whole arm, not the fingertips. So the comparable FK quantity is the
    # centroid of every moving body on that arm -- not the end-effector, which swings much
    # further. Measured on GR1: EE radius of gyration 0.1160 m against arm-centroid 0.0438 m,
    # a ratio of 2.65. Matching the mask centroid to the EE therefore inflated the metric scale
    # by that factor and made a mustard bottle come out 37.5 cm across; against the arm
    # centroid it lands at 14 cm.
    side = "l_" if best.startswith(("left", "l_")) else "r_"
    chain = []
    for i in range(model.nbody):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        if not (nm.startswith(side) or nm.startswith(best[:4])):
            continue
        if any(k in nm for k in ("thigh", "shank", "foot", "leg")):
            continue
        chain.append(i)
    arm_rg = None
    if chain:
        P = np.empty((len(q), len(chain), 3))
        for t, qi in enumerate(q):
            data.qpos[:] = 0.0
            if model.nq > 6:
                data.qpos[3] = 1.0
            data.qpos[kin.qpos_index] = qi
            mujoco.mj_kinematics(model, data)
            P[t] = data.xpos[chain]
        # Only bodies that actually move belong in the ruler: a body welded to the torso
        # contributes a constant that shrinks the second moment without carrying information.
        moving = [k for k in range(len(chain))
                  if float(np.sqrt(((P[:, k] - P[:, k].mean(0)) ** 2).sum(1).mean())) > 1e-4]
        if moving:
            C = P[:, moving].mean(axis=1)
            arm_rg = float(np.sqrt(((C - C.mean(0)) ** 2).sum(1).mean()))

    return {"path_len_m": lens[best], "rad_gyr_m": rgs[best],
            "arm_centroid_rad_gyr_m": arm_rg, "ee_body": best,
            "all_bodies_m": lens, "all_bodies_rg_m": rgs,
            "n_frames": int(len(q)), "robot": robot, "parquet": hits[0]}


def resolve_from_tracks(tracks_npz: str, datasets_root: str = "datasets") -> dict:
    """Infer (embodiment, view_type, split, episode) from a cheap-tier output path.

    The ``sim2real_out`` layout encodes them as
    ``.../<embodiment>_<view>_<model>/<split>/episode_N/tracks.npz``, mirroring the
    sampling-dataset convention.
    """
    p = Path(tracks_npz).resolve()
    episode = p.parent.name
    split = p.parent.parent.name
    tag = p.parent.parent.parent.name
    parts = tag.split("_")
    emb = view = None
    for i, s in enumerate(parts):
        if s in ("singleview", "multiview"):
            view, emb = s, "_".join(parts[:i])
            break
    if emb is None:
        return {"error": f"could not parse embodiment/view from {tag!r}"}
    out = gt_ee_path_length(emb, view, split, episode, datasets_root)
    out.update(embodiment=emb, view_type=view, split=split, episode=episode)
    return out
