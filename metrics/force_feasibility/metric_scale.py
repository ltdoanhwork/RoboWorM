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
    if not lens:
        return {"error": "no end-effector body resolved in the MJCF"}

    best = max(lens, key=lens.get)
    return {"path_len_m": lens[best], "ee_body": best, "all_bodies_m": lens,
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
