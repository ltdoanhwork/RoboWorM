"""Robot kinematics: the part of "convert to sim" that scales.

The expensive tier's grounding is the robot's own kinematic model, not a
reconstruction of the scene. That choice is what makes the tier applicable to the
whole benchmark: three robot models cover 100% of the episodes, whereas a digital
twin is per-scene and would cover a handful of tasks (RoboWM-Bench shipped 8+8+6
tasks and had a real-to-sim asset company as a co-author).

What this buys is the paper's own headline robot failure mode, which no pixel or mask
metric can see:

    the end effector may appear to reach the target in the video, while the recovered
    joint configuration yields a mismatched end-effector pose under the robot's true
    forward kinematics

Checking that needs exactly two things: forward kinematics, and joint limits. Both
come from the MJCF. No meshes, no object poses, no camera intrinsics, no simulation
rollout.

Validated on DROID (`FrankaPanda`, all 260 `single_arm/multiview` episodes):

    max |FK(joint_position) - observation.state.cartesian_position| = 3.3e-08 m
    ground-truth joint-limit violation rate (with `limit_slack`)    = 0.000%

An exact match to floating-point noise means the menagerie Panda *is* DROID's robot,
in the same joint order and the same base frame, so FK residual is a usable error
signal rather than a calibration offset.

The limit check needed one measured correction. Against the raw MJCF range, GT
violates on 0.186% of frames — and all 138 of those frames are joint6, whose MJCF
lower bound (-0.0175 rad) is tighter than the physical robot revision DROID was
recorded on. `PANDA_LIMIT_SLACK` widens that one joint by 0.5 rad, taking GT to
0.000%. That 0% GT rate is the whole point: it is what makes a violation on a
*prediction* mean something.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Searched in order; override with $MJCF_ROOT. Kept outside the repo on purpose —
# these are third-party assets, not our code.
MJCF_SEARCH = [
    Path(os.environ.get("MJCF_ROOT", "/tmp/mj_men")),
    _REPO_ROOT / "checkpoints/mjcf",
    # robosuite assets (GR1 and the real gripper models) live in a separate clone
    Path(os.environ.get("RS_MJCF_ROOT", _REPO_ROOT / "checkpoints/mjcf_robosuite")),
]


@dataclass
class RobotSpec:
    """Everything needed to map a dataset's state vector onto a kinematic model."""

    name: str
    mjcf_relpath: str
    ee_body: str
    n_arm_joints: int
    # slice of the dataset column that holds arm joint angles
    joint_source: str            # parquet column name
    joint_slice: tuple[int, int]
    # optional: column holding the EE pose FK should reproduce, for validation
    ee_source: str | None = None
    ee_slice: tuple[int, int] = (0, 3)
    # per-joint slack (rad) added to the MJCF range before calling something a
    # violation. See GT_LIMIT_NOTES: the real robot's joint range does not always
    # match the MJCF's, so a small measured slack keeps GT at a 0% violation rate.
    limit_slack: tuple[float, ...] = ()
    # dims to include in the joint-limit check; None = all. Used to exclude dims whose
    # dataset units do not match the MJCF (see ALOHA_GRIPPER_DIMS).
    limit_check_dims: tuple[int, ...] | None = None
    # dataset state dim -> MJCF qpos index. Required whenever the MJCF's qpos layout
    # differs from the dataset's (ALOHA: 16 qpos vs 14 dataset dims).
    joint_index_map: tuple[int, ...] | None = None
    # Preferred over `joint_index_map`: name the MJCF joints and let MuJoCo resolve the
    # indices. Hand-written index arrays conflate two different things -- a joint's id
    # (which indexes `jnt_range`) and its address in `qpos`. They coincide only when
    # every joint is a 1-DoF hinge starting at qpos 0, which is true for Panda and
    # ALOHA but NOT for a floating-base robot: GR1's free joint occupies qpos 0..6, so
    # hinge j sits at qpos j+6 while its range is still at jnt_range[j]. Naming the
    # joints removes that class of silent misalignment entirely.
    joint_names: tuple[str, ...] | None = None
    # dataset dims to gather, aligned with `joint_names`. Needed when the dims that map
    # to the model are not the leading contiguous block (GR1: 44-D state with the two
    # 6-DoF hands interleaved between arm/leg/neck/waist groups).
    state_dims: tuple[int, ...] | None = None
    notes: str = ""
    _model: object = field(default=None, repr=False, compare=False)


# Measured on all 260 DROID single_arm/multiview episodes (74,116 frames):
# GT exceeds the MJCF joint range on 0.186% of frames, and *every* one of those 138
# frames is joint6 (index 5), whose MJCF lower bound is a very tight -0.0175 rad.
# Only 3 episodes are involved, worst excess 0.485 rad. This is a robot-revision
# difference between the physical Panda and the menagerie model, not DROID violating
# physics, so joint6 gets 0.5 rad of slack and every other joint gets none. With the
# slack applied the GT violation rate is 0.000%, which is what makes a prediction-side
# violation meaningful.
GT_LIMIT_NOTES = "franka joint6 lower bound differs from the physical robot revision"
PANDA_LIMIT_SLACK = (0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0)

# Bimanual (ALOHA-like) measured over 40 episodes / 33,545 frames of
# `bimanual/singleview`, comparing GT `observation.state[:14]` against the ALOHA MJCF:
#
#   frames violating MJCF range : 33373 / 33545 (99.49%)
#   per-dim                     : [0]*6 + [31307] + [0]*6 + [31211]
#
# All 12 ARM joints fit strictly inside ALOHA's ranges, which is real evidence the arm
# model matches the physical robot. Every violation is the 2 GRIPPER dims, and it is a
# UNIT MISMATCH, not physics:
#
#   GT gripper   [-0.002, 1.000]  normalised open/close
#   MJCF gripper [ 0.000, 0.041]  metres of finger travel
#
# So the gripper dims must be rescaled before any limit or aperture claim. Until that
# calibration is validated the gripper is excluded from the limit check via
# `limit_check_dims`, otherwise bimanual reports a meaningless 99.49%.
#
# CAREFUL — the first version of this check was WRONG. ALOHA's MJCF has 16 qpos (TWO
# finger joints per side), while the dataset has 14 dims (ONE gripper value per side).
# Taking `jnt_range[:14]` therefore silently misaligns everything after index 5:
# MJCF index 6,7 are both LEFT fingers and `right/wrist_rotate` lands at 13. That
# mis-mapping is what produced a bogus "12/12 arm joints fit" reading. The dataset ->
# MJCF joint mapping must be explicit, hence `joint_index_map`.
ALOHA_GRIPPER_DIMS = (6, 13)
# dataset dim -> MJCF qpos index
ALOHA_JOINT_INDEX_MAP = (0, 1, 2, 3, 4, 5, 6,      # left arm 6 + left gripper
                         8, 9, 10, 11, 12, 13, 14)  # right arm 6 + right gripper


# GR1 (Fourier humanoid, `humanoid/singleview`) was previously BLOCKED: it is absent
# from mujoco_menagerie. robosuite ships it (`robosuite/models/assets/robots/gr1`),
# which unblocks this group.
#
# The dataset's 44-D `observation.state` decomposes, per its own modality.json, as
#   left_arm 7 | left_hand 6 | left_leg 6 | neck 3 | right_arm 7 | right_hand 6 |
#   right_leg 6 | waist 3
# and the MJCF has exactly 32 hinges = waist 3 + head 3 + arm 7x2 + leg 6x2, matching
# every group. The two 6-DoF hands live in separate files (fourier_{left,right}_hand.xml,
# 11 hinges each) and are excluded here -- 44 - 12 = 32.
#
# Verification, ALL 260 episodes / 77,890 frames of real GT against the MJCF ranges:
#   values out of range      7,087 of 2,492,480 (frame, joint) pairs = 0.284%
#   joints implicated        ONLY 4 of 32 -- r_shoulder_roll 4.89%, l_shoulder_roll 3.74%,
#                            l_elbow_pitch 0.36%, r_elbow_pitch 0.10%; 28/32 never violate
#   excess magnitude         p50 0.0035, p99 0.0208, max 0.0468 rad (2.68 deg)
#
# Every implicated joint has a suspiciously round bound of exactly 0.000 and the excess
# is at most 2.7 degrees. A wrong joint ORDER within a group would produce excursions of
# order 1 rad, not 0.03, so this is model conservatism at the boundary -- the same pattern
# as PANDA_LIMIT_SLACK. Measured slack sweep: 0.02 rad leaves 1.13% of the excesses,
# 0.03 leaves 0.085%, and 0.05 rad (2.9 deg) leaves 0.000%. Take 0.05 on those four
# joints and none elsewhere, which puts GT at a 0.000% violation rate and makes a
# prediction-side violation meaningful.
#
# Note the joint ORDER within each group remains an assumption: GR1 has no independent EE
# column, so FK cannot be cross-validated the way Panda's was (3.3e-08 m). The limit
# check is consistent with the mapping but is weaker evidence -- it can refute a bad
# mapping, not confirm a good one.
GR1_ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch",
           "wrist_yaw", "wrist_roll", "wrist_pitch")
GR1_LEG = ("leg_hip_roll", "leg_hip_yaw", "leg_hip_pitch", "leg_knee_pitch",
           "leg_ankle_pitch", "leg_ankle_roll")
GR1_JOINT_NAMES = (
    tuple(f"l_{j}" for j in GR1_ARM)                     # state 0:7   left_arm
    + tuple(f"l_{j}" for j in GR1_LEG)                   # state 13:19 left_leg
    + ("head_yaw", "head_roll", "head_pitch")            # state 19:22 neck
    + tuple(f"r_{j}" for j in GR1_ARM)                   # state 22:29 right_arm
    + tuple(f"r_{j}" for j in GR1_LEG)                   # state 35:41 right_leg
    + ("torso_waist_yaw", "torso_waist_pitch", "torso_waist_roll")  # state 41:44 waist
)
GR1_STATE_DIMS = (tuple(range(0, 7)) + tuple(range(13, 19)) + tuple(range(19, 22))
                  + tuple(range(22, 29)) + tuple(range(35, 41)) + tuple(range(41, 44)))
# the four joints measured above, by index into GR1_JOINT_NAMES
GR1_SLACK_DIMS = (1, 3, 17, 19)   # l_shoulder_roll, l_elbow_pitch, r_shoulder_roll, r_elbow_pitch
GR1_LIMIT_SLACK = tuple(0.05 if i in GR1_SLACK_DIMS else 0.0 for i in range(32))


REGISTRY: dict[str, RobotSpec] = {
    "franka_panda": RobotSpec(
        name="franka_panda",
        mjcf_relpath="franka_emika_panda/panda.xml",
        ee_body="hand",
        n_arm_joints=7,
        joint_source="observation.state.joint_position",
        joint_slice=(0, 7),
        ee_source="observation.state.cartesian_position",
        ee_slice=(0, 3),
        limit_slack=PANDA_LIMIT_SLACK,
        notes="DROID single_arm. FK reproduces GT cartesian_position to 3.3e-08 m.",
    ),
    "aloha": RobotSpec(
        name="aloha",
        mjcf_relpath="aloha/aloha.xml",
        ee_body="left/gripper_link",
        n_arm_joints=14,
        joint_source="observation.state",
        joint_slice=(0, 14),
        ee_source=None,                     # no independent EE column in this dataset
        # exclude the 2 gripper dims: dataset is normalised 0-1, MJCF is metres
        limit_check_dims=tuple(i for i in range(14) if i not in ALOHA_GRIPPER_DIMS),
        joint_index_map=ALOHA_JOINT_INDEX_MAP,
        notes=("bimanual qpos(0:14) = 2 x (6 arm + 1 gripper); ALOHA MJCF has 16 qpos "
               "(2 fingers/side) so joint_index_map is REQUIRED. Gripper dims need unit "
               "calibration (dataset 0-1 vs MJCF metres)."),
    ),
    "gr1": RobotSpec(
        name="gr1",
        mjcf_relpath="gr1/robot.xml",
        ee_body="left_hand_eefd",
        n_arm_joints=32,
        joint_source="observation.state",
        joint_slice=(0, 44),
        ee_source=None,                     # no EE column -> FK not cross-validatable
        joint_names=GR1_JOINT_NAMES,
        state_dims=GR1_STATE_DIMS,
        limit_slack=GR1_LIMIT_SLACK,
        notes=("Fourier GR1, humanoid/singleview. From robosuite (absent from "
               "menagerie). 32 body hinges; the two 6-DoF hands are separate MJCFs and "
               "excluded. Floating base occupies qpos 0..6, so joint ids and qpos "
               "addresses differ by 6 -- resolved via joint_names. Measured on all 260 "
               "episodes: 4/32 joints exceed the MJCF range by at most 2.68 deg, so "
               "0.05 rad slack on those four puts GT at 0.000%."),
    ),
}

# which robot each dataset group maps to.
#
# NOTE only single_arm/multiview carries `observation.state.joint_position`.
# single_arm/singleview stores a 7-D cartesian state (xyz + rpy + gripper) with no
# joint angles, so FK-consistency is not computable there and the checker must say so
# rather than silently score something else.
GROUP_ROBOT = {
    ("single_arm", "multiview"): "franka_panda",
    ("single_arm", "singleview"): "franka_panda",
    ("bimanual", "singleview"): "aloha",
    ("bimanual", "multiview"): "aloha",
    ("humanoid", "singleview"): "gr1",
}

# groups whose parquet actually carries joint angles
GROUP_HAS_JOINTS = {
    ("single_arm", "multiview"): True,
    ("single_arm", "singleview"): False,
    ("bimanual", "singleview"): True,
    ("bimanual", "multiview"): True,
    ("humanoid", "singleview"): True,
}


def resolve_mjcf(relpath: str) -> Path:
    for root in MJCF_SEARCH:
        p = root / relpath
        if p.exists():
            return p
    raise FileNotFoundError(
        f"MJCF {relpath!r} not found under {[str(r) for r in MJCF_SEARCH]}. "
        "Fetch it with:\n"
        "  git clone --depth 1 --filter=blob:none --sparse "
        "https://github.com/google-deepmind/mujoco_menagerie.git /tmp/mj_men\n"
        "  cd /tmp/mj_men && git sparse-checkout set franka_emika_panda"
    )


class Kinematics:
    """FK and joint-limit queries for one robot. Stateless from the caller's view."""

    def __init__(self, spec: RobotSpec):
        import mujoco

        self.spec = spec
        self.mujoco = mujoco
        path = resolve_mjcf(spec.mjcf_relpath)
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.ee_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, spec.ee_body
        )
        if self.ee_body_id < 0:
            raise ValueError(f"body {spec.ee_body!r} not in {path}")
        n = spec.n_arm_joints
        # Resolve dataset dims -> (joint id for ranges, qpos address for FK). These are
        # the same array only for chains of 1-DoF hinges based at qpos 0; see
        # RobotSpec.joint_names for why conflating them is a bug.
        if spec.joint_names is not None:
            jnt_ids = []
            for jn in spec.joint_names:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                if jid < 0:
                    raise ValueError(f"joint {jn!r} not in {path}")
                jnt_ids.append(jid)
            self.jnt_ids = np.asarray(jnt_ids, dtype=int)
            self.qpos_index = self.model.jnt_qposadr[self.jnt_ids].copy()
        else:
            self.qpos_index = (np.asarray(spec.joint_index_map, dtype=int)
                               if spec.joint_index_map is not None else np.arange(n))
            self.jnt_ids = self.qpos_index
        if len(self.qpos_index) != n:
            raise ValueError(f"joint map has {len(self.qpos_index)} entries, expected {n}")
        self.state_dims = (np.asarray(spec.state_dims, dtype=int)
                           if spec.state_dims is not None else np.arange(n))
        if len(self.state_dims) != n:
            raise ValueError(f"state_dims has {len(self.state_dims)} entries, expected {n}")
        slack = np.zeros(n)
        if spec.limit_slack:
            slack[: len(spec.limit_slack)] = np.asarray(spec.limit_slack[:n])
        self.joint_low = self.model.jnt_range[self.jnt_ids, 0].copy() - slack
        self.joint_high = self.model.jnt_range[self.jnt_ids, 1].copy() + slack
        self.joint_low_mjcf = self.model.jnt_range[self.jnt_ids, 0].copy()
        self.joint_high_mjcf = self.model.jnt_range[self.jnt_ids, 1].copy()
        self.limit_slack = slack
        self.mjcf_path = str(path)

    @classmethod
    def for_group(cls, embodiment: str, view_type: str) -> "Kinematics | None":
        key = GROUP_ROBOT.get((embodiment, view_type))
        return cls(REGISTRY[key]) if key else None

    def select_state(self, state: np.ndarray) -> np.ndarray:
        """Gather the dataset dims this model consumes, in `joint_names` order.

        GR1's 44-D state interleaves two 6-DoF hands between the body groups, so the
        model's dims are not a leading contiguous slice and `joint_slice` cannot
        express them.
        """
        state = np.atleast_2d(np.asarray(state, dtype=np.float64))
        return state[:, self.state_dims]

    def fk(self, q: np.ndarray) -> np.ndarray:
        """Joint angles (T,n) -> end-effector positions (T,3) in the robot base frame."""
        q = np.atleast_2d(np.asarray(q, dtype=np.float64))
        n = self.spec.n_arm_joints
        out = np.empty((len(q), 3))
        for i, qi in enumerate(q):
            self.data.qpos[self.qpos_index] = qi[:n]
            self.mujoco.mj_kinematics(self.model, self.data)
            out[i] = self.data.xpos[self.ee_body_id]
        return out

    def fk_frames(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Same as :meth:`fk` but also returns 3x3 rotations (T,3,3)."""
        q = np.atleast_2d(np.asarray(q, dtype=np.float64))
        n = self.spec.n_arm_joints
        pos = np.empty((len(q), 3))
        rot = np.empty((len(q), 3, 3))
        for i, qi in enumerate(q):
            self.data.qpos[self.qpos_index] = qi[:n]
            self.mujoco.mj_kinematics(self.model, self.data)
            pos[i] = self.data.xpos[self.ee_body_id]
            rot[i] = self.data.xmat[self.ee_body_id].reshape(3, 3)
        return pos, rot

    def limit_violation(self, q: np.ndarray) -> dict:
        """How far outside the mechanical joint range a trajectory goes.

        ``frac_frames`` is the fraction of frames with any joint out of range, and
        ``max_excess_rad`` the largest overshoot. Ranges are the MJCF's plus
        ``spec.limit_slack``; with that slack GT measures 0.000% on DROID, so a
        non-zero value is evidence about the prediction, not about the robot.
        """
        q = np.atleast_2d(np.asarray(q, dtype=np.float64))[:, : self.spec.n_arm_joints]
        below = np.maximum(self.joint_low - q, 0.0)
        above = np.maximum(q - self.joint_high, 0.0)
        excess = below + above
        bad = excess > 0
        if self.spec.limit_check_dims is not None:
            mask = np.zeros(bad.shape[1], dtype=bool)
            mask[list(self.spec.limit_check_dims)] = True
            bad = bad & mask
            excess = excess * mask
        return {
            "frac_frames": float(bad.any(1).mean()),
            "frac_joint_entries": float(bad.mean()),
            "max_excess_rad": float(excess.max()),
            "mean_excess_rad": float(excess[bad].mean()) if bad.any() else 0.0,
            "per_joint_frac": [float(v) for v in bad.mean(0)],
            "checked_dims": (list(self.spec.limit_check_dims)
                             if self.spec.limit_check_dims is not None else "all"),
        }

    def fk_residual(self, q: np.ndarray, ee_xyz: np.ndarray) -> dict:
        """Distance between FK of ``q`` and an independently measured EE position.

        On GT this is a calibration check (0.0 on DROID). On a *recovered* trajectory
        it is the metric: it asks whether the joint configuration implied by the video
        actually puts the end effector where the video shows it.
        """
        fk = self.fk(q)
        ee = np.asarray(ee_xyz, dtype=np.float64)[:, :3]
        n = min(len(fk), len(ee))
        err = np.linalg.norm(fk[:n] - ee[:n], axis=1)
        return {
            "mean_m": float(err.mean()),
            "median_m": float(np.median(err)),
            "max_m": float(err.max()),
            "p95_m": float(np.percentile(err, 95)),
            "n_frames": int(n),
        }


def validate_group(embodiment: str, view_type: str, split: str, max_episodes: int = 25) -> dict:
    """Check the model against dataset GT, by whichever evidence the dataset supports.

    Two levels, and the difference matters when reading the result:

    ``fk`` -- the dataset carries an independent end-effector column, so FK can be
    cross-validated directly. Strong evidence: Panda reproduces DROID's own
    ``cartesian_position`` to 3.3e-08 m, which pins joint order, base frame and units
    simultaneously.

    ``limits`` -- no EE column (GR1, ALOHA), so the available check is whether real GT
    stays inside the MJCF's joint ranges. Weaker: it can refute a model (a wrong joint
    order shows up as ~1 rad excursions) but cannot confirm the base frame or the joint
    ORDER within a group. Reported as ``evidence: limits`` so a reader does not mistake
    it for the FK-validated case.
    """
    import glob

    import pandas as pd

    kin = Kinematics.for_group(embodiment, view_type)
    if kin is None:
        return {"error": f"no robot registered for {embodiment}/{view_type}"}
    if not GROUP_HAS_JOINTS.get((embodiment, view_type), False):
        return {"skipped": "group has no joint_position column; FK-consistency N/A",
                "robot": kin.spec.name}
    spec = kin.spec
    have_ee = spec.ee_source is not None

    base = _REPO_ROOT / "datasets" / embodiment / view_type / split
    files = sorted(glob.glob(str(base / "**/data/chunk-*/episode_*.parquet"), recursive=True))
    files += sorted(glob.glob(str(base / "data/chunk-*/episode_*.parquet")))
    files = sorted(set(files))
    if not files:
        return {"error": f"no parquet under {base}"}
    step = max(1, len(files) // max_episodes)
    files = files[::step][:max_episodes]

    res, viol, excess, frames, skipped = [], [], [], 0, 0
    for f in files:
        df = pd.read_parquet(f)
        if spec.joint_source not in df.columns:
            skipped += 1
            continue
        state = np.stack(df[spec.joint_source].values)
        if spec.state_dims is not None:
            if state.shape[1] <= max(spec.state_dims):
                skipped += 1
                continue
            q = kin.select_state(state)
        else:
            q = state[:, slice(*spec.joint_slice)]
        if have_ee:
            if spec.ee_source not in df.columns:
                skipped += 1
                continue
            ee = np.stack(df[spec.ee_source].values)[:, slice(*spec.ee_slice)]
            res.append(kin.fk_residual(q, ee)["max_m"])
        v = kin.limit_violation(q)
        viol.append(v["frac_frames"])
        excess.append(v.get("max_excess_rad", 0.0))
        frames += len(q)

    if not viol:
        return {"error": f"all {skipped} episodes lack {spec.joint_source}"}
    out = {
        "robot": spec.name, "mjcf": kin.mjcf_path,
        "evidence": "fk" if have_ee else "limits",
        "episodes_checked": len(viol), "episodes_skipped": skipped,
        "frames": frames,
        "gt_limit_violation_frac_mean": float(np.mean(viol)),
        "gt_limit_violation_frac_worst": float(np.max(viol)),
        "gt_limit_max_excess_rad": float(np.max(excess)) if excess else 0.0,
    }
    if have_ee:
        out["fk_residual_max_m_mean"] = float(np.mean(res))
        out["fk_residual_max_m_worst"] = float(np.max(res))
    else:
        out["note"] = ("no independent EE column: joint ORDER within a group and the base "
                       "frame are NOT verified here, only that GT respects the MJCF ranges")
    return out


def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Validate FK/limits against dataset GT.")
    ap.add_argument("--groups", nargs="*", default=["single_arm/multiview", "single_arm/singleview"])
    ap.add_argument("--split", default="makovian")
    ap.add_argument("--max-episodes", type=int, default=25)
    args = ap.parse_args()

    for g in args.groups:
        emb, vt = g.split("/")
        r = validate_group(emb, vt, args.split, args.max_episodes)
        print(f"=== {g} (split={args.split})")
        print(json.dumps(r, indent=2))
        if "error" not in r and "skipped" not in r:
            if r.get("evidence") == "fk":
                ok = r["fk_residual_max_m_worst"] < 1e-4
                print(f"    FK match: {'OK' if ok else 'MISMATCH — check joint order/base frame'}")
            else:
                ok = r["gt_limit_violation_frac_mean"] < 1e-4
                print(f"    GT within MJCF limits: {'OK' if ok else 'VIOLATIONS — see note'}"
                      f"  (evidence=limits, weaker than FK)")
    # MuJoCo's C++ objects can abort at interpreter teardown; results are already out
    os._exit(0)


if __name__ == "__main__":
    main()
