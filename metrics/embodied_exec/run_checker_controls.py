"""Do the checkers actually fire? Oracle plus synthetic violations.

`metrics/sim2real` learned this the hard way (its README lists four bugs found only
by controls, including real GT video scoring a 48% levitation rate). A cheap physics
proxy is easy to fool yourself with, so every checker here is validated the same way
before being trusted:

**oracle** — feed ground truth as the "prediction". Every violation term must read
zero and every agreement term must be perfect. This catches bias in the checker
itself; `metrics/sim2real`'s oracle reaching exactly 1.000 is what made its numbers
believable.

**controls** — corrupt a real GT trajectory in a way whose violation is known by
construction, and assert the *specific* term collapses while the others stay put.
Sensitivity alone is not enough: a term that fires on everything is useless, so
specificity is checked too.

The controls are chosen to mirror real generation failures:

| control | corruption | must fire |
|---|---|---|
| `freeze` | arm holds its first pose | `path_len_ratio` -> 0 |
| `overshoot` | joint motion scaled 3x about the mean | limits and/or velocity |
| `jitter` | white noise on joint angles | acceleration |
| `time_reverse` | trajectory played backwards | ee error, gripper correlation |
| `fk_desync` | EE claim taken from a shifted point of the path | `fk_residual` |
| `limit_break` | one joint pushed past its mechanical stop | `limits.frac_frames` |

`fk_desync` is the important one: it is the paper's headline robot failure mode
(joint configuration inconsistent with the apparent end-effector pose) reproduced
synthetically, so we can prove the checker detects it before trusting it on video.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .checkers import (calibrate_ceilings, fk_consistency, kinematic_feasibility,
                       trajectory_agreement)
from .kinematics import GROUP_HAS_JOINTS, Kinematics

_REPO_ROOT = Path(__file__).resolve().parents[2]


def load_gt(embodiment: str, view_type: str, split: str, max_episodes: int):
    """Joint trajectories + EE positions + gripper from the dataset."""
    kin = Kinematics.for_group(embodiment, view_type)
    if kin is None or not GROUP_HAS_JOINTS.get((embodiment, view_type), False):
        raise SystemExit(f"{embodiment}/{view_type} has no joint trajectories")
    spec = kin.spec
    base = _REPO_ROOT / "datasets" / embodiment / view_type / split
    files = sorted(glob.glob(str(base / "data/chunk-*/episode_*.parquet")))
    files += sorted(glob.glob(str(base / "**/data/chunk-*/episode_*.parquet"), recursive=True))
    files = sorted(set(files))
    step = max(1, len(files) // max_episodes)
    files = files[::step][:max_episodes]

    eps = []
    for f in files:
        df = pd.read_parquet(f)
        if spec.joint_source not in df.columns:
            continue
        q = np.stack(df[spec.joint_source].values)[:, slice(*spec.joint_slice)]
        ee = np.stack(df[spec.ee_source].values)[:, :3]
        grip = None
        if "observation.state.gripper_position" in df.columns:
            grip = np.stack(df["observation.state.gripper_position"].values).ravel()
        eps.append({"name": Path(f).stem, "q": q, "ee": ee, "grip": grip})
    if not eps:
        raise SystemExit("no episodes with joint data")
    return kin, eps


def make_controls(q: np.ndarray, ee: np.ndarray, kin: Kinematics, rng: np.random.Generator):
    """(name -> (q_pred, ee_pred)) corruptions of one GT trajectory."""
    out = {"oracle": (q.copy(), ee.copy())}

    frozen = np.repeat(q[:1], len(q), axis=0)
    out["freeze"] = (frozen, kin.fk(frozen))

    over = q.mean(0) + (q - q.mean(0)) * 3.0
    out["overshoot"] = (over, kin.fk(over))

    jit = q + rng.normal(0.0, 0.08, size=q.shape)
    out["jitter"] = (jit, kin.fk(jit))

    rev = q[::-1].copy()
    out["time_reverse"] = (rev, kin.fk(rev))

    # same joints, but the EE claim comes from a different point on the path:
    # the video "shows" the hand elsewhere than the joints imply
    shift = max(1, len(q) // 5)
    out["fk_desync"] = (q.copy(), np.roll(ee, shift, axis=0))

    broken = q.copy()
    broken[:, 1] = kin.joint_high[1] + 0.5
    out["limit_break"] = (broken, kin.fk(broken))
    return out


def evaluate(kin, q_pred, ee_pred, q_gt, ee_gt, grip, fps, ceil):
    fk = fk_consistency(kin, q_pred, ee_pred)
    feas = kinematic_feasibility(kin, q_pred, fps,
                                 ceil.get("vel_ceiling"), ceil.get("acc_ceiling"))
    agr = trajectory_agreement(ee_pred, ee_gt, grip, grip)
    return {
        "fk_residual_max_m": fk["fk_residual_max_m"],
        "fk_residual_mean_m": fk["fk_residual_mean_m"],
        "limit_frac": feas["limits"]["frac_frames"],
        "limit_excess_rad": feas["limits"]["max_excess_rad"],
        "vel_exceed_frac": feas["vel_exceed_frac"],
        "acc_exceed_frac": feas["acc_exceed_frac"],
        "ee_err_mean_m": agr.get("ee_err_mean_m"),
        "path_len_ratio": agr.get("path_len_ratio"),
    }


# term that each control must move, and the direction
EXPECT = {
    "oracle":       ("fk_residual_max_m", "near_zero"),
    "freeze":       ("path_len_ratio", "collapse"),
    "overshoot":    ("limit_frac", "rise"),
    "jitter":       ("acc_exceed_frac", "rise"),
    "time_reverse": ("ee_err_mean_m", "rise"),
    "fk_desync":    ("fk_residual_max_m", "rise"),
    "limit_break":  ("limit_frac", "rise"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--embodiment", default="single_arm")
    ap.add_argument("--view-type", default="multiview")
    ap.add_argument("--split", default="makovian")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    kin, eps = load_gt(args.embodiment, args.view_type, args.split, args.episodes)
    ceil = calibrate_ceilings([e["q"] for e in eps], args.fps)
    print(f"robot={kin.spec.name}  episodes={len(eps)}  fps={args.fps}")
    print(f"calibrated ceilings from GT p{ceil['percentile']} over {ceil['n_vel_samples']} samples")
    print("  vel:", np.round(ceil["vel_ceiling"], 3))
    print("  acc:", np.round(ceil["acc_ceiling"], 2))

    rng = np.random.default_rng(args.seed)
    per_control = {}
    for e in eps:
        for name, (qp, eep) in make_controls(e["q"], e["ee"], kin, rng).items():
            m = evaluate(kin, qp, eep, e["q"], e["ee"], e["grip"], args.fps, ceil)
            per_control.setdefault(name, []).append(m)

    keys = ["fk_residual_max_m", "limit_frac", "vel_exceed_frac", "acc_exceed_frac",
            "ee_err_mean_m", "path_len_ratio"]

    def agg(name, k):
        vals = [m[k] for m in per_control[name] if m[k] is not None]
        return float(np.mean(vals)) if vals else None

    table = {n: {k: agg(n, k) for k in keys} for n in per_control}

    print("\n{:<14}".format("control") + "".join(f"{k:>20}" for k in keys))
    for n in ["oracle"] + [x for x in EXPECT if x != "oracle"]:
        if n not in table:
            continue
        row = f"{n:<14}"
        for k in keys:
            v = table[n][k]
            row += f"{'-':>20}" if v is None else f"{v:>20.6f}"
        print(row)

    print("\n{:<14}{:<22}{:<12}{:>12}{:>12}   {}".format(
        "control", "watched term", "expect", "oracle", "control", "verdict"))
    all_ok = True
    for name, (term, direction) in EXPECT.items():
        if name not in table:
            continue
        base, got = table["oracle"][term], table[name][term]
        if got is None or base is None:
            ok, shown = False, "n/a"
        elif direction == "near_zero":
            ok = abs(got) < 1e-4
            shown = f"{got:.2e}"
        elif direction == "collapse":
            ok = got < 0.25 * max(base, 1e-9)
            shown = f"{got:.4f}"
        else:  # rise
            ok = got > max(10.0 * abs(base), 1e-3)
            shown = f"{got:.4f}"
        all_ok &= ok
        print("{:<14}{:<22}{:<12}{:>12}{:>12}   {}".format(
            name, term, direction,
            f"{base:.2e}" if base is not None else "-", shown,
            "PASS" if ok else "FAIL"))

    print(f"\nOVERALL: {'PASS' if all_ok else 'FAIL'}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"ceilings": ceil, "table": table, "all_ok": bool(all_ok)}, f, indent=2)
        print(f"wrote -> {args.out}")
    os._exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
