"""Screen episodes for a valid grasp signal BEFORE measuring anything from them.

Two episodes were spent discovering, one at a time, that they could not support the measurement:
`000829` is an instruction/video mismatch (the hand grasps something, but never approaches the
object named and correctly detected), and `003151` has object and arm masks overlapping from
frame 0, so its contact fraction of 1.00 says nothing about when a grasp begins. Both failures
are detectable cheaply and *before* any 3D lifting, metric-scale fitting, or reconstruction runs.

The two gates are independent by construction -- one reads GT joint angles and never looks at the
video, the other reads only vision and never looks at the robot -- so agreeing is evidence and
disagreeing is a diagnosis:

**Gate A (kinematic, GT only).** The manipulating hand's finger joint sweeps a large range when it
closes (measured: ~1.4 rad on all three episodes examined). That locates a grasp window in
parquet-row terms and confirms the window lies inside the video segment.

**Gate B (vision only).** Inside a real grasp the object-to-arm gap must *approach zero*, and
before it the gap must be *non-zero*. A gap that is already zero at frame 0 is mask adjacency, not
contact -- it cannot distinguish holding from merely being next to.

Neither gate is a threshold invented here: A's is the midpoint of the joint's own observed range,
B's is the same `tau_gap` the metric already uses.

Run:
    models/dreamgen/.venv/bin/python -m metrics.real2sim.screen_episodes
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Screen:
    episode: str
    instruction: str = ""
    # Gate A
    grasp_start_frac: float | None = None
    grasp_frames: int = 0
    hand: str = ""
    joint_range: float = 0.0
    inside_video: bool = False
    # Gate B
    gap_before: float | None = None
    gap_during: float | None = None
    gap_at_frame0: float | None = None
    contact_before: float | None = None
    contact_during: float | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def gate_a(self) -> bool:
        return (self.grasp_start_frac is not None and self.grasp_frames >= 5
                and self.joint_range > 0.5 and self.inside_video)

    @property
    def gate_b(self) -> bool:
        if self.gap_during is None or self.gap_at_frame0 is None:
            return False
        # A grasp must (1) bring the gap into contact during the kinematic grasp window, and
        # (2) not already be in contact at frame 0 -- otherwise the two masks merely touch and
        # the signal cannot separate holding from adjacency.
        return (self.contact_during or 0.0) > 0.3 and self.gap_at_frame0 > 0.05

    @property
    def ok(self) -> bool:
        return self.gate_a and self.gate_b


def gate_a(episode: str, num_video_frames: int, embodiment: str = "humanoid",
           view: str = "singleview", split: str = "makovian",
           datasets_root: str = "datasets") -> dict:
    """Locate the grasp window from GT finger joints. Reads no video."""
    import pandas as pd

    eid = int(str(episode).replace("episode_", ""))
    hits = sorted(glob.glob(
        f"{datasets_root}/{embodiment}/{view}/{split}/data/chunk-*/episode_{eid:06d}.parquet"))
    if not hits:
        return {"error": "no parquet"}
    S = np.stack(pd.read_parquet(hits[0])["observation.state"].values)
    T = len(S)
    # GR1's 44-D state interleaves a 6-DoF hand after each arm group; both are checked and the
    # one with the larger sweep is taken as the manipulating hand, so no per-episode annotation
    # of left/right is needed.
    cands = {"left": (7, 13), "right": (29, 35)}
    best, best_rng = None, -1.0
    for name, (lo, hi) in cands.items():
        if S.shape[1] <= hi:
            continue
        H = S[:, lo:hi]
        rng = float((H.max(0) - H.min(0)).max())
        if rng > best_rng:
            best, best_rng = (name, H), rng
    if best is None:
        return {"error": "no hand dims in state"}
    name, H = best
    j = int(np.argmax(H.max(0) - H.min(0)))
    q = H[:, j]
    lo, hi = float(q.min()), float(q.max())
    mid = (lo + hi) / 2.0
    # "Closed" is whichever side of the midpoint the trajectory does NOT start on, so this works
    # whether the joint value increases or decreases on closing.
    closed = q > mid if abs(q[0] - lo) < abs(q[0] - hi) else q < mid
    idx = np.where(closed)[0]
    if not len(idx):
        return {"error": "no closing detected", "hand": name, "joint_range": best_rng}
    # Video-frame fraction, since the video is resampled relative to parquet rows.
    start_frac = float(idx.min() / max(T - 1, 1))
    return {"hand": name, "joint": j, "joint_range": best_rng,
            "grasp_start_frac": start_frac, "grasp_frames": int(len(idx)),
            "inside_video": bool(start_frac < 1.0 and num_video_frames > 0),
            "n_rows": T}


def gate_b(tracks_npz: str, grasp_start_frac: float | None, view: int = 0,
           tau_gap: float = 0.35) -> dict:
    """Object-to-arm gap before versus during the grasp window. Reads no robot state."""
    import cv2

    from metrics.force_feasibility.lift_to_3d import unpack_masks

    z = np.load(tracks_npz, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    hw = tuple(meta["mask_hw"])
    T = int(meta["num_frames"])
    mo = unpack_masks(z, f"v{view}_gt_object", hw, T)
    mg = unpack_masks(z, f"v{view}_gt_gripper", hw, T)

    gaps = np.full(T, np.nan)
    for t in range(T):
        o, g = mo[t], mg[t]
        if o.sum() < 10 or g.sum() < 10:
            continue
        dt = cv2.distanceTransform((~g).astype(np.uint8), cv2.DIST_L2, 3)
        # Object-diameter units, the same normalisation the metric uses; verified
        # scale-invariant (0.416 at 2304x1296 vs 0.425 at 768x432).
        gaps[t] = float(dt[o].min()) / max(1.0, float(np.sqrt(o.sum())))

    out = {"num_frames": T, "n_measured": int(np.isfinite(gaps).sum()),
           "gap_at_frame0": float(gaps[0]) if np.isfinite(gaps[0]) else None}
    if grasp_start_frac is None:
        return out
    cut = int(grasp_start_frac * T)
    for lo, hi, tag in ((0, cut, "before"), (cut, T, "during")):
        seg = gaps[lo:hi]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            out[f"gap_{tag}"] = float(np.median(seg))
            out[f"contact_{tag}"] = float(np.mean(seg < tau_gap))
    return out


def screen(ep_dir: str) -> Screen:
    ep = os.path.basename(ep_dir.rstrip("/"))
    tr = f"{ep_dir}/tracks.npz"
    z = np.load(tr, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    s = Screen(episode=ep)

    pair = (f"metrics/sim2real_out/_pairs/humanoid_singleview/makovian/{ep}/metadata.json")
    if os.path.exists(pair):
        s.instruction = str(json.load(open(pair)).get("instruction", ""))

    a = gate_a(ep, int(meta["num_frames"]))
    if "error" in a:
        s.reasons.append(f"gateA: {a['error']}")
    else:
        s.hand = a["hand"]
        s.joint_range = a["joint_range"]
        s.grasp_start_frac = a["grasp_start_frac"]
        s.grasp_frames = a["grasp_frames"]
        s.inside_video = a["inside_video"]

    b = gate_b(tr, s.grasp_start_frac)
    s.gap_at_frame0 = b.get("gap_at_frame0")
    s.gap_before = b.get("gap_before")
    s.gap_during = b.get("gap_during")
    s.contact_before = b.get("contact_before")
    s.contact_during = b.get("contact_during")

    if not s.gate_a:
        s.reasons.append("no clear grasp in GT joints")
    if s.gap_at_frame0 is not None and s.gap_at_frame0 <= 0.05:
        s.reasons.append(f"masks already touching at frame 0 (gap {s.gap_at_frame0:.3f}) "
                         "-> contact signal degenerate")
    if (s.contact_during or 0.0) <= 0.3:
        s.reasons.append(f"gap does not close during grasp window "
                         f"(contact_during={s.contact_during})")
    return s


def main() -> None:
    root = os.environ.get(
        "ROOT", "metrics/sim2real_out/dense/humanoid_singleview_dreamgen/makovian")
    eps = sorted(d for d in glob.glob(f"{root}/episode_*") if os.path.exists(f"{d}/tracks.npz"))
    print(f"screening {len(eps)} episodes in {root}\n")
    print(f"{'episode':16s} {'hand':6s} {'jrng':>5s} {'grasp@':>7s} {'gap_f0':>7s} "
          f"{'gap_bef':>8s} {'gap_dur':>8s} {'c_bef':>6s} {'c_dur':>6s}  A  B  verdict")
    rows = []
    for d in eps:
        s = screen(d)
        rows.append(s)

        def f(x, w=7, p=3):
            return f"{x:{w}.{p}f}" if isinstance(x, float) else " " * (w - 1) + "-"

        print(f"{s.episode:16s} {s.hand:6s} {s.joint_range:5.2f} "
              f"{(s.grasp_start_frac or 0) * 100:6.0f}% {f(s.gap_at_frame0)} "
              f"{f(s.gap_before, 8)} {f(s.gap_during, 8)} "
              f"{f(s.contact_before, 6, 2)} {f(s.contact_during, 6, 2)}  "
              f"{'Y' if s.gate_a else 'n'}  {'Y' if s.gate_b else 'n'}  "
              f"{'USABLE' if s.ok else 'reject'}")
    print()
    for s in rows:
        if not s.ok:
            print(f"  {s.episode}: {'; '.join(s.reasons)}")
    good = [s.episode for s in rows if s.ok]
    print(f"\n{len(good)}/{len(rows)} usable: {good}")

    out = os.environ.get("OUT")
    if out:
        json.dump([vars(s) | {"ok": s.ok, "gate_a": s.gate_a, "gate_b": s.gate_b}
                   for s in rows], open(out, "w"), indent=2)
        print(f"-> {out}")


if __name__ == "__main__":
    main()
