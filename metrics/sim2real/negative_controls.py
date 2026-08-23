"""Negative controls: prove each term fires on a violation it is supposed to catch.

Cheap physics proxies are easy to fool oneself with - a number that looks
plausible on real data may be measuring nothing. So we synthesise clips whose
violation is known by construction, by editing the *tracked masks* (no
regeneration, no GPU), and require each control to collapse the specific sub-term
it targets while leaving the others alone.

Controls (all operate on the cached ``tracks.npz`` of a real episode):

``oracle``          pred := GT. Every agreement term must be perfect and every
                    violation term must equal GT's own value. This is the test that
                    caught the occlusion bug: before the ternary contact state, GT
                    itself scored a 48% levitation rate.
``freeze_object``   object mask held at frame 0 while the gripper acts. The object
                    should have been carried -> ``obj_travel_pred`` ~ 0,
                    ``trajectory`` and the lift/transport nodes must fail.
``teleport_object`` object jumps to its final position at mid-clip with no gripper
                    nearby -> ``levitation`` must fire.
``no_gripper``      gripper mask emptied -> contact can never be established;
                    ``contact_onset_missing`` = pred and the agreement terms drop.
``swap_gripper``    the gripper track of ANOTHER episode is pasted in, so the arm
                    moves plausibly but is uncorrelated with this object ->
                    ``grasp_follow`` and contact timing must degrade.
``shuffle``         frame order permuted -> temporal terms (chatter, contact
                    timing, gap curve) must degrade while per-frame appearance
                    terms stay put.
``dissolve``        object mask eroded progressively to nothing -> the
                    ``object_integrity`` family must fire and nothing else.
"""

import copy

import numpy as np
import cv2


def _empty_like(masks):
    return [np.zeros_like(m) for m in masks]


def _shift_mask(mask, dy, dx):
    out = np.zeros_like(mask)
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return out
    ny, nx = ys + int(round(dy)), xs + int(round(dx))
    ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
    out[ny[ok], nx[ok]] = True
    return out


def _centroid(mask):
    ys, xs = np.nonzero(mask)
    return (float(ys.mean()), float(xs.mean())) if ys.size else None


def ctrl_oracle(tracks):
    t = copy.deepcopy(tracks)
    for vw in t["views"]:
        vw["pred"] = {r: [m.copy() for m in vw["gt"][r]] for r in vw["gt"]}
        vw["frames_dir"]["pred"] = vw["frames_dir"]["gt"]
    return t


def ctrl_freeze_object(tracks):
    t = copy.deepcopy(tracks)
    for vw in t["views"]:
        first = vw["gt"]["object"][0].copy()
        vw["pred"]["object"] = [first.copy() for _ in vw["gt"]["object"]]
        vw["pred"]["gripper"] = [m.copy() for m in vw["gt"]["gripper"]]
        vw["pred"]["target"] = [m.copy() for m in vw["gt"]["target"]]
    return t


def ctrl_teleport_object(tracks):
    """Object sits still, then jumps to its GT final position with no arm nearby."""
    t = copy.deepcopy(tracks)
    for vw in t["views"]:
        obj = vw["gt"]["object"]
        T = len(obj)
        half = T // 2
        c0, c1 = _centroid(obj[0]), None
        for m in reversed(obj):
            c1 = _centroid(m)
            if c1 is not None:
                break
        seq = []
        for i in range(T):
            if c0 is None or c1 is None:
                seq.append(obj[0].copy())
            elif i < half:
                seq.append(obj[0].copy())
            else:
                # after the jump keep drifting so `moving` frames exist to score
                f = (i - half) / max(1, T - 1 - half)
                seq.append(_shift_mask(obj[0], (c1[0] - c0[0]) * f + (c1[0] - c0[0]),
                                       (c1[1] - c0[1]) * f + (c1[1] - c0[1])))
        vw["pred"]["object"] = seq
        # arm parked in its frame-0 (arm-free) state so contact is definitely absent
        vw["pred"]["gripper"] = _empty_like(vw["gt"]["gripper"])
        vw["pred"]["target"] = [m.copy() for m in vw["gt"]["target"]]
    return t


def ctrl_no_gripper(tracks):
    t = copy.deepcopy(tracks)
    for vw in t["views"]:
        vw["pred"] = {r: [m.copy() for m in vw["gt"][r]] for r in vw["gt"]}
        vw["pred"]["gripper"] = _empty_like(vw["gt"]["gripper"])
    return t


def ctrl_swap_gripper(tracks, donor):
    t = copy.deepcopy(tracks)
    for vw, dv in zip(t["views"], donor["views"]):
        vw["pred"] = {r: [m.copy() for m in vw["gt"][r]] for r in vw["gt"]}
        src = dv["gt"]["gripper"]
        T = len(vw["gt"]["gripper"])
        H, W = vw["gt"]["gripper"][0].shape
        seq = []
        for i in range(T):
            m = src[i % len(src)]
            if m.shape != (H, W):
                m = cv2.resize(m.astype(np.uint8), (W, H),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            seq.append(m)
        vw["pred"]["gripper"] = seq
    return t


def ctrl_shuffle(tracks, seed=0):
    t = copy.deepcopy(tracks)
    rng = np.random.default_rng(seed)
    T = t["num_frames"]
    perm = rng.permutation(T)
    for vw in t["views"]:
        vw["pred"] = {r: [vw["gt"][r][i].copy() for i in perm] for r in vw["gt"]}
    return t


def ctrl_dissolve(tracks):
    """Object silhouette eroded to nothing - the 'melting object' artifact."""
    t = copy.deepcopy(tracks)
    for vw in t["views"]:
        obj = vw["gt"]["object"]
        T = len(obj)
        seq = []
        for i, m in enumerate(obj):
            k = int(1 + 8 * i / max(1, T - 1))
            if k <= 1 or not m.any():
                seq.append(m.copy())
                continue
            ker = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
            seq.append(cv2.erode(m.astype(np.uint8), ker).astype(bool))
        vw["pred"]["object"] = seq
        vw["pred"]["gripper"] = [g.copy() for g in vw["gt"]["gripper"]]
        vw["pred"]["target"] = [g.copy() for g in vw["gt"]["target"]]
    return t


CONTROLS = {
    "oracle": ctrl_oracle,
    "freeze_object": ctrl_freeze_object,
    "teleport_object": ctrl_teleport_object,
    "no_gripper": ctrl_no_gripper,
    "shuffle": ctrl_shuffle,
    "dissolve": ctrl_dissolve,
}

# which sub-term MUST react to each control (used for the pass/fail verdict)
EXPECT = {
    "oracle": {"score": ("high", 0.98)},
    "freeze_object": {"trajectory": ("low", 0.5), "step_pass_rate": ("low", 0.6)},
    "teleport_object": {"levitation": ("low", 0.5)},
    "no_gripper": {"contact_overlap": ("low", 0.2)},
    "swap_gripper": {"grasp_follow": ("low", 0.6)},
    "shuffle": {"gap_curve": ("low", 0.8), "trajectory": ("low", 0.8)},
    "dissolve": {"object_shape": ("low", 0.6)},
}


def verdict(name, metrics):
    """Did the control collapse the term it was designed to collapse?"""
    got, ok = {}, True
    for term, (direction, thr) in EXPECT.get(name, {}).items():
        if term == "score":
            v = metrics["sim2real_interaction_score"]
            if v is None:
                v = float("nan")
        elif term == "step_pass_rate":
            v = metrics["step_level"]["step_pass_rate"]
        else:
            v = metrics["score_terms"].get(term, {}).get("norm")
        got[term] = v
        if v is None or not np.isfinite(v):
            ok = False
        elif direction == "low" and v > thr:
            ok = False
        elif direction == "high" and v < thr:
            ok = False
    return ok, got
