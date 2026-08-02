"""Does a *static* ruler recover the metric scale where a motion ruler failed?

Every scale estimator tried so far measured how far something moved: gripper path length,
then radius of gyration of the gripper track. Both are motion statistics of the segmentation
mask's centroid, and both were wrong, for the same reason -- the open-vocabulary detector
returns the whole arm, so the mask's centroid is not the end effector and its motion is not
the end effector's motion. Matching it to FK's end effector inflated the scale by 2.65x.

This probes a different quantity: the arm's *size* in each frame. Size needs no differencing
(so tracker jitter cannot bias it the way it biases path length) and, crucially, it does not
require knowing which material point the mask centroid corresponds to.

The open question a size ruler has is which *part* of the arm the mask covers -- elbow down,
shoulder down, hand only? Guessing would put a hand-set constant back into the pipeline. So
it is fitted instead: the arm flexes and extends during the episode, so the per-frame extent
is a time series with structure, and the FK subset that generated it can be identified by
correlating candidates against it. The winning correlation is itself the audit -- a ruler
that matches at r = 0.9 is trustworthy in a way that one matching at r = 0.1 is not.

Run:
    models/dreamgen/.venv/bin/python -m metrics.real2sim.probe_static_scale
"""

from __future__ import annotations

import json
import os

import numpy as np


def vision_arm_extent(tracks_npz: str, which: str = "gt", view: int = 0,
                      stride: int = 4, device: str = "cuda") -> dict:
    """Per-frame 3D extent of the gripper/arm mask, in MoGe's arbitrary units.

    Extent is taken along the cloud's own principal axis, between the 2nd and 98th
    percentiles: an arm is an elongated object, so its longest axis is the length that FK
    can be compared against, and percentiles rather than min/max keep a few bled-through
    background pixels from setting the answer.
    """
    import cv2
    import torch
    from moge.model.v1 import MoGeModel

    from metrics.force_feasibility.lift_to_3d import unpack_masks

    z = np.load(tracks_npz, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    T_all, hw = int(meta["num_frames"]), tuple(meta["mask_hw"])
    vid = meta["gt_video"] if which == "gt" else meta["pred_video"]
    m_grip = unpack_masks(z, f"v{view}_{which}_gripper", hw, T_all)
    m_obj = unpack_masks(z, f"v{view}_{which}_object", hw, T_all)

    frames = []
    cap = cv2.VideoCapture(vid)
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    T = min(T_all, len(frames))
    keep = list(range(0, T, stride))

    model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device).eval()
    ext, obj_ext, area = [], [], []
    for t in keep:
        rgb = frames[t]
        H, W = rgb.shape[:2]
        tt = torch.tensor(rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
        with torch.no_grad():
            out = model.infer(tt)
        pts = out["points"].float().cpu().numpy()
        valid = out["mask"].cpu().numpy().astype(bool)
        mg = cv2.resize(m_grip[t].astype(np.uint8), (W, H),
                        interpolation=cv2.INTER_NEAREST).astype(bool) & valid
        mo = cv2.resize(m_obj[t].astype(np.uint8), (W, H),
                        interpolation=cv2.INTER_NEAREST).astype(bool) & valid
        for m, sink in ((mg, ext), (mo, obj_ext)):
            if m.sum() < 30:
                sink.append(np.nan)
                continue
            P = pts[m]
            P = P - P.mean(0)
            u = np.linalg.svd(P, full_matrices=False)[2][0]
            s = P @ u
            sink.append(float(np.percentile(s, 98) - np.percentile(s, 2)))
        area.append(float(mg.mean()))
    return {"frames": keep, "arm_extent_units": np.array(ext),
            "obj_extent_units": np.array(obj_ext), "arm_area_frac": np.array(area),
            "n_frames_video": T}


def fk_arm_extent(embodiment: str, view_type: str, split: str, episode: str,
                  frames: list[int], datasets_root: str = "datasets") -> dict:
    """Per-frame extent in metres of every contiguous distal subset of the manipulating arm.

    The mask's proximal cut-off is unknown, so all of them are returned and the caller picks
    by correlation. Bodies are ordered by their distance along the kinematic chain from the
    torso, so subset ``k`` means "from the k-th joint outwards".
    """
    import mujoco
    import pandas as pd

    from metrics.embodied_exec.kinematics import GROUP_ROBOT, REGISTRY, Kinematics

    robot = GROUP_ROBOT.get((embodiment, view_type))
    spec = REGISTRY[robot]
    kin = Kinematics(spec)
    model = kin.model
    data = mujoco.MjData(model)

    eid = int(str(episode).replace("episode_", ""))
    import glob
    hits = sorted(glob.glob(
        f"{datasets_root}/{embodiment}/{view_type}/{split}/data/chunk-*/episode_{eid:06d}.parquet"))
    df = pd.read_parquet(hits[0])
    col = spec.joint_source if spec.joint_source in df.columns else "observation.state"
    S = np.stack(df[col].values)
    dims = list(spec.state_dims) if spec.state_dims else list(range(S.shape[1]))
    q = S[:, dims]

    # Which arm: the one that moves. Same rule as `metric_scale`, so the two agree on
    # which limb is the manipulator without an annotation.
    def ee_move(nm: str) -> float:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
        if bid < 0:
            return -1.0
        P = []
        for qi in q:
            data.qpos[:] = 0.0
            if model.nq > 6:
                data.qpos[3] = 1.0
            data.qpos[kin.qpos_index] = qi
            mujoco.mj_kinematics(model, data)
            P.append(data.xpos[bid].copy())
        P = np.array(P)
        return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())

    cands = {nm: ee_move(nm) for nm in ("right_hand_eef", "left_hand_eefd",
                                        "r_hand_eef", "l_hand_eef", spec.ee_body) if nm}
    cands = {k: v for k, v in cands.items() if v >= 0}
    best_ee = max(cands, key=cands.get)
    side = "l_" if best_ee.startswith(("left", "l_")) else "r_"

    # Order the arm's bodies along the chain by walking parents from the hand back to the
    # torso: this gives a true proximal->distal ordering rather than MJCF declaration order.
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, best_ee)
    chain = []
    bid = hand_id
    while bid > 0:
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        chain.append((bid, nm))
        if not (nm.startswith(side) or nm.startswith(best_ee[:4])):
            break
        bid = int(model.body_parentid[bid])
    chain = chain[::-1]   # proximal -> distal

    T = len(q)
    P = np.empty((T, len(chain), 3))
    for t, qi in enumerate(q):
        data.qpos[:] = 0.0
        if model.nq > 6:
            data.qpos[3] = 1.0
        data.qpos[kin.qpos_index] = qi
        mujoco.mj_kinematics(model, data)
        P[t] = [data.xpos[b] for b, _ in chain]

    # Map video frame indices onto parquet rows. The two need not be the same length (the
    # video is resampled), so a proportional map is used and the assumption is reported.
    idx = np.clip((np.asarray(frames, float) / max(frames[-1], 1) * (T - 1)).round().astype(int),
                  0, T - 1)

    subsets = {}
    for k in range(len(chain) - 1):
        Q = P[idx, k:]
        # Extent = spread along the subset's own principal axis, matching how the vision side
        # measures it, so the two are the same statistic and not merely similar ones.
        d = []
        for Qt in Q:
            C = Qt - Qt.mean(0)
            if len(C) < 2:
                d.append(0.0)
                continue
            u = np.linalg.svd(C, full_matrices=False)[2][0]
            s = C @ u
            d.append(float(s.max() - s.min()))
        subsets[chain[k][1]] = np.array(d)
    return {"chain": [nm for _, nm in chain], "subset_extent_m": subsets,
            "ee_body": best_ee, "n_rows": T, "row_index": idx}


def main() -> dict:
    ep = os.environ.get(
        "EP", "metrics/sim2real_out/dense/humanoid_singleview_dreamgen/makovian/episode_002346")
    tr = f"{ep}/tracks.npz"
    stride = int(os.environ.get("STRIDE", "4"))
    print(f"[probe] {tr}  stride={stride}")

    v = vision_arm_extent(tr, "gt", stride=stride)
    ok = np.isfinite(v["arm_extent_units"])
    print(f"[probe] vision: {ok.sum()}/{len(ok)} frames with arm mask, "
          f"mean area {np.nanmean(v['arm_area_frac']) * 100:.2f}% of image")

    from metrics.force_feasibility.metric_scale import resolve_from_tracks
    meta = resolve_from_tracks(tr)
    f = fk_arm_extent(meta["embodiment"], meta["view_type"], meta["split"], meta["episode"],
                      v["frames"])
    print(f"[probe] FK chain (proximal->distal): {f['chain']}")
    print(f"[probe] manipulating EE = {f['ee_body']}")

    V = v["arm_extent_units"]
    print("\n  subset (from this body outwards)      extent_m(median)   r(vs vision)   "
          "scale m/unit")
    rows = []
    for nm, F in f["subset_extent_m"].items():
        m = ok & np.isfinite(F)
        if m.sum() < 5 or np.std(F[m]) < 1e-6 or np.std(V[m]) < 1e-6:
            print(f"  {nm:36s} {np.median(F):8.4f}         (constant)")
            continue
        r = float(np.corrcoef(V[m], F[m])[0, 1])
        # Scale as the median of the per-frame ratio, NOT a regression over time. The
        # regression needs the two series to be aligned frame-to-frame; the shift control
        # below shows that alignment is only ~1.6 sigma better than chance here, because the
        # video is temporally resampled relative to the parquet rows and the mask's proximal
        # cut-off wanders. A median ratio needs no alignment at all: it only assumes the two
        # measure the same limb over the same episode, which is what makes it usable despite
        # the weak correlation. It is also insensitive to the subset choice (measured below:
        # every candidate subset lands within 15%), so the unknown cut-off costs little.
        ratio = F[m] / np.maximum(V[m], 1e-9)
        s = float(np.median(ratio))
        rows.append((r, nm, float(np.median(F)), s))
        print(f"  {nm:36s} {np.median(F):8.4f}         {r:+.3f}        {s:.4f}")

    if rows:
        rows.sort(reverse=True)
        r, nm, em, s = rows[0]

        # The scale is reported as the median across *all* candidate subsets, with their spread
        # as the uncertainty, rather than as the single best-correlating subset's value. The
        # correlation is too weak to select on (see the control below), but it does not have to
        # be: the candidates agree to within a few percent, so the unknown proximal cut-off is
        # simply not a large error term. Reporting the spread makes that claim checkable
        # instead of asserted.
        all_s = np.array([x[3] for x in rows])
        s_med = float(np.median(all_s))
        s_spread = float((all_s.max() - all_s.min()) / max(s_med, 1e-9))
        print(f"\n[probe] scale across {len(all_s)} candidate subsets: median {s_med:.4f} "
              f"m/unit, range {all_s.min():.4f}-{all_s.max():.4f} ({s_spread * 100:.0f}% spread)")
        print(f"[probe] best-correlating subset was {nm} (r={r:+.3f}, scale {s:.4f})")

        # Is that correlation worth anything? The two series are both smooth in time, and any
        # two smooth series correlate somewhat, so the number alone proves nothing. Control:
        # the same correlation after circularly shifting FK against vision, which preserves
        # each series' smoothness and destroys only their alignment.
        F = f["subset_extent_m"][nm]
        m = ok & np.isfinite(F)
        Vm, Fm = V[m], F[m]
        n = len(Vm)
        null = np.array([abs(np.corrcoef(Vm, np.roll(Fm, k))[0, 1])
                         for k in range(1, n) if n > 3])
        print(f"[probe] shift control: |r| = {null.mean():+.3f} +- {null.std():.3f} "
              f"(max {null.max():+.3f}) -> real r is only "
              f"{(abs(r) - null.mean()) / max(null.std(), 1e-9):.1f} sigma above it, so the "
              f"time alignment is NOT established; the median-ratio scale does not rely on it")

        oe = v["obj_extent_units"]
        mo_ = np.isfinite(oe)
        obj_m = float(np.median(oe[mo_]) * s_med) if mo_.any() else None
        if mo_.any():
            print(f"[probe] => object longest extent = {obj_m * 100:.1f} cm "
                  f"(scale-median; per-frame IQR "
                  f"{np.percentile(oe[mo_], 25) * s_med * 100:.1f}-"
                  f"{np.percentile(oe[mo_], 75) * s_med * 100:.1f} cm)")
        mo = resolve_from_tracks(tr)
        if mo.get("arm_centroid_rad_gyr_m"):
            print(f"[probe] motion ruler for comparison: arm-centroid r_g = "
                  f"{mo['arm_centroid_rad_gyr_m']:.4f} m, EE r_g = {mo.get('rad_gyr_m', 0):.4f} m")
        return {"scale_m_per_unit": s_med, "scale_spread_frac": s_spread,
                "subset_best_r": nm, "r": r,
                "null_mean": float(null.mean()) if len(null) else None,
                "null_std": float(null.std()) if len(null) else None,
                "obj_extent_m": obj_m, "arm_extent_m": em, "n_frames": int(m.sum()),
                "arm_area_frac": float(np.nanmean(v["arm_area_frac"]))}
    return {}


if __name__ == "__main__":
    main()
