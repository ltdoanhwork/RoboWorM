"""Which arm-separation strategy actually fixes the ruler? Measured, not argued.

The gripper mask unions both arms with background, which broke every metric-scale estimator
tried so far (see ``metrics/real2sim/README.md``). Three fixes are available and they differ in
cost and in what they assume:

1. **re-prompt** GroundingDINO for left/right arms separately, or just fix the existing
   ``pick_acting_arms`` threshold -- requires re-running the tracker over every episode;
2. **connected components** on the masks already stored in ``tracks.npz``, keeping the
   component nearest the object -- costs nothing and needs no re-run;
3. **SAM prompted with FK-projected hand points** -- most precise in principle, but it needs
   the ego camera's intrinsics *and* its pose in the robot's frame to project with, and
   neither is recorded in this dataset, so it cannot be evaluated here at all.

The decisive test is not "does the mask look better". It is whether the metric scale becomes
**reproducible**: ``episode_000829`` and ``episode_003151`` manipulate the same green bok choy,
so any correct ruler must return the same object size for both. That test is what this reports.

Run:
    models/dreamgen/.venv/bin/python -m metrics.real2sim.probe_arm_split
"""

from __future__ import annotations

import json
import os

import numpy as np


BOK_CHOY = ("episode_000829", "episode_003151")
ROOT = "metrics/sim2real_out/dense/humanoid_singleview_dreamgen/makovian"


def load_masks(ep: str, which: str = "gt", view: int = 0):
    import cv2

    from metrics.force_feasibility.lift_to_3d import unpack_masks

    tr = f"{ROOT}/{ep}/tracks.npz"
    z = np.load(tr, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    T, hw = int(meta["num_frames"]), tuple(meta["mask_hw"])
    mg = unpack_masks(z, f"v{view}_{which}_gripper", hw, T)
    mo = unpack_masks(z, f"v{view}_{which}_object", hw, T)

    frames = []
    cap = cv2.VideoCapture(meta["gt_video"] if which == "gt" else meta["pred_video"])
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return mg, mo, frames, min(T, len(frames)), meta


def split_components(mask: np.ndarray, obj: np.ndarray, min_frac: float = 0.004):
    """Connected components of an arm mask, ranked by proximity to the object.

    Returns ``(components, chosen_index)``. Proximity is measured between component and object
    *pixels*, not centroids: the arm is long and curved, so its centroid can be far from the
    object while its fingers touch it -- which is the whole reason the centroid was the wrong
    trajectory in the first place.
    """
    import cv2

    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    comps = []
    total = mask.size
    for k in range(1, n):
        c = lab == k
        if c.sum() / total < min_frac:
            continue
        comps.append(c)
    if not comps:
        return [], -1
    if not obj.any():
        # No object to be near: fall back to the largest component and let the caller see
        # (via the returned list) that the choice was unconstrained.
        return comps, int(np.argmax([c.sum() for c in comps]))
    dist_to_obj = cv2.distanceTransform((~obj).astype(np.uint8), cv2.DIST_L2, 3)
    d = [float(np.percentile(dist_to_obj[c], 1.0)) for c in comps]
    return comps, int(np.argmin(d))


def measure(ep: str, mode: str, stride: int = 6, device: str = "cuda") -> dict:
    """3D extents of the arm (whole vs nearest-component) and the object, in MoGe units."""
    import cv2
    import torch
    from moge.model.v1 import MoGeModel

    mg, mo, frames, T, meta = load_masks(ep)
    model = measure._model  # noqa: SLF001
    _ = model

    res = {k: [] for k in ("armXY", "armZ", "arm3", "objXY", "obj3", "armArea", "nComp")}
    for t in range(0, T, stride):
        rgb = frames[t]
        H, W = rgb.shape[:2]
        tt = torch.tensor(rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
        with torch.no_grad():
            out = model.infer(tt)
        pts = out["points"].float().cpu().numpy()
        val = out["mask"].cpu().numpy().astype(bool)

        g = cv2.resize(mg[t].astype(np.uint8), (W, H),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
        o = cv2.resize(mo[t].astype(np.uint8), (W, H),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
        comps, pick = split_components(g, o)
        res["nComp"].append(len(comps))
        if mode == "component" and pick >= 0:
            g = comps[pick]
        g = g & val
        o = o & val

        def spread(Q, cols):
            if len(Q) < 30:
                return np.nan
            C = Q[:, cols] - Q[:, cols].mean(0)
            if C.shape[1] == 1:
                s = C[:, 0]
            else:
                s = C @ np.linalg.svd(C, full_matrices=False)[2][0]
            return float(np.percentile(s, 98) - np.percentile(s, 2))

        res["arm3"].append(spread(pts[g], [0, 1, 2]))
        res["armXY"].append(spread(pts[g], [0, 1]))
        res["armZ"].append(spread(pts[g], [2]))
        res["obj3"].append(spread(pts[o], [0, 1, 2]))
        res["objXY"].append(spread(pts[o], [0, 1]))
        res["armArea"].append(float(g.mean()))
    return {k: float(np.nanmedian(v)) if v else np.nan for k, v in res.items()}


def fk_arm_extent_m(ep: str) -> float:
    """Median FK extent in metres of the manipulating arm, for the same episode."""
    from metrics.force_feasibility.metric_scale import resolve_from_tracks
    from metrics.real2sim.probe_static_scale import fk_arm_extent

    md = resolve_from_tracks(f"{ROOT}/{ep}/tracks.npz")
    f = fk_arm_extent(md["embodiment"], md["view_type"], md["split"], md["episode"],
                      list(range(0, 60)))
    # Median over candidate subsets: they agree to within ~15% (measured in
    # `probe_static_scale`), so the unknown proximal cut-off is a small error term and picking
    # among them is not worth a weak correlation.
    return float(np.median([np.median(v) for v in f["subset_extent_m"].values()]))


def main() -> None:
    from moge.model.v1 import MoGeModel

    measure._model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to("cuda").eval()  # noqa: SLF001

    print("Test: two episodes manipulate the SAME green bok choy, so a correct ruler must give")
    print("the same object size for both. Anything else is a broken ruler.\n")
    rows = {}
    for mode in ("whole", "component"):
        print(f"--- gripper mask = {mode} ---")
        print("  ep              armArea%  nComp  armXY   armZ  Z/XY   objXY   "
              "armFK_m  ->  obj_cm")
        sizes = []
        for ep in BOK_CHOY:
            m = measure(ep, mode)
            fk = fk_arm_extent_m(ep)
            scale = fk / max(m["arm3"], 1e-9)
            obj_cm = m["obj3"] * scale * 100.0
            sizes.append(obj_cm)
            print(f"  {ep}  {m['armArea'] * 100:7.2f}  {m['nComp']:5.1f}  "
                  f"{m['armXY']:6.3f} {m['armZ']:6.3f} {m['armZ'] / max(m['armXY'], 1e-9):5.2f}  "
                  f"{m['objXY']:6.3f}  {fk:7.3f}  ->  {obj_cm:6.1f}")
        ratio = max(sizes) / max(min(sizes), 1e-9)
        rows[mode] = ratio
        print(f"  => disagreement between the two bok choy episodes: "
              f"{ratio:.2f}x  {'PASS' if ratio < 1.5 else 'FAIL'}\n")
    print("summary:", {k: f"{v:.2f}x" for k, v in rows.items()})
    print("\nStrategy 3 (SAM prompted with FK-projected hand points) is not evaluated: it needs")
    print("the ego camera's intrinsics and its pose in the robot frame, neither of which this")
    print("dataset records. Claiming a number for it would be inventing one.")


if __name__ == "__main__":
    main()
