"""Run the force-feasibility tier over episodes the cheap tier has already tracked.

Reports ``residual_excess = pred - gt`` as the headline. The raw prediction residual is
also written, but the difference is what should be compared across methods: the vision
stage (monocular depth, mask boundaries, a primitive stand-in for the real object) has
biases that are shared between a prediction and its paired GT and therefore cancel in
the difference. A raw residual mixes "the video is unphysical" with "our proxy of it is
imperfect"; the excess isolates the former.

Usage
-----
    python -m metrics.force_feasibility.run_episodes \
        --root metrics/sim2real_out/dense/humanoid_singleview_dreamgen/makovian \
        --out metrics/force_feasibility_out/humanoid_singleview_dreamgen_makovian.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np

from .lift_to_3d import lift_episode
from .metric_scale import resolve_from_tracks
from .newton import newton_feasibility


def run_one(npz: str, stride: int, device: str, verbose: bool,
            use_fk_scale: bool = True) -> dict:
    rec: dict = {"tracks": npz, "episode": Path(npz).parent.name}
    gt_len = None
    if use_fk_scale:
        sc = resolve_from_tracks(npz)
        rec["fk_scale"] = sc
        gt_len = sc.get("path_len_m")
        # The GT episode is longer than the tracked window when the cheap tier matched a
        # shorter prediction, so scale by frames actually used rather than the full path.
        if gt_len and sc.get("n_frames"):
            rec["fk_scale"]["path_len_full_m"] = gt_len
    # GT is lifted first and its object size then anchors the prediction's metric scale.
    # Grip demand is proportional to scale (measured 2.52 -> 11.44 for s = 0.5 -> 4), so
    # scaling each side by its own gripper path length made the comparison meaningless:
    # GT/pred scale ratios ranged 0.20x-2.35x on the same episode and the resulting grip
    # ratio correlated with the scale ratio at r = +0.943. The same physical object has one
    # size, so both sides are pinned to it.
    ref_extent = None
    for which in ("gt", "pred"):
        L = lift_episode(npz, which=which, stride=stride, device=device, verbose=verbose,
                         gt_path_len_m=gt_len, ref_obj_extent_m=ref_extent)
        if which == "gt":
            ref_extent = 2.0 * float(max(L.obj_half))
        r = newton_feasibility(L.to_observation(), base=L.to_params())
        rec[which] = {
            **{k: v for k, v in r.items() if not isinstance(v, np.ndarray)},
            "lift": L.summary(),
        }
    rec["obj_size_ratio"] = (2.0 * max(rec["pred"]["lift"]["obj_half_m"])
                             / max(2.0 * max(rec["gt"]["lift"]["obj_half_m"]), 1e-12))
    rec["residual_excess"] = rec["pred"]["unexplained_mean"] - rec["gt"]["unexplained_mean"]
    # Headline: excess restricted to frames where a contact was found in both. The
    # all-frame version is kept for reference but is dominated by contact detection
    # (measured corr -0.9999 with the change in contact fraction).
    gh, ph = rec["gt"]["unexplained_mean_held"], rec["pred"]["unexplained_mean_held"]
    rec["held_excess"] = (float(ph - gh) if not (np.isnan(gh) or np.isnan(ph)) else None)
    rec["held_frac_min"] = min(rec["gt"]["held_frac"], rec["pred"]["held_frac"])
    rec["lift_bias_excess"] = rec["pred"]["lift_bias"] - rec["gt"]["lift_bias"]
    rec["p95_excess"] = rec["pred"]["unexplained_p95"] - rec["gt"]["unexplained_p95"]
    # Grip demand excess: the informative comparison for frames where the object is held.
    # Feasibility cannot discriminate there -- two opposing frictional contacts span R^6,
    # so every wrench is explainable (measured: rank 6, worst residual 2.3e-15 over 500
    # random wrenches) -- but the force the fingers must apply is finite and comparable.
    gg, pg = rec["gt"]["grip_demand_mean"], rec["pred"]["grip_demand_mean"]
    rec["grip_excess"] = (float(pg - gg) if gg is not None and pg is not None else None)
    rec["grip_ratio"] = (float(pg / gg) if gg not in (None, 0.0) and pg is not None else None)
    rec["grip_frames_min"] = min(rec["gt"]["n_frames_grip"], rec["pred"]["n_frames_grip"])
    rec["aliased_frac_max"] = max(rec["gt"]["orientation_aliased_frac"],
                                  rec["pred"]["orientation_aliased_frac"])
    return rec


def _f(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir containing episode_*/tracks.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-fk-scale", action="store_true",
                    help="skip robot-kinematics metric scale, use the object-size fallback")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    npzs = sorted(glob.glob(os.path.join(a.root, "episode_*", "tracks.npz")))
    if a.limit:
        npzs = npzs[: a.limit]
    print(f"{len(npzs)} episodes under {a.root}")

    recs = []
    for i, npz in enumerate(npzs, 1):
        t0 = time.time()
        try:
            rec = run_one(npz, a.stride, a.device, not a.quiet, use_fk_scale=not a.no_fk_scale)
            recs.append(rec)
            he = rec["held_excess"]
            ge = rec["grip_excess"]
            print(f"[{i}/{len(npzs)}] {rec['episode']}  "
                  f"grip: gt={_f(rec['gt']['grip_demand_mean'])} "
                  f"pred={_f(rec['pred']['grip_demand_mean'])} "
                  f"excess={'n/a' if ge is None else f'{ge:+.3f}'} "
                  f"| feas held excess={'n/a' if he is None else f'{he:+.3f}'} "
                  f"(nfg={rec['grip_frames_min']}, alias={rec['aliased_frac_max']:.2f})"
                  f"  ({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"[{i}/{len(npzs)}] {Path(npz).parent.name} FAILED: {type(e).__name__}: {e}")
            recs.append({"tracks": npz, "episode": Path(npz).parent.name, "error": str(e)})

    ok = [r for r in recs if "error" not in r]
    summary = {"n_total": len(recs), "n_ok": len(ok), "root": a.root}
    if ok:
        for key, path in (("gt_residual_mean", ("gt", "unexplained_mean")),
                          ("pred_residual_mean", ("pred", "unexplained_mean")),
                          ("gt_held_mean", ("gt", "unexplained_mean_held")),
                          ("pred_held_mean", ("pred", "unexplained_mean_held"))):
            v = [r[path[0]][path[1]] for r in ok]
            summary[key] = float(np.nanmean(v))
        ex = np.array([r["residual_excess"] for r in ok])
        summary.update(excess_mean=float(ex.mean()), excess_std=float(ex.std()),
                       frac_pred_worse=float((ex > 0).mean()))
        hv = [r for r in ok if r["held_excess"] is not None]
        if hv:
            h = np.array([r["held_excess"] for r in hv])
            summary.update(n_held=len(hv), held_excess_mean=float(h.mean()),
                           held_excess_std=float(h.std()),
                           held_frac_pred_worse=float((h > 0).mean()))
        # Degeneracy audit: if the score just restates whether contact was detected, this
        # correlation is ~-1. Reported every run so the failure cannot go unnoticed.
        dg = np.array([r["pred"]["contact_frac_gripper"] - r["gt"]["contact_frac_gripper"]
                       for r in ok])
        if len(ok) > 2 and dg.std() > 1e-6:
            summary["corr_excess_vs_dcontact"] = float(np.corrcoef(ex, dg)[0, 1])
            if hv:
                dgh = np.array([r["pred"]["contact_frac_gripper"] - r["gt"]["contact_frac_gripper"]
                                for r in hv])
                if dgh.std() > 1e-6:
                    summary["corr_heldexcess_vs_dcontact"] = float(np.corrcoef(h, dgh)[0, 1])

        # Grip demand: the headline for held frames. Reported with the same degeneracy audit,
        # because a metric that merely restates how often contact was detected is not physics.
        gv = [r for r in ok if r["grip_excess"] is not None]
        if gv:
            g = np.array([r["grip_excess"] for r in gv])
            summary.update(
                n_grip=len(gv),
                gt_grip_mean=float(np.mean([r["gt"]["grip_demand_mean"] for r in gv])),
                pred_grip_mean=float(np.mean([r["pred"]["grip_demand_mean"] for r in gv])),
                grip_excess_mean=float(g.mean()), grip_excess_std=float(g.std()),
                grip_frac_pred_worse=float((g > 0).mean()),
                aliased_frac_mean=float(np.mean([r["aliased_frac_max"] for r in ok])),
                n_with_orientation=int(sum(bool(r["pred"]["torque_checked"]) for r in ok)))
            dgg = np.array([r["pred"]["contact_frac_gripper"] - r["gt"]["contact_frac_gripper"]
                            for r in gv])
            if len(gv) > 2 and dgg.std() > 1e-6:
                summary["corr_gripexcess_vs_dcontact"] = float(np.corrcoef(g, dgg)[0, 1])
            # Scale-confound audit. Grip demand is proportional to the metric scale, so if
            # GT and pred are scaled differently the comparison measures units, not physics.
            # This caught exactly that once (r = +0.943 between grip ratio and scale ratio),
            # so it is checked on every run rather than trusted to stay fixed.
            #
            # The audited quantity is the object's *true size*, not `scale_m_per_unit`. The
            # latter is a ratio between two different unit spaces -- MoGe returns depth in
            # its own arbitrary scale per video -- so comparing it across GT and pred is
            # meaningless, and a first version of this check reported a spurious r = +0.61
            # from exactly that mistake. Object extent in metres is the physical invariant.
            sz = np.array([2.0 * max(r["pred"]["lift"]["obj_half_m"])
                           / max(2.0 * max(r["gt"]["lift"]["obj_half_m"]), 1e-12) for r in gv])
            summary["obj_size_ratio_max_dev"] = float(np.abs(sz - 1.0).max())
            grr = np.array([r["grip_ratio"] for r in gv if r["grip_ratio"] is not None])
            szr = np.array([s for s, r in zip(sz, gv) if r["grip_ratio"] is not None])
            if len(grr) > 2 and szr.std() > 1e-9 and grr.std() > 1e-9:
                summary["corr_gripratio_vs_sizeratio"] = float(np.corrcoef(grr, szr)[0, 1])

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"summary": summary, "episodes": recs}, f, indent=1)
    print("\n" + json.dumps(summary, indent=1))
    print(f"\nwrote {a.out}")
    os._exit(0)


if __name__ == "__main__":
    main()
