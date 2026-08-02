"""Render the fitted proxy scene next to the video it was fitted from.

The numbers this tier reports are only trustworthy if the scene behind them is the scene the
video shows, and a table of residuals cannot demonstrate that. This renders the actual MuJoCo
model -- the same one `newton.py` queries for contacts -- driven by the same lifted trajectory,
with contact points and contact forces drawn, beside the source frame and a time plot of the
per-frame score.

What to look for, in order of how badly it breaks the measurement if wrong:

1. the object is on the plane, not floating above or sunk below it (a bad `support_z`
   turns every resting frame into a violation),
2. the fingers touch the object exactly when the video shows contact (contact detection is
   what the score is conditioned on),
3. object size is plausible in metres (it sets the moment arm, and the metric scale).

Lifting is the expensive step (MoGe over every frame, ~2 min per episode per side), so the
lifted trajectory is cached to npz and re-renders are instant.

Usage:
    MUJOCO_GL=egl python -m metrics.force_feasibility.render_sim \
        --tracks metrics/sim2real_out/dense/humanoid_singleview_dreamgen/makovian/episode_002346/tracks.npz \
        --out /tmp/sim_vis/ep002346
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def _load_frames(tracks_npz: str, which: str, stride: int) -> np.ndarray:
    """The RGB frames the masks were computed on, so the panel shows the true source."""
    import json

    z = np.load(tracks_npz, allow_pickle=True)
    meta = json.loads(str(z["meta"])) if "meta" in z else {}
    key = f"{which}_video" if f"{which}_video" in meta else None
    path = meta.get(key) or meta.get(f"{which}_path") or meta.get(which)
    if path and os.path.exists(path):
        import imageio.v3 as iio

        frames = np.stack(list(iio.imiter(path)))
        return frames[::stride]
    return None


def _cache_path(out: str, which: str) -> str:
    return f"{out}_{which}_lift.npz"


def lift_cached(tracks_npz: str, which: str, out: str, stride: int, device: str,
                gt_path_len_m: float | None, force: bool,
                ref_obj_extent_m: float | None = None,
                gt_rad_gyr_m: float | None = None):
    """Lift once, reuse forever. MoGe dominates the cost and is deterministic here."""
    from .lift_to_3d import LiftedEpisode

    tag = which if ref_obj_extent_m is None else f"{which}_ref"
    cp = _cache_path(out, tag)
    if os.path.exists(cp) and not force:
        z = np.load(cp, allow_pickle=True)
        return LiftedEpisode(
            obj_pos=z["obj_pos"], finger_pos=z["finger_pos"], fps=float(z["fps"]),
            support_z=float(z["support_z"]), obj_half=tuple(z["obj_half"]),
            scale_m_per_unit=float(z["scale"]), scale_source=str(z["scale_source"]),
            plane_inlier_frac=float(z["plane_inlier_frac"]),
            obj_valid_frac=float(z["obj_valid_frac"]), grip_valid_frac=float(z["grip_valid_frac"]),
            obj_quat=(z["obj_quat"] if "obj_quat" in z and z["obj_quat"].ndim == 2 else None),
            grip_body_pos=(z["grip_body_pos"] if "grip_body_pos" in z
                           and z["grip_body_pos"].ndim == 2 else None),
            notes=str(z["notes"]))

    from .lift_to_3d import lift_episode
    L = lift_episode(tracks_npz, which=which, stride=stride, device=device, verbose=True,
                     gt_path_len_m=gt_path_len_m, gt_rad_gyr_m=gt_rad_gyr_m,
                     ref_obj_extent_m=ref_obj_extent_m)
    os.makedirs(os.path.dirname(cp) or ".", exist_ok=True)
    np.savez_compressed(
        cp, obj_pos=L.obj_pos, finger_pos=L.finger_pos, fps=L.fps, support_z=L.support_z,
        obj_half=np.array(L.obj_half), scale=L.scale_m_per_unit, scale_source=L.scale_source,
        plane_inlier_frac=L.plane_inlier_frac, obj_valid_frac=L.obj_valid_frac,
        grip_valid_frac=L.grip_valid_frac,
        obj_quat=(L.obj_quat if L.obj_quat is not None else np.zeros(0)),
        grip_body_pos=(L.grip_body_pos if L.grip_body_pos is not None else np.zeros(0)),
        notes=L.notes)
    return L


def render_episode(L, params, width: int = 480, height: int = 360,
                   show_contacts: bool = True) -> np.ndarray:
    """One RGB frame per observed frame, from the same model `newton.py` queries."""
    import mujoco

    from .proxy_scene import build_proxy_xml

    model = mujoco.MjModel.from_xml_string(build_proxy_xml(params))
    data = mujoco.MjData(model)
    mocap = [model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)]
             for n in ("finger0", "finger1")]

    opt = mujoco.MjvOption()
    if show_contacts:
        # The two flags that make the picture diagnostic rather than decorative: where
        # MuJoCo thinks contact is, and how the solver's forces are distributed.
        opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        # Default force/contact glyphs are sized for human-scale robots. On a 2 cm object
        # they render as beams that leave the frame and hide the geometry, which is the
        # opposite of the point. Scale them to the object instead.
        obj_span = max(float(np.mean(params.obj_half)), 1e-3)
        model.vis.scale.contactwidth = obj_span * 0.6
        model.vis.scale.contactheight = obj_span * 0.25
        model.vis.scale.forcewidth = obj_span * 0.3
        model.vis.map.force = obj_span * 4.0

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    # Frame the object, not the (huge) support plane: a fixed wide shot makes a 2 cm object
    # a few pixels across, so the checks this render exists for become impossible.
    traj = np.asarray(L.obj_pos)
    centre = np.nanmean(traj, axis=0)
    cam.lookat[:] = centre
    reach = float(np.nanmax(np.linalg.norm(traj - centre, axis=1)))
    cam.distance = max(2.5 * reach, 12.0 * float(np.max(params.obj_half)), 0.20)
    cam.azimuth, cam.elevation = 135.0, -20.0

    out = []
    with mujoco.Renderer(model, height, width) as r:
        for t in range(len(L.obj_pos)):
            mujoco.mj_resetData(model, data)
            data.qpos[0:3] = L.obj_pos[t]
            data.qpos[3:7] = (L.obj_quat[t] if L.obj_quat is not None
                              else (1.0, 0.0, 0.0, 0.0))
            for k in range(2):
                data.mocap_pos[mocap[k]] = L.finger_pos[t, k]
            mujoco.mj_forward(model, data)
            r.update_scene(data, camera=cam, scene_option=opt)
            out.append(r.render())
    return np.stack(out)


def build_panel(frames, sim, series: dict, title: str, out_png: str, out_mp4: str,
                fps: float):
    """Source frame | simulated scene | score-vs-time, written as PNG stills and an MP4."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = len(sim)
    t = np.arange(T) / max(fps, 1e-6)
    n_panels = 3 if frames is not None else 2

    # Stills at the frames a reader should check: start, the peak of the score, and end.
    score = series.get("grip", np.zeros(T))
    finite = np.where(np.isfinite(score))[0]
    peak = int(finite[np.argmax(np.asarray(score)[finite])]) if len(finite) else T // 2
    picks = sorted({0, peak, T - 1})

    fig, axes = plt.subplots(len(picks), n_panels,
                             figsize=(4.2 * n_panels, 3.2 * len(picks)), squeeze=False)
    for row, ti in enumerate(picks):
        col = 0
        if frames is not None:
            ax = axes[row][col]
            ax.imshow(frames[min(ti, len(frames) - 1)])
            ax.set_title(f"video  t={ti}", fontsize=9)
            ax.axis("off")
            col += 1
        ax = axes[row][col]
        ax.imshow(sim[ti])
        ax.set_title("fitted MuJoCo scene (contacts drawn)", fontsize=9)
        ax.axis("off")
        col += 1
        ax = axes[row][col]
        for key, lab, c in (("grip", "grip demand / (m g)", "tab:red"),
                            ("feas", "feasibility residual", "tab:blue")):
            if key in series:
                ax.plot(t, series[key], color=c, lw=1.4, label=lab)
        if "contact" in series:
            ax.fill_between(t, 0, 1, where=np.asarray(series["contact"], bool),
                            transform=ax.get_xaxis_transform(), color="k", alpha=0.07,
                            label="contact detected")
        ax.axvline(t[ti], color="k", ls="--", lw=1)
        ax.set_xlabel("s", fontsize=8)
        ax.legend(fontsize=7, loc="upper left")
        ax.tick_params(labelsize=7)
        ax.set_title("per-frame score", fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)

    import imageio.v3 as iio
    vid = sim
    if frames is not None:
        H = sim.shape[1]
        rs = []
        for t_ in range(T):
            f = frames[min(t_, len(frames) - 1)]
            sc = H / f.shape[0]
            w = max(int(round(f.shape[1] * sc)), 1)
            idx_y = (np.arange(H) / sc).astype(int).clip(0, f.shape[0] - 1)
            idx_x = (np.arange(w) / sc).astype(int).clip(0, f.shape[1] - 1)
            rs.append(f[np.ix_(idx_y, idx_x)][..., :3])
        vid = np.concatenate([np.stack(rs), sim], axis=2)
    iio.imwrite(out_mp4, vid, fps=max(int(round(fps)), 1))
    return out_png, out_mp4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--which", default="pred", choices=["pred", "gt", "both"])
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-fk-scale", action="store_true")
    ap.add_argument("--force-lift", action="store_true", help="ignore the lift cache")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=360)
    a = ap.parse_args()

    gt_len = None
    gt_rg = None
    if not a.no_fk_scale:
        from .metric_scale import resolve_from_tracks
        try:
            _sc = resolve_from_tracks(a.tracks) or {}
            gt_len = _sc.get("path_len_m")
            gt_rg = _sc.get("rad_gyr_m")
        except Exception as e:
            print(f"[render] FK scale unavailable ({type(e).__name__}: {e}); "
                  f"falling back to the object-size prior")

    from .newton import newton_feasibility

    whichs = ["gt", "pred"] if a.which == "both" else [a.which]
    # In paired mode GT is lifted first and its object size anchors the prediction's metric
    # scale. Grip demand is proportional to scale, so independently-scaled sides compare
    # units rather than physics (measured r = +0.943 between grip ratio and scale ratio).
    ref_extent = None
    for which in whichs:
        L = lift_cached(a.tracks, which, a.out, a.stride, a.device, gt_len, a.force_lift,
                        ref_obj_extent_m=ref_extent, gt_rad_gyr_m=gt_rg)
        if which == "gt" and a.which == "both":
            ref_extent = 2.0 * float(max(L.obj_half))
        params = L.to_params()
        r = newton_feasibility(L.to_observation(), base=params)
        # Render at the friction the sweep selected, so the drawn contacts are the ones the
        # reported number was computed from rather than a default.
        params = L.to_params(mu=r["mu"])
        sim = render_episode(L, params, a.width, a.height)
        frames = _load_frames(a.tracks, which, a.stride)

        series = {"grip": r["grip_series"], "feas": r["ratio_series"],
                  "contact": r["contact_series"]}
        gm = r["grip_demand_mean"]
        title = (f"{which}  |  scale={L.scale_m_per_unit:.4f} m/unit ({L.scale_source})  "
                 f"obj_half={tuple(round(v, 3) for v in L.obj_half)} m  mu={r['mu']}\n"
                 f"grip demand mean={'n/a' if gm is None else f'{gm:.3f}'} x (m g)   "
                 f"feasibility={r['unexplained_mean']:.3f}   "
                 f"contact={r['contact_frac_gripper']:.2f}   "
                 f"orientation aliased={r['orientation_aliased_frac']:.2f}")
        png, mp4 = build_panel(frames, sim, series, title,
                               f"{a.out}_{which}_panel.png", f"{a.out}_{which}.mp4", L.fps)
        print(f"[render] {which}: {png}\n[render] {which}: {mp4}")
        if frames is None:
            print("[render] note: source video path not in tracks meta; sim-only panel")
    os._exit(0)


if __name__ == "__main__":
    main()
