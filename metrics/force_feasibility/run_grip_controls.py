"""Controls for held objects: what is actually measurable once a grasp exists.

Feasibility -- "could contact forces inside the friction cones explain this motion?" -- is
the right question for a *free* or *supported* object, and `run_newton_controls.py` shows it
works there. For a *held* object it is vacuous, and this was measured rather than assumed:
two opposing frictional face contacts give a wrench matrix of rank 6, and over 500 random
wrenches the worst relative NNLS residual was 2.3e-15. Every wrench is explainable. Adding
the torque rows did not change that, because the span already covered all of R^6.

So the question is replaced, not patched. The informative quantity is **how hard the
fingers must squeeze**: the minimum total contact force, in units of the object's weight,
that explains the observed motion. It is finite, monotone in the demanded acceleration, and
scale-free. A grasp that needs 1.2 m g is a normal pinch; one that needs 500 m g is a video
model asserting a force no hand applies.

Every case below is held for the whole clip, so feasibility is satisfied by construction and
any signal has to come from the grip demand.
"""

from __future__ import annotations

import os

import numpy as np

from .newton import newton_residual
from .proxy_scene import G, Observation, ProxyParams

FPS = 30.0
T = 16
DT = 1.0 / FPS


def _quat_axis(axis: np.ndarray, ang: np.ndarray) -> np.ndarray:
    """Quaternion series for a rotation of ``ang[t]`` about a fixed axis."""
    a = axis / np.linalg.norm(axis)
    h = ang / 2.0
    return np.stack([np.cos(h), a[0] * np.sin(h), a[1] * np.sin(h), a[2] * np.sin(h)], axis=1)


def make_controls(p: ProxyParams) -> dict:
    z0 = p.support_z + p.obj_half[2]
    grip_y = p.obj_half[1] + p.finger_half[1] - 0.004
    zeros = np.zeros(T)
    t = np.arange(T) * DT
    out: dict[str, tuple[Observation, str, str]] = {}

    # Held and clear of the table, so the support plane cannot carry the load and the
    # number describes the grasp rather than the table.
    z_hold = z0 + 0.15
    pos = np.stack([zeros, zeros, np.full(T, z_hold)], axis=1)

    def grip_at(a):
        f = np.zeros((T, 2, 3))
        f[:, 0] = a + np.array([0.0, -grip_y, 0.0])
        f[:, 1] = a + np.array([0.0, +grip_y, 0.0])
        return f

    fing = grip_at(pos)
    ident = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (T, 1))

    # ---- baseline: holding still costs a bounded, small grip ------------------
    out["held_static"] = (Observation(pos, fing, FPS, obj_quat=ident, notes="held, still"),
                          "grip_demand_mean", "just carrying the weight")

    # ---- linear acceleration: grip must grow with it --------------------------
    for a_lin, tag in ((2.0, "held_accel_mild"), (20.0, "held_accel_hard"),
                       (200.0, "held_accel_absurd")):
        pa = np.stack([0.5 * a_lin * t**2, zeros, np.full(T, z_hold)], axis=1)
        out[tag] = (Observation(pa, grip_at(pa), FPS, obj_quat=ident,
                               notes=f"held, linear accel {a_lin} m/s^2"),
                    "grip_demand_mean", f"{a_lin:g} m/s^2 sideways")

    # ---- angular acceleration: same, through the moment arm ------------------
    # Rates kept inside the certified band (the estimator is exact for every frame it
    # certifies; beyond ~pi/2 per step orientation is aliased and is flagged instead).
    for a_ang, tag in ((20.0, "held_spin_mild"), (80.0, "held_spin_hard")):
        q = _quat_axis(np.array([0.0, 0.0, 1.0]), 0.5 * a_ang * t**2)
        out[tag] = (Observation(pos, fing, FPS, obj_quat=q,
                               notes=f"held, alpha {a_ang} rad/s^2"),
                    "grip_demand_mean", f"{a_ang:g} rad/s^2 spin-up")

    # ---- specificity: unmeasurable rotation must be flagged, not scored ------
    out["held_spin_aliased"] = (Observation(pos, fing, FPS,
                                           obj_quat=_quat_axis(np.array([0.0, 0.0, 1.0]),
                                                               0.5 * 8000.0 * t**2),
                                           notes="held, alpha 8000 rad/s^2"),
                                "orientation_aliased_frac", "unmeasurable: flag, don't score")

    tc = t[T // 2]
    ang_r = np.where(t <= tc, 12.0 * t, 12.0 * tc - 12.0 * (t - tc))
    out["held_spin_reverse"] = (Observation(pos, fing, FPS,
                                           obj_quat=_quat_axis(np.array([0.0, 1.0, 0.0]), ang_r),
                                           notes="held, omega flips sign"),
                                "orientation_aliased_frac",
                                "ambiguous with a 2pi wrap: flag")

    # ---- specificity: object on the table, hand touching but not lifting -----
    # This is the most common frame in real episodes, and the first version of grip demand
    # refused to score it (it excluded any frame with support contact), which is why the
    # real-data run came back n/a. The table can carry the weight, so the *fingers* need
    # ~nothing: the LP charges only the gripper's own cone columns.
    pos_rest = np.stack([zeros, zeros, np.full(T, z0)], axis=1)
    out["resting_hand_touching"] = (Observation(pos_rest, grip_at(pos_rest), FPS,
                                              obj_quat=ident,
                                              notes="on table, fingers touching, static"),
                                    "grip_demand_mean", "table carries it: fingers ~0")

    # ---- specificity: no grasp -> grip demand undefined, not zero -----------
    far = np.zeros((T, 2, 3))
    far[:, 0] = (0.0, -0.6, 0.4)
    far[:, 1] = (0.0, +0.6, 0.4)
    ff = np.stack([zeros, zeros, 5.0 - 0.5 * G * t**2], axis=1)
    out["free_fall_no_grasp"] = (Observation(ff, far, FPS, obj_quat=ident,
                                           notes="free fall, nothing touching"),
                                 "grip_demand_mean", "nothing to squeeze")
    return out


def main():
    p = ProxyParams(mass=0.05, mu=0.8, obj_half=(0.03, 0.03, 0.03))
    C = make_controls(p)
    print("Held objects: grip demand = min total contact force / (m g)")
    print(f"proxy obj_half={p.obj_half}  mu={p.mu}  mass={p.mass} kg  fps={FPS:.0f}  T={T}")
    print("feasibility residual is ~0 in every case here BY CONSTRUCTION --")
    print("two opposing frictional contacts span R^6 (rank 6, worst residual 2.3e-15),")
    print("so 'is it feasible' cannot fail for a grasp. Grip demand is what carries signal.\n")

    def fmt(x, w=8, d=3):
        return f"{'n/a':>{w}s}" if x is None or (isinstance(x, float) and np.isnan(x)) \
            else f"{x:{w}.{d}f}"

    hdr = (f"{'control':22s} {'feas':>7s} {'grip_mean':>10s} {'grip_max':>9s} "
           f"{'alias':>6s} {'nval':>5s} {'ngrip':>6s}  note")
    print(hdr); print("-" * len(hdr))
    rows = {}
    for name, (obs, term, why) in C.items():
        r = newton_residual(p, obs)
        rows[name] = r
        print(f"{name:22s} {fmt(r['unexplained_mean'], 7):>7s} "
              f"{fmt(r['grip_demand_mean'], 10):>10s} {fmt(r['grip_demand_max'], 9):>9s} "
              f"{r['orientation_aliased_frac']:6.2f} {r['n_frames_scored']:5d} "
              f"{r['n_frames_grip']:6d}  {why}")

    g = lambda k: rows[k]["grip_demand_mean"]
    have_spin = g("held_spin_mild") is not None and g("held_spin_hard") is not None
    checks = [
        ("held_static grip is finite and small (< 3 m g)",
         g("held_static") is not None and g("held_static") < 3.0),
        ("grip grows with linear accel (mild < hard < absurd)",
         g("held_accel_mild") < g("held_accel_hard") < g("held_accel_absurd")),
        ("grip grows with angular accel (mild < hard)",
         have_spin and g("held_spin_mild") < g("held_spin_hard")),
        ("absurd accel demands >10x the static grip",
         g("held_accel_absurd") > 10 * g("held_static")),
        ("feasibility stays ~0 on translation (it CANNOT discriminate a grasp)",
         max(rows[k]["unexplained_mean"] for k in
             ("held_static", "held_accel_mild", "held_accel_hard", "held_accel_absurd")) < 0.02),
        ("resting on the table costs the fingers ~nothing (< 0.1 m g)",
         g("resting_hand_touching") is not None and g("resting_hand_touching") < 0.1),
        ("held (airborne) costs ~the weight, far more than resting",
         g("held_static") > 10 * max(g("resting_hand_touching"), 1e-3)),
        ("no grasp -> grip demand is None, not 0",
         rows["free_fall_no_grasp"]["grip_demand_mean"] is None),
        ("aliased spin is FLAGGED",
         rows["held_spin_aliased"]["orientation_aliased_frac"] > 0.5),
        ("reversal is FLAGGED as ambiguous",
         rows["held_spin_reverse"]["orientation_aliased_frac"] > 0.1),
        ("certified controls are not falsely flagged",
         max(rows[k]["orientation_aliased_frac"] for k in
             ("held_static", "held_accel_hard", "held_spin_mild",
              "held_spin_hard", "free_fall_no_grasp")) == 0.0),
    ]
    print("\nverdict")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print(f"\n{sum(ok for _, ok in checks)}/{len(checks)} pass")

    print("\nscale-freedom: grip demand is a force RATIO, so it must not move with mass")
    for mass in (0.01, 0.05, 5.0):
        r = newton_residual(ProxyParams(**{**p.__dict__, "mass": mass}),
                            C["held_accel_hard"][0])
        print(f"  mass={mass:6.2f} kg -> grip demand {r['grip_demand_mean']:.6f} x (m g)")

    print("\nfriction dependence (lower mu -> must squeeze harder for the same motion):")
    for mu in (0.2, 0.5, 0.8, 1.5):
        r = newton_residual(ProxyParams(**{**p.__dict__, "mu": mu}),
                            C["held_accel_hard"][0])
        print(f"  mu={mu:4.1f} -> grip demand {r['grip_demand_mean']:8.3f} x (m g)")
    os._exit(0)


if __name__ == "__main__":
    main()
