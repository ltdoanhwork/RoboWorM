"""Controls with closed-form answers: is the residual quantitatively right?

Ranking violations correctly is not enough -- a broken estimator can still rank. Each
control below has an expected value derived on paper, so the residual has to land on
a number, not merely in an order.

The three specificity controls are the point of the tier. All three look suspicious
to a purely geometric checker and all three are legitimate physics:

``rest_on_table``      nothing moves -- trivial, but catches sign errors in gravity.
``freefall_true``      touches nothing at all, yet needs zero force. The cheap tier's
                       ``levitation`` predicate must special-case gravity by hand;
                       here it falls out.
``grasp_lift_valid``   accelerating upward against gravity, explained entirely by
                       friction at two opposing contacts. Note this is feasible for
                       *any* ``mu``: opposing fingers may squeeze arbitrarily hard, so
                       the normal force is unbounded and the cone always reaches. That
                       is correct physics, and it is why the friction sweep matters
                       only for single-contact cases like sliding.

``ghost_follow`` is the same motion as ``grasp_lift_valid`` with the gripper moved
0.4 m away. Geometrically the object still "follows the gripper", so the cheap tier's
``grasp_follow`` term is satisfied; dynamically it must read one full body weight.
That pair is the clearest statement of what this tier adds.
"""

from __future__ import annotations

import os

import numpy as np

from .newton import newton_feasibility, newton_residual
from .proxy_scene import G, Observation, ProxyParams

FPS = 16.0
T = 12
DT = 1.0 / FPS


def _far(n: int) -> np.ndarray:
    """Fingers parked far from the object: no contact can be claimed."""
    f = np.zeros((n, 2, 3))
    f[:, 0] = (0.0, -0.4, 0.3)
    f[:, 1] = (0.0, +0.4, 0.3)
    return f


def make_controls(p: ProxyParams) -> dict:
    t = np.arange(T) * DT
    z0 = p.support_z + p.obj_half[2]
    grip_y = p.obj_half[1] + p.finger_half[1] - 0.004
    z_air = 5.0  # high enough that a full g fall never reaches the support plane
    far = _far(T)
    zeros = np.zeros(T)
    out: dict[str, tuple[Observation, str, float | None, str]] = {}

    def obsv(pos, fing, note):
        return Observation(pos, fing, FPS, notes=note)

    # ---- specificity -------------------------------------------------------
    rest = np.stack([zeros, zeros, np.full(T, z0)], axis=1)
    out["rest_on_table"] = (obsv(rest, far, "static, supported"),
                            "unexplained_mean", 0.0, "supported, nothing moves")

    ff = np.stack([zeros, zeros, z_air - 0.5 * G * t**2], axis=1)
    out["freefall_true"] = (obsv(ff, _far(T), "a = g, airborne"),
                            "unexplained_mean", 0.0, "no contact, but Newtonian")

    a_lift = 0.5
    z_lift = z0 + 0.5 * a_lift * t**2
    grip = np.zeros((T, 2, 3))
    grip[:, 0] = np.stack([zeros, np.full(T, -grip_y), z_lift], axis=1)
    grip[:, 1] = np.stack([zeros, np.full(T, +grip_y), z_lift], axis=1)
    lift = np.stack([zeros, zeros, z_lift], axis=1)
    out["grasp_lift_valid"] = (obsv(lift, grip, f"gripped, a = {a_lift} m/s^2 up"),
                               "unexplained_mean", 0.0, "friction cone supplies the lift")

    # ---- sensitivity -------------------------------------------------------
    # Identical object motion to grasp_lift_valid, but airborne and with the gripper
    # 0.4 m away. Offset upward so no support contact can be claimed at t=0 -- otherwise
    # the first frames are legitimately supported and dilute the mean.
    ghost = lift.copy()
    ghost[:, 2] += z_air - z0
    out["ghost_follow"] = (obsv(ghost, _far(T), "same lift, gripper 0.4 m away"),
                           "lift_bias", 1.0 + a_lift / G, "must hold its own weight")

    hover = np.stack([zeros, zeros, np.full(T, z_air)], axis=1)
    out["levitation"] = (obsv(hover, _far(T), "hover, no contact"),
                         "lift_bias", 1.0, "needs exactly m g")

    slow = np.stack([zeros, zeros, z_air - 0.5 * (0.3 * G) * t**2], axis=1)
    out["freefall_slow"] = (obsv(slow, _far(T), "a = 0.3g, airborne"),
                            "lift_bias", 0.7, "needs 0.7 m g of support")

    a_h = 2.0
    drift = np.stack([0.5 * a_h * t**2, zeros, np.full(T, z_air)], axis=1)
    out["drift_airborne"] = (obsv(drift, _far(T), f"a_h = {a_h}, airborne"),
                             "unexplained_horz_mean", a_h / G, "lateral force from nothing")

    # Sliding on the table faster than any swept mu can push. With a normal force N the
    # cone reaches horizontal mu*N while also carrying vertical N; the closest point of
    # the cone to the demanded (m a_h, m g) is a projection, not a subtraction, so the
    # residual is the perpendicular distance to the cone edge:
    #   r = (a_h - mu g) / (g sqrt(1 + mu^2))   in units of m g
    a_slide = 25.0
    mu_max = 1.5
    r_slide = (a_slide - mu_max * G) / (G * np.sqrt(1.0 + mu_max**2))
    slide = np.stack([0.5 * a_slide * t**2, zeros, np.full(T, z0)], axis=1)
    out["slide_beyond_friction"] = (obsv(slide, far, f"a_h = {a_slide} on table"),
                                    "unexplained_mean", None,
                                    f"exceeds the cone at every mu (min at mu={mu_max}: {r_slide:.3f})")

    tele = hover.copy()
    tele[T // 2:, 0] += 0.25
    out["teleport"] = (obsv(tele, _far(T), "0.25 m jump"),
                       "unexplained_max", None, "impulse spike")
    return out


def main():
    p = ProxyParams(mass=0.05, mu=0.8)
    controls = make_controls(p)
    print("Newton feasibility: residual = |m a_obs - m g - sum(cone forces)| / (m g)")
    print(f"proxy obj_half={p.obj_half}  mass={p.mass} kg (cancels by homogeneity)  "
          f"mu={p.mu}  fps={FPS:.0f}  T={T}\n")
    hdr = (f"{'control':24s} {'mean':>7s} {'max':>7s} {'lift_bias':>10s} {'horz':>7s} "
           f"{'ncon':>5s} {'expect':>18s} {'got':>8s}")
    print(hdr); print("-" * len(hdr))
    rows, checks = {}, []
    for name, (obs, term, exp, why) in controls.items():
        r = newton_residual(p, obs)
        rows[name] = r
        got = r[term]
        es = "-" if exp is None else f"{term}~{exp:.3f}"
        print(f"{name:24s} {r['unexplained_mean']:7.3f} {r['unexplained_max']:7.3f} "
              f"{r['lift_bias']:10.3f} {r['unexplained_horz_mean']:7.3f} "
              f"{r['n_contacts_mean']:5.1f} {es:>18s} {got:8.3f}")
        if exp is not None:
            checks.append((f"{name}: {term} ~ {exp:.3f} (got {got:.3f}) -- {why}",
                           abs(got - exp) < 0.08))
    floor = max(rows[k]["unexplained_mean"] for k in ("rest_on_table", "freefall_true",
                                                      "grasp_lift_valid"))
    checks.append((f"teleport max >> floor ({rows['teleport']['unexplained_max']:.2f} vs "
                   f"{floor:.4f})", rows["teleport"]["unexplained_max"] > 10 * max(floor, 1e-3)))
    r_slide_exp = (25.0 - 1.5 * G) / (G * np.sqrt(1.0 + 1.5**2))
    f_slide = newton_feasibility(controls["slide_beyond_friction"][0], base=p)
    checks.append((f"slide_beyond_friction: best-mu residual ~ {r_slide_exp:.3f} "
                   f"(got {f_slide['unexplained_mean']:.3f}) -- distance to the friction cone",
                   abs(f_slide["unexplained_mean"] - r_slide_exp) < 0.08))
    checks.append(("levitation separable from freefall_true",
                   rows["levitation"]["lift_bias"] - rows["freefall_true"]["lift_bias"] > 0.5))
    checks.append(("ghost_follow separable from grasp_lift_valid (same pixels, different physics)",
                   rows["ghost_follow"]["unexplained_mean"]
                   - rows["grasp_lift_valid"]["unexplained_mean"] > 0.5))

    print("\nverdict")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print(f"\n{sum(ok for _, ok in checks)}/{len(checks)} pass   "
          f"noise floor = {floor:.5f} (units of m g)")

    print("\nfriction sweep -- does the search change the verdict?")
    for name in ("grasp_lift_valid", "slide_beyond_friction", "levitation"):
        f = newton_feasibility(controls[name][0], base=p)
        print(f"  {name:24s} best={f['unexplained_mean']:.3f} at mu={f['mu']:.1f}   "
              f"worst={f['worst_mu_unexplained_mean']:.3f}")

    print("\nmass invariance (formulation is homogeneous in m -- should be identical):")
    for mass in (0.01, 0.05, 5.0):
        r = newton_residual(ProxyParams(**{**p.__dict__, "mass": mass}),
                            controls["levitation"][0])
        print(f"  mass={mass:6.2f} kg -> levitation residual {r['unexplained_mean']:.6f}")
    os._exit(0)


if __name__ == "__main__":
    main()
