"""Newton feasibility: can *any* admissible contact force explain the observed motion?

Why this replaces force-tracking
--------------------------------
The first version of this tier drove the object along the observed path with a PD
controller inside ``mj_step`` and reported the injected force. It worked, but it
introduced three constants that had to be tuned by hand -- tracking bandwidth,
damping, force cap -- and the residual inherited controller lag, which is exactly
why an honest free-fall control scored 0.77 instead of 0.

Worse, it depended on *penetration depth* to produce grip force. A kinematically
driven rigid finger generates almost no normal force in a rigid contact solver
(measured: the proxy grasp slipped with 4 mm of overlap), so grip had to be faked
with an explicit contact stiffness -- another constant, and one that would have been
read off noisy monocular depth in the real pipeline.

The formulation here has none of that. At each frame we ask a feasibility question:

    does there exist a set of contact forces, each inside its friction cone,
    that together with gravity produces the acceleration we observed?

    minimise ||  A lambda  -  (m a_obs - m g)  ||   subject to  lambda >= 0

``A``'s columns are friction-cone generators built from the contact normals MuJoCo
detects at the observed configuration. The residual is the leftover force -- the
part of the motion no admissible contact could have caused.

Two properties make this the right measurement for video
--------------------------------------------------------
*Mass cancels exactly.* The problem is homogeneous of degree one in mass: scaling
``m`` scales both ``b`` and the optimal ``lambda``, so the normalised residual
``||r|| / (m g)`` is invariant. Object mass is unknowable from a video, and here it
simply does not enter. Contact stiffness does not enter either.

*Only ``mu`` remains*, and it is swept rather than calibrated, so a trajectory is
charged only when no plausible friction explains it.

MuJoCo still does the physics that matters: collision detection and contact frames
at the observed configuration. What we drop is forward integration, which is what
made the result depend on controller gains.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog, nnls

try:
    import mujoco
except Exception as _e:  # pragma: no cover
    mujoco = None
    _MJ_ERR = _e

from .proxy_scene import G, Observation, ProxyParams, build_proxy_xml

# Tangent directions used to linearise each friction cone as a pyramid. Four is the
# standard choice: it under-approximates the cone, so the residual is conservative
# (we never invent contact authority the real cone would not have).
_TANGENTS = 4


def _min_contact_force(A: np.ndarray, b: np.ndarray,
                       weights: np.ndarray | None = None) -> float:
    """Smallest weighted contact force explaining wrench ``b``, or inf if none does.

    With ``weights`` marking the gripper's own cone columns, this becomes "how much must the
    *fingers* carry", leaving the support plane free to take the rest. That is what makes the
    number defined on the frames that dominate real episodes -- an object resting on a table
    while a hand approaches -- instead of only during a clean airborne hold. An object at rest
    on the table correctly costs 0: the table does all the work.

    This is the quantity that stays meaningful once a grasp is force-closure. Two opposing
    frictional face contacts span all of R^6 (measured: rank 6, worst relative NNLS residual
    2.3e-15 over 500 random wrenches), so asking *whether* a wrench is feasible is vacuous for
    a held object -- the answer is always yes, for the torque rows as much as the force rows.
    Adding torque balance therefore did not make grasps falsifiable, and no amount of extra
    geometry would have; the question itself was wrong.

    An LP over the same non-negative cone coefficients used for the feasibility residual, so
    the friction model is identical and no new assumption enters.
    """
    if A.shape[1] == 0:
        return float("inf")
    c = np.ones(A.shape[1]) if weights is None else weights
    r = linprog(c, A_eq=A, b_eq=b, bounds=(0.0, None), method="highs")
    return float(r.fun) if r.success else float("inf")


def _cone_generators(normal: np.ndarray, frame: np.ndarray, mu: float) -> np.ndarray:
    """Pyramid generators for one contact: ``n + mu * t_j``, columns of a 3xK block.

    Any force inside the linearised cone is a non-negative combination of these, so
    imposing ``lambda >= 0`` in the least-squares problem is exactly the friction
    constraint -- no inequality solver beyond NNLS is needed.
    """
    t1, t2 = frame[1], frame[2]
    cols = []
    for j in range(_TANGENTS):
        a = 2.0 * np.pi * j / _TANGENTS
        t = np.cos(a) * t1 + np.sin(a) * t2
        cols.append(normal + mu * t)
    return np.stack(cols, axis=1)


def _quat_to_R(q: np.ndarray) -> np.ndarray:
    """(w, x, y, z) -> rotation matrix."""
    w, x, y, z = q / max(float(np.linalg.norm(q)), 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis-angle vector (the SO(3) logarithm).

    Exact for any rotation up to pi, unlike a finite difference on the matrix entries.
    """
    ang = float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
    if ang < 1e-9:
        return np.zeros(3)
    if ang > np.pi - 1e-6:
        # Near pi the skew part vanishes; recover the axis from the symmetric part.
        A = (R + np.eye(3)) / 2.0
        k = int(np.argmax(np.diag(A)))
        axis = A[:, k] / max(np.sqrt(max(A[k, k], 1e-12)), 1e-12)
        return ang * axis / max(float(np.linalg.norm(axis)), 1e-12)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (ang / (2.0 * np.sin(ang)))


def _angular_kinematics(quat: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rotations, angular velocity, angular acceleration, and a per-frame aliasing flag.

    ``omega`` is built from single-step relative rotations, ``log(R[t] R[t+1]^T)/dt``
    evaluated at frame midpoints and averaged to the frame centres. Two alternatives were
    measured and rejected: a finite difference on the matrix entries (``skew(w) = Rdot
    R^T``) is only first-order accurate in the per-step angle and read **62% low at
    150 rad/s^2**; and the log map over a *centred two-frame* window halves the usable
    rate, because it is the rotation across the whole window that must stay below pi.
    Single-step logs are exact for rotation about a fixed axis (verified to 0.0000% error)
    and push the wrap limit as high as the frame rate allows.

    ``alpha`` is a central difference on omega. Both are undefined (NaN) at the window
    edges for the same reason the linear stencil is: a one-sided estimate of a derivative
    at a boundary is a guess, and a guess here would appear as fictitious torque on the
    first and last frames.

    Rotation beyond pi radians in a single step is **aliased and unrecoverable** -- no
    estimator can distinguish a large rotation from its wrapped equivalent, and ``arccos``
    silently returns the wrapped value, so a fast spin can *appear* slow. Measured: at
    30 fps a 2000 rad/s^2 spin-up has a true omega of 333 rad/s but reads -13.2. Two
    guards, both threshold-free in the sense that they use only the geometry of SO(3):

    1. A step angle above pi/2 is not certifiable. Only ``theta <= pi/2`` guarantees that
       the measured value is the smaller of the two candidates *and* leaves a factor-of-two
       margin, which is what makes the derivative meaningful rather than merely defined.
       Verified: with this bound the estimator is exact (0.0000% on omega and alpha) up to
       175 rad/s^2 at 30 fps, and flags everything above.
    2. Neighbouring single-step logs must not point in opposite directions (a dot product
       below zero). Wrapping reverses the apparent axis, so this catches wraps that happen
       to land back under the angle bound. Orthogonality is a natural boundary, not a
       tuned constant.

    Flagged frames are dropped, not reported as a small residual; without this an absurd
    spin scored *lower* than a merely implausible one.
    """
    T = len(quat)
    R = np.stack([_quat_to_R(q) for q in quat])
    omega = np.full((T, 3), np.nan)
    aliased = np.zeros(T, dtype=bool)
    # Angular velocity at the midpoint between frames t and t+1, in the world frame.
    mid = np.full((max(T - 1, 1), 3), np.nan)
    for t in range(T - 1):
        rel = R[t + 1] @ R[t].T
        ang = float(np.arccos(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)))
        if ang > 0.5 * np.pi:
            aliased[max(t - 1, 0):min(t + 3, T)] = True
            continue
        mid[t] = _so3_log(rel) / dt
    for t in range(len(mid) - 1):
        a, b = mid[t], mid[t + 1]
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 1e-6 and nb > 1e-6 and float(a @ b) / (na * nb) < 0.0:
            aliased[max(t - 1, 0):min(t + 4, T)] = True
    for t in range(1, T - 1):
        if np.isnan(mid[t - 1]).any() or np.isnan(mid[t]).any():
            continue
        omega[t] = 0.5 * (mid[t - 1] + mid[t])
    alpha = np.full((T, 3), np.nan)
    alpha[2:-2] = (omega[3:-1] - omega[1:-3]) / (2.0 * dt)
    return R, omega, alpha, aliased


def _box_inertia(mass: float, half: tuple[float, float, float]) -> np.ndarray:
    """Body-frame inertia of the proxy box."""
    hx, hy, hz = half
    c = mass / 3.0
    return np.diag([c * (hy * hy + hz * hz), c * (hx * hx + hz * hz), c * (hx * hx + hy * hy)])


def _contacts_at(model, data, gid_obj: int, mu: float,
                 com: np.ndarray | None = None,
                 gid_grip: set | None = None) -> tuple[np.ndarray, int, np.ndarray]:
    """Wrench generator matrix for all contacts touching the object.

    Returns ``(A, n_contacts, is_grip)`` where ``A`` is 3x(4N) (force only) when ``com`` is
    None and 6x(4N) (force stacked on torque about ``com``) otherwise, and ``is_grip`` marks
    which columns belong to gripper contacts -- needed to ask how much of the load the
    *fingers* must carry while still letting the support plane participate in equilibrium.

    Signs are normalised so every generator pushes *on the object*: MuJoCo's contact
    frame points from geom1 to geom2, so the direction flips depending on which slot
    the object occupies. Getting this wrong would let the solver pull the object into
    surfaces and silently explain away levitation.
    """
    cols = []
    grip_flags = []
    n_con = 0
    for c in range(data.ncon):
        con = data.contact[c]
        g1, g2 = int(con.geom1), int(con.geom2)
        if gid_obj not in (g1, g2):
            continue
        frame = np.array(con.frame).reshape(3, 3)
        normal = frame[0].copy()
        if g1 == gid_obj:
            normal = -normal
        F = _cone_generators(normal, frame, mu)
        if com is None:
            cols.append(F)
        else:
            r = np.array(con.pos) - com
            Tq = np.cross(r, F, axisb=0).T
            cols.append(np.concatenate([F, Tq], axis=0))
        other = g2 if g1 == gid_obj else g1
        is_g = bool(gid_grip and other in gid_grip)
        grip_flags.extend([is_g] * F.shape[1])
        n_con += 1
    if not cols:
        return np.zeros((3 if com is None else 6, 0)), 0, np.zeros(0, dtype=bool)
    return np.concatenate(cols, axis=1), n_con, np.array(grip_flags, dtype=bool)


def newton_residual(params: ProxyParams, obs: Observation,
                    smooth: int = 0) -> dict:
    """Per-frame unexplained force, in units of the object's own weight.

    ``smooth`` optionally applies a centred moving average to the observed positions
    before differentiating. Acceleration is a second derivative, so tracker noise is
    amplified; this is exposed as a knob rather than hard-coded because the right
    amount depends on the tracker, and the paired-GT difference is what ultimately
    cancels the bias.
    """
    if mujoco is None:
        raise RuntimeError(f"mujoco unavailable: {_MJ_ERR}")
    obj = np.asarray(obs.obj_pos, dtype=float)
    fing = np.asarray(obs.finger_pos, dtype=float)
    T = len(obj)
    if T < 3:
        raise ValueError("need >= 3 frames")

    if smooth and smooth > 1:
        k = np.ones(smooth) / smooth
        obj = np.stack([np.convolve(obj[:, i], k, mode="same") for i in range(3)], axis=1)

    model = mujoco.MjModel.from_xml_string(build_proxy_xml(params))
    data = mujoco.MjData(model)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    gid_obj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obj")
    gid_sup = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "support")
    gid_f = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("f0", "f1")}
    mocap = [model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)]
             for n in ("finger0", "finger1")]
    mass = float(model.body_mass[bid])
    mg = mass * G
    g_vec = np.array([0.0, 0.0, -G])

    dt = 1.0 / float(obs.fps)
    # Central second difference, exact for quadratic motion (i.e. constant
    # acceleration -- free fall, the case we must not charge). Applying
    # ``np.gradient`` twice instead falls back to one-sided first-order stencils at
    # the boundary, which put a spurious 0.5 m g on the first and last frames of an
    # honest free fall. Acceleration simply is not determined at the edge of a finite
    # window, so those frames are marked invalid rather than estimated.
    acc = np.full_like(obj, np.nan)
    acc[1:-1] = (obj[2:] - 2.0 * obj[1:-1] + obj[:-2]) / (dt * dt)
    valid = np.zeros(T, dtype=bool)
    valid[1:-1] = True

    # Angular half of Newton-Euler, when orientation was measured.
    quat = getattr(obs, "obj_quat", None)
    use_torque = quat is not None
    if use_torque:
        quat = np.asarray(quat, dtype=float)
        if len(quat) != T:
            raise ValueError(f"obj_quat has {len(quat)} frames, obj_pos has {T}")
        Rb, omega, alpha, aliased = _angular_kinematics(quat, dt)
        # Orientation is scored per frame, not per episode. A frame whose rotation is
        # aliased or at the derivative boundary still has a perfectly good *position*
        # second difference, and force balance never needed orientation. Dropping such
        # frames outright discarded everything on real video (measured: 11 of 12
        # gt/pred passes reported aliased_frac >= 0.97 and n_frames_scored = 0, because
        # PCA axes on monocular point clouds jitter far more than a rigid object does).
        # So the angular rows are switched off frame by frame and the fraction reported.
        torque_ok = (~aliased) & ~np.isnan(alpha).any(axis=1)
        aliased_frac = float(aliased.mean())
    else:
        torque_ok = np.zeros(T, dtype=bool)
        aliased_frac = 0.0

    if use_torque:
        I_body = _box_inertia(mass, params.obj_half)
    res = np.zeros(T)
    res_v = np.zeros(T)
    res_h = np.zeros(T)
    res_tau = np.zeros(T)
    grip = np.full(T, np.nan)
    n_contacts = np.zeros(T, dtype=int)
    touch_grip = np.zeros(T, dtype=bool)
    touch_sup = np.zeros(T, dtype=bool)

    for t in range(T):
        if not valid[t]:
            continue
        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = obj[t]
        data.qpos[3:7] = quat[t] if use_torque else (1.0, 0.0, 0.0, 0.0)
        for k in range(2):
            data.mocap_pos[mocap[k]] = fing[t, k]
        mujoco.mj_forward(model, data)

        for c in range(data.ncon):
            pair = {int(data.contact[c].geom1), int(data.contact[c].geom2)}
            if gid_obj in pair and pair & gid_f:
                touch_grip[t] = True
            if gid_obj in pair and gid_sup in pair:
                touch_sup[t] = True

        rot_t = bool(use_torque and torque_ok[t])
        n_rows = 6 if rot_t else 3
        A, nc, is_grip = _contacts_at(model, data, gid_obj, params.mu,
                                     com=obj[t] if rot_t else None, gid_grip=gid_f)
        n_contacts[t] = nc
        b = np.zeros(n_rows)
        b[:3] = mass * acc[t] - mass * g_vec       # net force the motion demands
        if rot_t:
            # Euler's equation in the world frame: tau = I w' + w x (I w), with the
            # inertia rotated into the current orientation.
            Iw = Rb[t] @ I_body @ Rb[t].T
            b[3:] = Iw @ alpha[t] + np.cross(omega[t], Iw @ omega[t])
        # Non-dimensionalise BEFORE solving: force rows by m g, torque rows by m g L
        # (L = the object's own size, the natural torque scale). Solving the raw system
        # would have NNLS minimise a sum of newtons and newton-metres, whose relative
        # weight is arbitrary -- with L = 0.03 m the torque block is ~30x smaller in
        # magnitude, so the solver effectively ignored it and the reported torque residual
        # was just whatever was left over. Measured symptom: a 80 rad/s^2 spin scored
        # 0.190 while a 40 rad/s^2 spin scored 0.260, i.e. non-monotone in alpha.
        row_scale = np.empty(n_rows)
        row_scale[:3] = mg
        if rot_t:
            row_scale[3:] = mg * max(float(np.mean(params.obj_half)), 1e-4)
        b /= row_scale
        if A.shape[1] == 0:
            r = b
        else:
            A = A / row_scale[:, None]
            lam, _ = nnls(A, b)
            r = b - A @ lam
        res[t] = float(np.linalg.norm(r[:3]))
        res_v[t] = float(r[2])
        res_h[t] = float(np.linalg.norm(r[:2]))
        res_tau[t] = float(np.linalg.norm(r[3:])) if rot_t else np.nan
        # Grip demand, in units of the object's weight. Only defined when the fingers are
        # the ones doing the work; with support contact the table can carry the load and the
        # number would describe the table, not the grasp.
        # Grip demand, in units of the object's weight: the load the *fingers* must carry.
        # Support contact is left free (weight 0) rather than excluded, so the number is
        # defined on the frames that dominate real episodes -- object on a table, hand
        # arriving -- and correctly reads 0 when the table is doing all the work.
        if touch_grip[t] and A.shape[1] > 0 and is_grip.any():
            # The row rescaling above divides both A and b by the same factors, so the LP
            # solution lambda keeps its original units (newtons). Dividing by m g is what
            # actually makes the number a ratio -- verified by the mass sweep in the
            # controls, which reported 0.373 / 1.863 / 186.3 for 0.01 / 0.05 / 5.00 kg
            # before this division, i.e. exactly proportional to mass.
            f = _min_contact_force(A, b, weights=is_grip.astype(float))
            grip[t] = f / mg if np.isfinite(f) else np.nan

    v = valid
    if not v.any():
        # Every frame was excluded (edges plus aliasing). Returning NaNs is the honest
        # outcome; a zero here would read as "perfectly physical".
        nan = float("nan")
        return {"mass_kg": mass, "mu": params.mu, "n_frames_scored": 0,
                "unexplained_mean": nan, "unexplained_p95": nan, "unexplained_max": nan,
                "unexplained_vert_mean": nan, "unexplained_horz_mean": nan, "lift_bias": nan,
                "n_contacts_mean": nan, "contact_frac_gripper": nan,
                "contact_frac_support": nan, "unsupported_frac": nan,
                "unexplained_mean_held": nan, "unexplained_max_held": nan,
                "n_frames_held": 0, "held_frac": 0.0,
                "torque_checked": bool(use_torque),
                "orientation_aliased_frac": aliased_frac,
                "torque_frac_scored": 0.0,
                "grip_demand_mean": None, "grip_demand_p95": None, "grip_demand_max": None,
                "n_frames_grip": 0,
                "unexplained_torque_mean": nan, "unexplained_torque_max": nan,
                "unexplained_torque_mean_held": nan,
                "ratio_series": res, "ratio_vert_series": res_v, "torque_series": res_tau,
                "grip_series": grip,
                "contact_series": touch_grip | touch_sup, "valid_series": v,
                "all_frames_excluded": True}
    # Frames where the proxy found no contact at all are reported separately. In a clip
    # where the object is supported throughout, "no contact detected" forces the residual
    # to ~1 (an unsupported object must fall), so mixing those frames in makes the score a
    # restatement of the segmentation's reach rather than a measure of physics. Measured on
    # 6 real episodes: excess correlated -0.9999 with the change in contact fraction, slope
    # -1.000 -- the metric carried no dynamical content until these were separated.
    held = (touch_grip | touch_sup) & v
    res_held = res[held] if held.any() else np.array([np.nan])
    return {
        "mass_kg": mass,
        "mu": params.mu,
        "n_frames_scored": int(v.sum()),
        "unexplained_mean": float(res[v].mean()),
        "unexplained_p95": float(np.percentile(res[v], 95)),
        "unexplained_max": float(res[v].max()),
        "unexplained_vert_mean": float(np.abs(res_v[v]).mean()),
        "unexplained_horz_mean": float(res_h[v].mean()),
        "lift_bias": float(res_v[v].mean()),
        "n_contacts_mean": float(n_contacts[v].mean()),
        "contact_frac_gripper": float(touch_grip[v].mean()),
        "contact_frac_support": float(touch_sup[v].mean()),
        "unsupported_frac": float((~touch_grip[v] & ~touch_sup[v]).mean()),
        # The physics-carrying quantity: unexplained force *on frames where a contact was
        # actually found*. On no-contact frames the residual is pinned to ~1 by geometry,
        # so only these frames can distinguish "unphysical" from "unobserved".
        "unexplained_mean_held": float(np.nanmean(res_held)),
        "unexplained_max_held": float(np.nanmax(res_held)),
        "n_frames_held": int(held.sum()),
        "held_frac": float(held.sum() / max(int(v.sum()), 1)),
        # Angular balance. Reported separately because it is the term that can fail while
        # force balance succeeds -- the case a grasped object always falls into.
        "torque_checked": bool(use_torque),
        "orientation_aliased_frac": aliased_frac,
        # Fraction of scored frames that actually carried usable orientation. Reported
        # because the torque statistics below describe only those frames, and on real
        # monocular video that subset is often small.
        "torque_frac_scored": float((torque_ok & v).sum() / max(int(v.sum()), 1)),
        # Grip demand: the informative statistic for held objects, where feasibility is
        # vacuous. Units of the object's weight; a firm human pinch is a few m g.
        "grip_demand_mean": float(np.nanmean(grip[v])) if np.isfinite(grip[v]).any() else None,
        "grip_demand_p95": (float(np.nanpercentile(grip[v][np.isfinite(grip[v])], 95))
                            if np.isfinite(grip[v]).any() else None),
        "grip_demand_max": float(np.nanmax(grip[v])) if np.isfinite(grip[v]).any() else None,
        "n_frames_grip": int(np.isfinite(grip[v]).sum()),
        "unexplained_torque_mean": (float(np.nanmean(res_tau[v])) if use_torque
                                    and np.isfinite(res_tau[v]).any() else None),
        "unexplained_torque_max": (float(np.nanmax(res_tau[v])) if use_torque
                                   and np.isfinite(res_tau[v]).any() else None),
        "unexplained_torque_mean_held": (float(np.nanmean(res_tau[held]))
                                         if use_torque and held.any()
                                         and np.isfinite(res_tau[held]).any() else None),
        "ratio_series": res,
        "ratio_vert_series": res_v,
        "torque_series": res_tau,
        "grip_series": grip,
        "contact_series": touch_grip | touch_sup,
        "valid_series": v,
    }


# Only friction survives as an unknown; mass and contact stiffness cancel out of the
# formulation entirely, so the sweep is one-dimensional.
MU_GRID = (0.2, 0.4, 0.8, 1.5)


def newton_feasibility(obs: Observation, base: ProxyParams | None = None,
                       mu_grid=MU_GRID, **kw) -> dict:
    """Best explanation over the friction sweep: charge only what no ``mu`` explains.

    Selection uses the held-frame residual, since that is the quantity being reported;
    picking on the all-frame mean would let a run win by detecting fewer contacts.
    """
    base = base or ProxyParams()
    runs = []
    for mu in mu_grid:
        p = ProxyParams(**{**base.__dict__, "mu": float(mu)})
        runs.append(newton_residual(p, obs, **kw))

    def key(r):
        v = r["unexplained_mean_held"]
        return r["unexplained_mean"] if np.isnan(v) else v

    best = min(runs, key=key)
    return {**best,
            "n_mu_tried": len(runs),
            "worst_mu_unexplained_mean": float(max(r["unexplained_mean"] for r in runs))}
