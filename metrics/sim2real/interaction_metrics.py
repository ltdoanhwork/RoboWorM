"""Physics predicates for the sim2real object-interaction metric.

Replaces RoboWM-Bench's "execute the action in a simulator and run step-level
checkers" with the checkers alone, evaluated on tracked masks. Two families:

**Violations** (reference-free): is what the video shows physically possible at
all? Chiefly ``levitation`` - the object translating while nothing is touching it,
which is exactly the failure the paper illustrates in Fig. 3 (fingers merely
graze the object, yet it gets lifted).

**Agreement** (paired with GT): does the interaction match the physics of *this*
episode? Running the identical tracker on the GT video and differencing cancels
the systematic error of GroundingDINO / SAM2 / depth, which is what makes the
cheap tier trustworthy without a simulator.

Everything here is pure numpy + OpenCV: masks in, JSON-able numbers out.
"""

import numpy as np
import cv2

from metrics.sim2real.lift_backend import mask_centroid, mask_scale, mask_extent

# defaults calibrated on ctrlworld single_arm (see metrics/sim2real/README.md)
DEFAULTS = {
    "tau_gap": 0.35,        # contact when gap < 0.35 object-diameters
    "eps_move": 0.10,       # object "moved" when it travels > 0.10 diameters/frame
    "gap_pct": 5.0,         # percentile of the boundary-distance distribution
    "min_object_px": 12,    # below this the object track is considered lost
    "chatter_tol": 1,       # allowed extra contact on/off transitions vs GT
    "min_present_gt": 0.20, # GT object must be trackable this often, else invalid
    "min_scored_weight": 0.5,   # fraction of the score weight that must be measurable
    "min_gt_contact": 3,    # GT must establish contact for >= this many frames
    # above this median IoU with the arm, the "object" track IS the arm (not an
    # object being held by it) and the episode is unmeasurable
    "max_obj_arm_iou": 0.5,
    # CLIP softmax over the true phrase + 4 distractors -> chance = 1/5 = 0.20.
    # Below chance the tracked crop is no evidence for the named object.
    "min_referent_look": 0.20,
}


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def boundary_gap(mask_a, mask_b, pct=5.0):
    """Robust separation between two masks, in pixels; 0 when they overlap.

    A plain ``min`` distance is decided by a single pixel and so is dominated by
    SAM2 boundary noise; we take a low percentile of the distances from mask A's
    pixels to the nearest pixel of mask B instead. Computed with a distance
    transform, which is O(HW) rather than O(|A|.|B|).
    """
    if not mask_a.any() or not mask_b.any():
        return float("nan")
    if (mask_a & mask_b).any():
        return 0.0
    dt = cv2.distanceTransform((~mask_b).astype(np.uint8), cv2.DIST_L2, 3)
    return float(np.percentile(dt[mask_a], pct))


def overlap_frac(mask_a, mask_b):
    """Fraction of mask A covered by mask B (occlusion / penetration proxy)."""
    a = float(mask_a.sum())
    if a == 0:
        return float("nan")
    return float((mask_a & mask_b).sum() / a)


def gripper_tip(mask_grip, obj_centroid):
    """Point of the arm mask closest to the object centroid.

    The background-subtraction gripper mask is the WHOLE arm (base to fingers), so
    its centroid is dragged toward the base and barely reflects fingertip motion.
    The arm pixel nearest the object is a much better stand-in for the contact
    point, and it is what the grasp-follow test needs.
    """
    if not mask_grip.any() or obj_centroid is None:
        return None
    ys, xs = np.nonzero(mask_grip)
    d2 = (ys - obj_centroid[0]) ** 2 + (xs - obj_centroid[1]) ** 2
    i = int(np.argmin(d2))
    return float(ys[i]), float(xs[i])


def track_series(masks_by_role, gap_pct=5.0, depths=None, lift=None):
    """Per-frame geometric series for one (video, view).

    Returns a dict of length-T arrays:
        ``gap``       object<->gripper separation in object-diameter units (nan if
                      either mask is missing)
        ``cy``,``cx`` object centroid in object-diameter units, nan when lost
        ``scale``     sqrt(object area) in px
        ``area``      object area in px
        ``aspect``    object bbox aspect ratio
        ``ovl``       fraction of the object covered by the gripper mask
        ``present``   object mask non-empty
        ``z_obj``,``z_grip``  optional depth summaries (nan without a depth backend)
    """
    obj = masks_by_role["object"]
    grip = masks_by_role["gripper"]
    T = len(obj)
    out = {k: np.full(T, np.nan) for k in
           ("gap", "cy", "cx", "gy", "gx", "scale", "area", "aspect", "ovl",
            "z_obj", "z_grip")}
    out["present"] = np.zeros(T, bool)
    out["grip_present"] = np.zeros(T, bool)

    # a single reference scale for the whole clip: the median object diameter over
    # frames where it is visible. Using a per-frame scale would make the unit move
    # with the very deformation we are trying to detect.
    scales = np.array([mask_scale(m) for m in obj], dtype=np.float64)
    ref = float(np.median(scales[scales > 0])) if np.any(scales > 0) else 0.0
    ref = ref if ref > 0 else 1.0

    for t in range(T):
        mo, mg = obj[t], grip[t]
        out["present"][t] = bool(mo.any())
        out["scale"][t] = scales[t]
        out["area"][t] = float(mo.sum())
        c = mask_centroid(mo)
        if c is not None:
            out["cy"][t], out["cx"][t] = c[0] / ref, c[1] / ref
        out["grip_present"][t] = bool(mg.any())
        cg = gripper_tip(mg, c)
        if cg is not None:
            out["gy"][t], out["gx"][t] = cg[0] / ref, cg[1] / ref
        h, w = mask_extent(mo)
        if h > 0 and w > 0:
            out["aspect"][t] = w / h
        g = boundary_gap(mo, mg, pct=gap_pct)
        out["gap"][t] = g / ref if np.isfinite(g) else np.nan
        out["ovl"][t] = overlap_frac(mo, mg)
        if depths is not None and lift is not None and lift.has_depth:
            out["z_obj"][t] = lift.entity_z(mo, depths[t]) or np.nan
            out["z_grip"][t] = lift.entity_z(mg, depths[t]) or np.nan

    out["ref_scale_px"] = ref
    return out


CONTACT_YES, CONTACT_NO, CONTACT_UNKNOWN = 1, 0, -1


def contact_from_gap(gap, tau_gap, obj_present=None, grip_present=None):
    """Ternary contact indicator: 1 = contact, 0 = no contact, -1 = unknown.

    ``gap`` is nan whenever either mask is missing, but the two causes are NOT
    equivalent:

    * object missing -> we cannot say anything -> unknown. Typically the arm
      occludes the object; treating that as "no contact" made GT clips register a
      48% levitation rate (the object was being carried while one view could not
      see it).
    * object visible but arm absent -> positive evidence that nothing is touching
      it -> no contact. Collapsing this into unknown is equally wrong: it made the
      teleport-object negative control score a *perfect* levitation term, because
      every frame was skipped for lack of evidence.
    """
    g = np.asarray(gap, dtype=np.float64)
    out = np.full(g.shape, CONTACT_UNKNOWN, dtype=np.int8)
    ok = np.isfinite(g)
    out[ok & (g < tau_gap)] = CONTACT_YES
    out[ok & (g >= tau_gap)] = CONTACT_NO
    if obj_present is not None and grip_present is not None:
        o = np.asarray(obj_present, bool)
        gp = np.asarray(grip_present, bool)
        out[(~ok) & o & (~gp)] = CONTACT_NO
    return out


def and_gate(contacts):
    """Combine per-view ternary indicators; views that cannot see abstain.

    This is the depth-free substitute for 3D. Two cameras looking from different
    angles rarely produce the same *apparent* touch by accident, so a contact is only
    asserted when every informative view agrees - projection makes distant things
    look adjacent, and requiring agreement removes those artifacts.

    Crucially, **disagreement yields UNKNOWN, not NO**. One view seeing a touch is
    real evidence, so a split vote cannot license a levitation claim; it only blocks
    the contact claim. Treating a split as NO made episode_008843 report levitation on
    frames where the primary view measured a gap of 0.00 diameters - a visible
    contradiction in the rendered panel, which is how it was caught.

    Returns 1 (unanimous contact), 0 (unanimous no contact), -1 (split or blind).
    """
    if not contacts:
        return np.zeros(0, dtype=np.int8)
    stack = np.stack([np.asarray(c, dtype=np.int8) for c in contacts], axis=0)
    known = stack != CONTACT_UNKNOWN
    any_known = known.any(axis=0)
    any_yes = ((stack == CONTACT_YES) & known).any(axis=0)
    any_no = ((stack == CONTACT_NO) & known).any(axis=0)
    out = np.full(stack.shape[1], CONTACT_UNKNOWN, dtype=np.int8)
    out[any_known & any_yes & ~any_no] = CONTACT_YES
    out[any_known & any_no & ~any_yes] = CONTACT_NO
    return out


def contact_bool(contact):
    """Treat only explicit contact as True (for timelines / IoU)."""
    return np.asarray(contact, dtype=np.int8) == CONTACT_YES


# --------------------------------------------------------------------------- #
# violation terms (reference-free)
# --------------------------------------------------------------------------- #
def _finite_diff(series):
    """Forward difference with nan propagation; length T (last entry repeated)."""
    s = np.asarray(series, dtype=np.float64)
    d = np.full_like(s, np.nan)
    d[:-1] = s[1:] - s[:-1]
    return d


def object_velocity(series):
    """(speed, dy, dx) per frame in object-diameters/frame.

    ``dy`` follows image convention: positive = moving DOWN the image.
    """
    dy = _finite_diff(series["cy"])
    dx = _finite_diff(series["cx"])
    speed = np.sqrt(dy ** 2 + dx ** 2)
    return speed, dy, dx


def levitation(series, contact, eps_move, grace=1):
    """Object translates while nothing touches it, and not by falling.

    A free object may only accelerate downwards (gravity) or stay put. Motion that
    is sideways or *upwards* with no contact is unsupported - the physical
    impossibility RoboWM-Bench catches by executing the action in a simulator.

    Only frames where contact is KNOWN to be absent count. Frames where no view can
    see both entities are skipped: claiming a violation from an occluded frame is
    what produced a 48% levitation rate on real GT video before the ternary
    contact state was introduced.

    ``grace`` dilates the known-contact indicator by +-grace frames so a contact
    detected one frame late does not read as levitation (at 5 fps one frame is
    200 ms, so this is a real quantisation concern, not a fudge).
    """
    speed, dy, dx = object_velocity(series)
    c = np.asarray(contact, dtype=np.int8)
    yes = c == CONTACT_YES
    unknown = c == CONTACT_UNKNOWN
    if grace > 0 and yes.size:
        k = np.ones(2 * grace + 1)
        yes = np.convolve(yes.astype(float), k, mode="same") > 0
        unknown = np.convolve(unknown.astype(float), k, mode="same") > 0

    valid = np.isfinite(speed) & series["present"]
    moving = valid & (speed > eps_move)
    # "free" needs positive evidence of NO contact, not merely absence of a YES
    free = moving & (~yes) & (~unknown)
    # falling = predominantly downward motion (dy > 0 dominates |dx|)
    falling = free & (dy > 0) & (dy > np.abs(dx))
    viol = free & (~falling)

    n_valid = int((moving & (~unknown)).sum())
    return {
        "levitation_rate": float(viol.sum() / n_valid) if n_valid else 0.0,
        "levitation_frames": np.flatnonzero(viol).tolist(),
        "free_move_frames": int(free.sum()),
        "moving_frames": n_valid,
        "unknown_contact_frames": int((c == CONTACT_UNKNOWN).sum()),
        "mean_free_speed": float(np.nanmean(speed[free])) if free.any() else 0.0,
    }


def grasp_follow(series, contact, eps_move):
    """During contact a rigidly grasped object must move with the gripper.

    Reports the median ratio ``|v_obj - v_grip| / |v_grip|`` and the median cosine
    between the two velocities. A perfect grasp gives ratio 0 and cosine 1; an
    object that sits still while the gripper drags away gives ratio ~1.
    """
    speed_o, dy_o, dx_o = object_velocity(series)
    dy_g = _finite_diff(series["gy"]) if "gy" in series else None
    dx_g = _finite_diff(series["gx"]) if "gx" in series else None
    if dy_g is None:
        return {"grasp_follow_ratio": float("nan"), "grasp_follow_cos": float("nan"),
                "grasp_frames": 0}

    c = contact_bool(contact)
    m = c & np.isfinite(dy_o) & np.isfinite(dy_g)
    # only frames where the gripper is actually moving; a stationary gripper makes
    # the ratio ill-conditioned
    gs = np.sqrt(dy_g ** 2 + dx_g ** 2)
    m &= gs > eps_move
    if not m.any():
        return {"grasp_follow_ratio": float("nan"), "grasp_follow_cos": float("nan"),
                "grasp_frames": 0}

    rel = np.sqrt((dy_o - dy_g) ** 2 + (dx_o - dx_g) ** 2)
    ratio = rel[m] / np.maximum(gs[m], 1e-6)
    dot = dy_o[m] * dy_g[m] + dx_o[m] * dx_g[m]
    norm = np.maximum(np.sqrt(dy_o[m] ** 2 + dx_o[m] ** 2) * gs[m], 1e-6)
    return {
        "grasp_follow_ratio": float(np.median(ratio)),
        "grasp_follow_cos": float(np.median(dot / norm)),
        "grasp_frames": int(m.sum()),
    }


def object_integrity(series, gt_series=None):
    """Does the object keep a stable size/shape and stay trackable?

    Catches the "object dissolves / melts" artifact (PAI-Bench VQA Q4 in the paper)
    that SAM2 reports as an area collapse or an aspect-ratio swing. Scored against
    the GT clip's own variability when available, since a genuinely rotating object
    also changes its silhouette.
    """
    area = np.asarray(series["area"], dtype=np.float64)
    ok = np.isfinite(area) & (area > 0)
    present = float(np.mean(series["present"])) if len(series["present"]) else 0.0

    def cv(x):
        x = x[np.isfinite(x) & (x > 0)]
        return float(np.std(x) / np.mean(x)) if x.size > 1 and np.mean(x) > 0 else 0.0

    area_cv = cv(area)
    aspect_cv = cv(np.asarray(series["aspect"], dtype=np.float64))
    out = {
        "object_present_frac": present,
        "object_area_cv": area_cv,
        "object_aspect_cv": aspect_cv,
        "object_lost_frames": int((~np.asarray(series["present"], bool)).sum()),
    }
    if gt_series is not None:
        gt_area_cv = cv(np.asarray(gt_series["area"], dtype=np.float64))
        gt_aspect_cv = cv(np.asarray(gt_series["aspect"], dtype=np.float64))
        out["object_area_cv_excess"] = float(max(0.0, area_cv - gt_area_cv))
        out["object_aspect_cv_excess"] = float(max(0.0, aspect_cv - gt_aspect_cv))
        out["object_present_deficit"] = float(
            max(0.0, float(np.mean(gt_series["present"])) - present))
    _ = ok
    return out


def contact_chatter(contact, gt_contact=None, tol=1):
    """Number of contact on/off transitions in excess of GT.

    A real grasp is made once and held; a generated clip whose fingers flicker in
    and out of the object produces many transitions.
    """
    def transitions(c):
        c = contact_bool(c)
        return int(np.sum(c[1:] != c[:-1])) if c.size > 1 else 0

    n = transitions(contact)
    out = {"contact_transitions": n}
    if gt_contact is not None:
        g = transitions(gt_contact)
        out["contact_transitions_gt"] = g
        out["contact_chatter_excess"] = int(max(0, n - g - tol))
    return out


def penetration(series, gt_series=None):
    """Gripper mask covering more of the object than physically sensible.

    Without depth we cannot distinguish occlusion from interpenetration, so we
    measure the *excess* over the same quantity in GT: the real gripper occludes
    the real object by some amount, and a prediction whose gripper swallows the
    object well beyond that is either penetrating it or has lost its geometry.
    """
    ovl = np.asarray(series["ovl"], dtype=np.float64)
    ovl = ovl[np.isfinite(ovl)]
    peak = float(np.percentile(ovl, 95)) if ovl.size else float("nan")
    out = {"gripper_overlap_p95": peak}
    if gt_series is not None:
        g = np.asarray(gt_series["ovl"], dtype=np.float64)
        g = g[np.isfinite(g)]
        gpeak = float(np.percentile(g, 95)) if g.size else float("nan")
        out["gripper_overlap_p95_gt"] = gpeak
        if np.isfinite(peak) and np.isfinite(gpeak):
            out["penetration_excess"] = float(max(0.0, peak - gpeak))
    return out


# --------------------------------------------------------------------------- #
# agreement terms (paired with GT)
# --------------------------------------------------------------------------- #
def first_true(mask):
    idx = np.flatnonzero(np.asarray(mask, bool))
    return int(idx[0]) if idx.size else None


def contact_agreement(contact_pred, contact_gt, fps=5.0):
    """Contact-timing agreement: onset error, temporal IoU, F1."""
    p = contact_bool(contact_pred)
    g = contact_bool(contact_gt)
    T = min(p.size, g.size)
    p, g = p[:T], g[:T]

    op, og = first_true(p), first_true(g)
    inter = float((p & g).sum())
    union = float((p | g).sum())
    tp, fp, fn = inter, float((p & ~g).sum()), float((~p & g).sum())

    out = {
        "contact_onset_pred": op,
        "contact_onset_gt": og,
        "contact_frac_pred": float(p.mean()) if T else float("nan"),
        "contact_frac_gt": float(g.mean()) if T else float("nan"),
        "contact_temporal_iou": float(inter / union) if union > 0 else float("nan"),
        "contact_f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else float("nan"),
    }
    if op is not None and og is not None:
        out["contact_onset_err_frames"] = int(abs(op - og))
        out["contact_onset_err_s"] = float(abs(op - og) / fps)
    else:
        # one side never makes contact at all - a hard failure, not a small error
        out["contact_onset_err_frames"] = None
        out["contact_onset_err_s"] = None
        out["contact_onset_missing"] = ("pred" if op is None else "gt")
    return out


def trajectory_agreement(series_pred, series_gt):
    """Object-centroid trajectory error in object-diameter units.

    Both series are already normalised by their own clip's median object diameter,
    so this is directly comparable across episodes and resolutions.
    """
    py, px = np.asarray(series_pred["cy"]), np.asarray(series_pred["cx"])
    gy, gx = np.asarray(series_gt["cy"]), np.asarray(series_gt["cx"])
    T = min(py.size, gy.size)
    py, px, gy, gx = py[:T], px[:T], gy[:T], gx[:T]
    m = np.isfinite(py) & np.isfinite(gy)
    if not m.any():
        return {"obj_traj_err": float("nan"), "obj_traj_err_max": float("nan"),
                "obj_traj_valid_frac": 0.0}
    d = np.sqrt((py[m] - gy[m]) ** 2 + (px[m] - gx[m]) ** 2)
    # displacement error: how far each clip's object travelled from its own start
    def travel(y, x):
        mm = np.isfinite(y)
        if mm.sum() < 2:
            return float("nan")
        yy, xx = y[mm], x[mm]
        return float(np.sqrt((yy[-1] - yy[0]) ** 2 + (xx[-1] - xx[0]) ** 2))

    return {
        "obj_traj_err": float(np.mean(d)),
        "obj_traj_err_max": float(np.max(d)),
        "obj_traj_valid_frac": float(m.mean()),
        "obj_travel_pred": travel(py, px),
        "obj_travel_gt": travel(gy, gx),
    }


def gap_agreement(series_pred, series_gt):
    """Mean absolute difference of the approach profile gap(t)."""
    p = np.asarray(series_pred["gap"], dtype=np.float64)
    g = np.asarray(series_gt["gap"], dtype=np.float64)
    T = min(p.size, g.size)
    p, g = p[:T], g[:T]
    m = np.isfinite(p) & np.isfinite(g)
    if not m.any():
        return {"gap_curve_err": float("nan"), "gap_min_pred": float("nan"),
                "gap_min_gt": float("nan")}
    return {
        "gap_curve_err": float(np.mean(np.abs(p[m] - g[m]))),
        "gap_min_pred": float(np.nanmin(p)) if np.isfinite(p).any() else float("nan"),
        "gap_min_gt": float(np.nanmin(g)) if np.isfinite(g).any() else float("nan"),
    }


# --------------------------------------------------------------------------- #
# step-level nodes (simulator-free version of RoboWM-Bench Table 1)
# --------------------------------------------------------------------------- #
def derive_nodes_from_gt(series_gt, contact_gt, eps_move, lift_frac=0.5):
    """Infer this episode's key interaction stages from the GT clip.

    RoboWM-Bench hand-specifies key nodes per task and checks them in simulation.
    We instead read them off the GT trajectory, which keeps the protocol automatic
    across 130 episodes and grounds every threshold in what actually happened:

      ``contact``   first GT contact frame
      ``lift``      first frame after contact where the object has risen more than
                    ``lift_frac`` diameters above its pre-contact height
      ``transport`` frame of maximum horizontal displacement while in contact
      ``release``   last GT contact frame
    """
    c = contact_bool(contact_gt)
    cy = np.asarray(series_gt["cy"], dtype=np.float64)
    cx = np.asarray(series_gt["cx"], dtype=np.float64)
    onset = first_true(c)
    nodes = {}
    if onset is None:
        return nodes

    nodes["contact"] = {"frame": onset}
    base = np.nanmedian(cy[:max(1, onset)]) if onset > 0 else cy[onset]
    rise = base - cy  # image y grows downward, so rise > 0 means the object went UP
    after = np.arange(len(cy)) >= onset
    lifted = after & c & np.isfinite(rise) & (rise > lift_frac)
    lf = first_true(lifted)
    if lf is not None:
        nodes["lift"] = {"frame": int(lf), "rise": float(rise[lf]),
                         "base_cy": float(base), "thresh": float(lift_frac)}

    inc = after & c & np.isfinite(cx)
    if inc.any():
        disp = np.abs(cx - np.nanmedian(cx[:max(1, onset)]))
        disp = np.where(inc, disp, -np.inf)
        tf = int(np.argmax(disp))
        if np.isfinite(disp[tf]) and disp[tf] > eps_move:
            nodes["transport"] = {"frame": tf, "disp": float(disp[tf])}

    idx = np.flatnonzero(c)
    if idx.size:
        nodes["release"] = {"frame": int(idx[-1])}
    return nodes


def check_nodes(nodes_gt, series_pred, contact_pred, eps_move,
                onset_tol_frames=5, pos_tol=1.0):
    """Pass/fail each GT-derived node against the prediction.

    Every tolerance is expressed in the same units as the GT measurement it is
    compared to, so the table is interpretable next to the paper's step-level one.
    """
    c = contact_bool(contact_pred)
    cy = np.asarray(series_pred["cy"], dtype=np.float64)
    cx = np.asarray(series_pred["cx"], dtype=np.float64)
    res = {}

    if "contact" in nodes_gt:
        op = first_true(c)
        gt_f = nodes_gt["contact"]["frame"]
        res["contact"] = {
            "gt_frame": gt_f, "pred_frame": op,
            "pass": bool(op is not None and abs(op - gt_f) <= onset_tol_frames),
        }

    if "lift" in nodes_gt:
        gt_f = nodes_gt["lift"]["frame"]
        thresh = nodes_gt["lift"]["thresh"]
        onset = nodes_gt["contact"]["frame"]
        base = np.nanmedian(cy[:max(1, onset)]) if onset > 0 else (
            cy[onset] if onset < cy.size else np.nan)
        rise = base - cy
        after = np.arange(cy.size) >= onset
        lifted = after & c & np.isfinite(rise) & (rise > thresh)
        pf = first_true(lifted)
        res["lift"] = {
            "gt_frame": gt_f, "pred_frame": pf,
            "gt_rise": nodes_gt["lift"]["rise"],
            "pred_rise": float(np.nanmax(np.where(after, rise, np.nan)))
            if np.isfinite(rise).any() else None,
            "pass": bool(pf is not None),
        }

    if "transport" in nodes_gt:
        gt_f = nodes_gt["transport"]["frame"]
        gt_disp = nodes_gt["transport"]["disp"]
        onset = nodes_gt["contact"]["frame"]
        ref = np.nanmedian(cx[:max(1, onset)]) if onset > 0 else (
            cx[onset] if onset < cx.size else np.nan)
        disp = np.abs(cx - ref)
        pd = float(np.nanmax(disp)) if np.isfinite(disp).any() else float("nan")
        res["transport"] = {
            "gt_frame": gt_f, "gt_disp": gt_disp, "pred_disp": pd,
            "pass": bool(np.isfinite(pd) and abs(pd - gt_disp) <= pos_tol),
        }

    if "release" in nodes_gt:
        idx = np.flatnonzero(c)
        pf = int(idx[-1]) if idx.size else None
        gt_f = nodes_gt["release"]["frame"]
        res["release"] = {
            "gt_frame": gt_f, "pred_frame": pf,
            "pass": bool(pf is not None and abs(pf - gt_f) <= onset_tol_frames),
        }
    _ = eps_move
    return res


# --------------------------------------------------------------------------- #
# episode-level assembly
# --------------------------------------------------------------------------- #
def _norm(value, good, bad):
    """Map a raw term to [0,1] (1 = good). Linear between ``good`` and ``bad``."""
    if value is None or not np.isfinite(value):
        return None
    if bad == good:
        return 1.0
    x = (value - good) / (bad - good)
    return float(np.clip(1.0 - x, 0.0, 1.0))


# term -> (json key, good value, bad value, weight). "good"/"bad" anchors come from
# the negative-control sweep; see metrics/sim2real/README.md.
#
# Violation terms are scored on their GT-relative EXCESS, not their raw value: the
# tracker leaves a residual violation signal even on real video (occlusion, mask
# jitter), and the oracle control (pred := GT) only reaches 1.0 if that residual is
# differenced away. The raw values are still reported for diagnosis.
# (name, flat key, good value, bad value, weight, group)
#
# ``group`` decides whether a term is fair to charge against a given model:
#   "physics"   - judges the prediction on its own (GT enters only as a per-episode
#                 baseline that cancels tracker bias, never as a per-frame target),
#                 so it is fair for ANY model.
#   "agreement" - requires reproducing THIS GT rollout frame-for-frame. Only fair
#                 when the model was told what to do, i.e. action-conditioned.
SCORE_TERMS = [
    ("levitation", "levitation_rate_excess", 0.0, 0.30, 1.0, "physics"),
    ("grasp_follow", "grasp_follow_ratio_excess", 0.0, 0.80, 1.0, "physics"),
    ("object_presence", "object_present_deficit", 0.0, 0.30, 1.0, "physics"),
    ("object_shape", "object_shape_excess", 0.0, 0.50, 1.0, "physics"),
    ("penetration", "penetration_excess", 0.0, 0.30, 0.5, "physics"),
    ("chatter", "contact_chatter_excess", 0.0, 6.0, 0.5, "physics"),
    ("contact_timing", "contact_onset_err_frames", 1.0, 15.0, 1.0, "agreement"),
    ("contact_overlap", "contact_temporal_iou", 1.0, 0.0, 1.0, "agreement"),
    ("trajectory", "obj_traj_err", 0.2, 3.0, 1.0, "agreement"),
    ("gap_curve", "gap_curve_err", 0.2, 3.0, 0.5, "agreement"),
]

# Which term groups count towards the headline score, per model conditioning.
SCORE_GROUPS_BY_CONDITIONING = {
    # Ctrl-World & friends: fed the per-frame action/state sequence, so GT IS the
    # expected answer and timing disagreement is a real error.
    "action": ("physics", "agreement"),
    # DreamGen video2world_lvg: fed ONE frame + a text sentence. It cannot know that
    # GT grasped at frame 6, so charging it for grasping at frame 18 measures luck,
    # not physics. Measured on GR1 episode_002133: physics 0.983 (clean grasp, no
    # levitation/penetration, object keeps shape) but agreement 0.529 (onset 12
    # frames late) dragged the headline to 0.796 - a plausible rollout scored as the
    # worst in the set. Agreement stays reported, just not charged.
    "text": ("physics",),
}


def interaction_score(flat, groups=("physics", "agreement")):
    """Weighted mean of normalised sub-terms, plus the per-term breakdown.

    ``groups`` selects which term groups are *charged*; every term is still
    normalised and reported. See :data:`SCORE_GROUPS_BY_CONDITIONING` for why a
    text-conditioned model must not be charged the agreement group.

    The breakdown is ALWAYS reported: metrics/dynamics/README.md records how
    WorldScore's single max-minus-max number misled us, and a lone scalar here
    would repeat that mistake.

    Also returns the fraction of the *charged* weight that was actually measurable,
    which the caller uses to reject episodes: a total tracking failure leaves every
    informative term at None and the few defaults at 1.0, which would otherwise
    score a PERFECT 1.000 (observed on episode_000397, where the object was never
    detected in any view).
    """
    parts, total_w, acc, all_w = {}, 0.0, 0.0, 0.0
    sub_acc, sub_w = {}, {}
    for name, key, good, bad, w, grp in SCORE_TERMS:
        v = flat.get(key)
        n = _norm(v if v is None else float(v), good, bad)
        parts[name] = {"raw": None if v is None else float(v), "norm": n,
                       "weight": w, "group": grp,
                       "charged": grp in groups}
        if n is not None:
            sub_acc[grp] = sub_acc.get(grp, 0.0) + w * n
            sub_w[grp] = sub_w.get(grp, 0.0) + w
        if grp not in groups:
            continue
        all_w += w
        if n is not None:
            acc += w * n
            total_w += w
    score = float(acc / total_w) if total_w > 0 else float("nan")
    subscores = {g: (float(sub_acc[g] / sub_w[g]) if sub_w.get(g) else None)
                 for g in {t[5] for t in SCORE_TERMS}}
    return score, parts, (total_w / all_w if all_w > 0 else 0.0), subscores


def check_validity(series_gt, series_pred, scored_weight, p, contact_gt=None,
                   obj_arm_iou_gt=None, referent_look=None):
    """Is this episode measurable at all?

    A metric that silently rewards its own tracking failures is worse than no
    metric, so an episode is only scored when the GT object track exists, the GT
    actually contains an interaction, and enough of the score weight is measurable.
    Invalid episodes get ``score = None`` and are excluded from summaries and
    rankings.

    The GT-contact requirement matters as much as the tracking one: on
    episode_001471 the arm never came within 1.5 object diameters of the tracked
    object in GT, so there was no interaction to reproduce - yet the episode scored
    0.984 simply because no interaction term could be violated.

    ``obj_arm_iou_gt`` is the last line of defence against the object track having
    latched onto the ARM (GR1 episode_001065: IoU 0.87, "object" area 4.7% vs arm
    5.4%). Then the gripper-object gap is a gap to itself, every contact term is
    trivially perfect, and the episode scores a false-clean 0.975. The tracker
    already vetoes such candidates; this catches the case where every candidate was
    the arm and the veto had to fall back.

    ``referent_look`` gates on *identity*: every check above asks whether the track
    is geometrically sane, none asks whether the thing being tracked is the object
    the instruction named. On GR1 episode_003085 ("picks up the yellow corn") the
    winning candidate had CLIP P=0.09 for its own phrase and the mask sat on a green
    pepper, yet the episode published 0.779. ``look`` is a softmax over the true
    phrase plus 4 distractor captions, so chance is 1/5 = 0.20; below that the crop
    is not evidence for the named object at all and the episode is an identity
    failure, not a physics measurement. The threshold is chance level rather than a
    tuned value on purpose - it only removes picks that are *unsupported*, and the
    correct picks here clear it comfortably (0.29-0.77).
    """
    reasons = []
    pres_gt = float(np.mean(series_gt["present"])) if len(series_gt["present"]) else 0.0
    pres_pred = float(np.mean(series_pred["present"])) if len(series_pred["present"]) else 0.0
    if pres_gt < p["min_present_gt"]:
        reasons.append(f"gt_object_untracked({pres_gt:.2f}<{p['min_present_gt']})")
    if scored_weight < p["min_scored_weight"]:
        reasons.append(f"too_few_terms({scored_weight:.2f}<{p['min_scored_weight']})")
    if series_gt["ref_scale_px"] <= 1.0:
        reasons.append("gt_object_degenerate")
    n_contact_gt = None
    if contact_gt is not None:
        n_contact_gt = int(contact_bool(contact_gt).sum())
        if n_contact_gt < p["min_gt_contact"]:
            reasons.append(f"no_gt_contact({n_contact_gt}<{p['min_gt_contact']})")
    if obj_arm_iou_gt is not None and obj_arm_iou_gt > p["max_obj_arm_iou"]:
        reasons.append(f"object_is_arm(iou={obj_arm_iou_gt:.2f}>"
                       f"{p['max_obj_arm_iou']})")
    if referent_look is not None and referent_look < p["min_referent_look"]:
        reasons.append(f"referent_unverified(look={referent_look:.2f}<"
                       f"{p['min_referent_look']})")
    return {
        "valid": not reasons,
        "invalid_reasons": reasons,
        "object_present_frac_gt": pres_gt,
        "object_present_frac_pred": pres_pred,
        "scored_weight_frac": float(scored_weight),
        "gt_contact_frames": n_contact_gt,
        "obj_arm_iou_gt": (None if obj_arm_iou_gt is None else float(obj_arm_iou_gt)),
        "referent_look": (None if referent_look is None else float(referent_look)),
    }


def _obj_arm_iou(masks):
    """Median IoU between the object and gripper masks (identity check, not contact).

    High IoU means the two tracks are the SAME thing, i.e. the object track latched
    onto the arm. A grasped object stays a small island inside the arm mask and so
    has low IoU (0.13 measured), whereas an arm-as-object track sits near 0.87.
    """
    o, g = masks.get("object"), masks.get("gripper")
    if not o or not g:
        return None
    n = min(len(o), len(g))
    vals = []
    for i in range(n):
        a, b = o[i], g[i]
        if not a.any():
            continue
        union = np.logical_or(a, b).sum()
        if union:
            vals.append(np.logical_and(a, b).sum() / union)
    return float(np.median(vals)) if vals else None


def view_metrics(masks_pred, masks_gt, gap_pct, tau_gap, eps_move,
                 depths_pred=None, depths_gt=None, lift=None):
    """All per-view series + the per-view contact indicators."""
    sp = track_series(masks_pred, gap_pct=gap_pct, depths=depths_pred, lift=lift)
    sg = track_series(masks_gt, gap_pct=gap_pct, depths=depths_gt, lift=lift)
    return {
        "pred": sp, "gt": sg,
        "contact_pred": contact_from_gap(sp["gap"], tau_gap, sp["present"], sp["grip_present"]),
        "contact_gt": contact_from_gap(sg["gap"], tau_gap, sg["present"], sg["grip_present"]),
        "obj_arm_iou_gt": _obj_arm_iou(masks_gt),
    }


def episode_metrics(views, static_views, fps=5.0, params=None, referent_look=None,
                    conditioning="action"):
    """Full metric set for one episode.

    ``views`` is a list of dicts as returned by :func:`view_metrics`, indexed the
    same way as the tracker's views; ``static_views`` selects which of them take
    part in the AND-gated contact decision (the wrist camera moves, so "the object
    moved but nothing touched it" is meaningless there and it is excluded).

    ``conditioning`` says what the model was given ("action" or "text") and thus
    which term groups may be charged - see :data:`SCORE_GROUPS_BY_CONDITIONING`.
    """
    p = dict(DEFAULTS)
    if params:
        p.update(params)

    sel = [i for i in static_views if i < len(views)]
    if not sel:
        sel = list(range(len(views)))
    contact_pred = and_gate([views[i]["contact_pred"] for i in sel])
    contact_gt = and_gate([views[i]["contact_gt"] for i in sel])

    # geometry is reported from the first static view (the primary exterior camera)
    main = views[sel[0]]
    sp, sg = main["pred"], main["gt"]

    flat = {}
    flat.update(levitation(sp, contact_pred, p["eps_move"]))
    lev_gt = levitation(sg, contact_gt, p["eps_move"])
    flat["levitation_rate_gt"] = lev_gt["levitation_rate"]
    flat["levitation_rate_excess"] = float(
        max(0.0, flat["levitation_rate"] - lev_gt["levitation_rate"]))
    flat.update(grasp_follow(sp, contact_pred, p["eps_move"]))
    gf_gt = grasp_follow(sg, contact_gt, p["eps_move"])
    flat["grasp_follow_ratio_gt"] = gf_gt["grasp_follow_ratio"]
    flat["grasp_follow_cos_gt"] = gf_gt["grasp_follow_cos"]
    if np.isfinite(flat["grasp_follow_ratio"]) and np.isfinite(gf_gt["grasp_follow_ratio"]):
        flat["grasp_follow_ratio_excess"] = float(
            max(0.0, flat["grasp_follow_ratio"] - gf_gt["grasp_follow_ratio"]))
    flat.update(object_integrity(sp, gt_series=sg))
    # one shape term: whichever of area / aspect deviates from GT more
    flat["object_shape_excess"] = float(max(
        flat.get("object_area_cv_excess", 0.0) or 0.0,
        flat.get("object_aspect_cv_excess", 0.0) or 0.0))
    flat.update(contact_chatter(contact_pred, gt_contact=contact_gt, tol=p["chatter_tol"]))
    flat.update(penetration(sp, gt_series=sg))
    flat.update(contact_agreement(contact_pred, contact_gt, fps=fps))
    flat.update(trajectory_agreement(sp, sg))
    flat.update(gap_agreement(sp, sg))

    nodes_gt = derive_nodes_from_gt(sg, contact_gt, p["eps_move"])
    node_res = check_nodes(nodes_gt, sp, contact_pred, p["eps_move"])
    n_pass = sum(1 for v in node_res.values() if v.get("pass"))
    step_level = {
        "nodes_gt": nodes_gt,
        "nodes": node_res,
        "num_nodes": len(node_res),
        "num_passed": n_pass,
        "step_pass_rate": float(n_pass / len(node_res)) if node_res else float("nan"),
        "task_success": bool(node_res) and all(v.get("pass") for v in node_res.values()),
    }

    groups = SCORE_GROUPS_BY_CONDITIONING.get(conditioning, ("physics", "agreement"))
    score, parts, scored_w, subscores = interaction_score(flat, groups=groups)
    ious = [views[i].get("obj_arm_iou_gt") for i in static_views
            if views[i].get("obj_arm_iou_gt") is not None]
    validity = check_validity(sg, sp, scored_w, p, contact_gt=contact_gt,
                              obj_arm_iou_gt=(min(ious) if ious else None),
                              referent_look=referent_look)
    if not validity["valid"]:
        score = None
    per_view = {}
    for i, vw in enumerate(views):
        cp, cg = np.asarray(vw["contact_pred"], np.int8), np.asarray(vw["contact_gt"], np.int8)
        per_view[f"view_{i}"] = {
            "contact_frac_pred": float(np.mean(cp == CONTACT_YES)),
            "contact_frac_gt": float(np.mean(cg == CONTACT_YES)),
            "unknown_frac_pred": float(np.mean(cp == CONTACT_UNKNOWN)),
            "unknown_frac_gt": float(np.mean(cg == CONTACT_UNKNOWN)),
            "gap_min_pred": float(np.nanmin(vw["pred"]["gap"]))
            if np.isfinite(vw["pred"]["gap"]).any() else None,
            "gap_min_gt": float(np.nanmin(vw["gt"]["gap"]))
            if np.isfinite(vw["gt"]["gap"]).any() else None,
            "object_present_frac_pred": float(np.mean(vw["pred"]["present"])),
            "object_present_frac_gt": float(np.mean(vw["gt"]["present"])),
            "in_and_gate": i in sel,
        }

    return {
        "sim2real_interaction_score": score,
        "physics_score": subscores.get("physics"),
        "agreement_score": subscores.get("agreement"),
        "conditioning": conditioning,
        "charged_groups": list(groups),
        "valid": validity["valid"],
        "validity": validity,
        "score_terms": parts,
        "violations": {k: flat[k] for k in (
            "levitation_rate", "levitation_rate_gt", "levitation_rate_excess",
            "moving_frames", "free_move_frames", "unknown_contact_frames",
            "grasp_follow_ratio", "grasp_follow_cos", "grasp_follow_ratio_gt",
            "grasp_follow_cos_gt", "grasp_frames", "grasp_follow_ratio_excess",
            "object_present_frac", "object_area_cv", "object_aspect_cv",
            "object_lost_frames", "object_area_cv_excess", "object_aspect_cv_excess",
            "object_present_deficit", "object_shape_excess",
            "contact_transitions", "contact_transitions_gt",
            "contact_chatter_excess", "gripper_overlap_p95", "gripper_overlap_p95_gt",
            "penetration_excess") if k in flat},
        "agreement": {k: flat[k] for k in (
            "contact_onset_pred", "contact_onset_gt", "contact_onset_err_frames",
            "contact_onset_err_s", "contact_onset_missing", "contact_frac_pred",
            "contact_frac_gt", "contact_temporal_iou", "contact_f1",
            "obj_traj_err", "obj_traj_err_max", "obj_traj_valid_frac",
            "obj_travel_pred", "obj_travel_gt", "gap_curve_err",
            "gap_min_pred", "gap_min_gt") if k in flat},
        "step_level": step_level,
        "per_view": per_view,
        "levitation_frames": flat.get("levitation_frames", []),
        "contact_pred": np.asarray(contact_pred, np.int8).tolist(),
        "contact_gt": np.asarray(contact_gt, np.int8).tolist(),
        "params": p,
        "fps": fps,
        "and_gate_views": sel,
        # gap curves of the primary view, for interaction_viz (underscore = not a metric)
        "_gap_pred": np.asarray(sp["gap"], float).tolist(),
        "_gap_gt": np.asarray(sg["gap"], float).tolist(),
        "_present_pred": np.asarray(sp["present"], bool).tolist(),
        "_present_gt": np.asarray(sg["present"], bool).tolist(),
        "_main_view": sel[0],
    }

