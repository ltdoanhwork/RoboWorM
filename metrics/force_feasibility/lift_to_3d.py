"""Bridge: masks from the cheap tier -> a metric 3D trajectory Newton can be applied to.

``metrics/sim2real`` already segments the manipulated object and the gripper in every
frame. What it deliberately does not produce is 3D, or metres: its distances are in
"object diameters", which is enough for geometric predicates and not enough for force.

This module closes that gap using only what is already in the repo (the packed masks
in ``tracks.npz``, the paired GT video, and MoGe for point maps). Three problems have
to be solved, and each is solved by fitting to data rather than by a constant:

**Where is "down"?** MoGe's camera frame has no notion of gravity, and these cameras
are tilted. A vertical axis asserted by hand would make every levitation measurement a
function of that guess. Instead the support surface is found by RANSAC on the point map
and its normal *is* the vertical axis. Gravity then points along a measured direction.

**What is the metric scale?** MoGe is scale-ambiguous. One scalar per episode is fitted
by matching the tracked gripper's total 3D path length to the paired GT end-effector
path length in metres (via ``metrics/embodied_exec/episode_index.py``). When no EE
column exists for that embodiment, the object's own fitted size provides a fallback and
``scale_source`` records which was used -- the number is never silently invented.

**Which pixels?** Object and gripper centroids come from the existing masks; depth is
the median over the mask (robust to the boundary bleed that ruins a mean).

Note the measured scale sensitivity (see README): severe violations are scale-invariant,
so the scale step sets resolution, not validity.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass

import numpy as np


def unpack_masks(z, key: str, hw: tuple[int, int], num_frames: int) -> np.ndarray:
    """Undo the bit packing ``track_backend`` uses to store per-frame masks."""
    flat = np.unpackbits(z[key])
    h, w = hw
    need = num_frames * h * w
    if flat.size < need:
        raise ValueError(f"{key}: {flat.size} bits < {need} needed for {num_frames}x{h}x{w}")
    return flat[:need].reshape(num_frames, h, w).astype(bool)


def fit_plane_ransac(pts: np.ndarray, iters: int = 200, thresh: float = 0.02,
                     seed: int = 0) -> tuple[np.ndarray, float, float]:
    """Dominant plane in a point cloud -> (unit normal, offset, inlier fraction).

    Used to recover the vertical axis from the support surface. The inlier fraction is
    returned so a failed fit (no dominant plane in view) can be detected instead of
    silently producing a wrong gravity direction.
    """
    rng = np.random.default_rng(seed)
    n = len(pts)
    if n < 3:
        return np.array([0.0, 0.0, 1.0]), 0.0, 0.0
    best = (None, 0.0, 0.0)
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        p0, p1, p2 = pts[idx]
        nv = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(nv)
        if nn < 1e-9:
            continue
        nv = nv / nn
        d = -float(nv @ p0)
        inl = float(np.mean(np.abs(pts @ nv + d) < thresh))
        if inl > best[2]:
            best = (nv, d, inl)
    if best[0] is None:
        return np.array([0.0, 0.0, 1.0]), 0.0, 0.0
    return best


def _gripper_contact_point(pts: np.ndarray, valid: np.ndarray, m_grip: np.ndarray,
                           m_obj: np.ndarray) -> np.ndarray | None:
    """The gripper point closest to the object, not the gripper's centroid.

    Segmentation returns the whole arm: measured on a real episode the gripper mask is
    ~25x the object's area and its centroid sits 1065 px (8.4 object diameters) away,
    while the two masks' nearest boundaries are 0.0 px apart -- they touch. A centroid
    therefore reports "no grasp" on every frame of a successful pick. What the proxy
    needs is the part of the arm that could be in contact, so we take the gripper pixels
    nearest the object in the image and use their median 3D point.
    """
    import cv2

    mg = m_grip & valid
    if not mg.any() or not m_obj.any():
        return None
    dist_to_obj = cv2.distanceTransform((~m_obj).astype(np.uint8), cv2.DIST_L2, 3)
    d = np.where(mg, dist_to_obj, np.inf)
    thresh = np.percentile(d[mg], 2.0)
    sel = mg & (d <= max(thresh, 1.0))
    if not sel.any():
        return None
    return np.median(pts[sel], axis=0)


def object_orientation(obj_pts: list[np.ndarray | None]) -> tuple[np.ndarray | None, np.ndarray]:
    """Per-frame object orientation from its own 3D points, as (T, 4) (w, x, y, z).

    The axes come from a PCA of the object's point cloud in the gravity frame. PCA alone is
    not enough: eigenvectors have arbitrary sign and, for a near-symmetric object, arbitrary
    order, so a raw per-frame PCA jitters by 90/180 degrees between frames and would read as
    an enormous angular acceleration -- exactly the failure the aliasing guard then reports
    as unmeasurable. Each frame is therefore resolved against the previous one by choosing,
    among the 24 rotations of a cube (all axis permutations with positive determinant), the
    one closest to it. Aligning to the *previous frame* rather than to a canonical pose is
    what makes the derivative well-defined; absolute orientation is not needed, since Euler's
    equation only involves omega and alpha.

    Returns ``(None, valid)`` if too few frames carry points to be worth attempting, so the
    caller runs the linear-only check rather than a fabricated rotation.
    """
    T = len(obj_pts)
    valid = np.array([p is not None and len(p) >= 20 for p in obj_pts])
    if valid.sum() < 5:
        return None, valid

    perms = []
    for p in itertools.permutations(range(3)):
        for s in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for i, j in enumerate(p):
                M[i, j] = s[i]
            if np.linalg.det(M) > 0:
                perms.append(M)
    perms = np.stack(perms)

    R_out = np.zeros((T, 3, 3))
    prev = np.eye(3)
    for t in range(T):
        if not valid[t]:
            R_out[t] = prev
            continue
        P = obj_pts[t] - obj_pts[t].mean(axis=0)
        _, _, Vt = np.linalg.svd(P, full_matrices=False)
        R = Vt  # rows = principal axes
        if np.linalg.det(R) < 0:
            R[2] *= -1.0
        # Pick the axis relabelling closest to the previous frame (max trace of R_cand prev^T).
        cand = perms @ R
        k = int(np.argmax(np.einsum("kij,ij->k", cand, prev)))
        R_out[t] = cand[k]
        prev = cand[k]

    quat = np.zeros((T, 4))
    for t in range(T):
        R = R_out[t]
        tr = np.trace(R)
        if tr > 0:
            w = np.sqrt(1.0 + tr) / 2.0
            quat[t] = [w, (R[2, 1] - R[1, 2]) / (4 * w),
                       (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)]
        else:
            i = int(np.argmax(np.diag(R)))
            j, k = (i + 1) % 3, (i + 2) % 3
            s = np.sqrt(max(1.0 + R[i, i] - R[j, j] - R[k, k], 1e-12)) * 2.0
            q = np.zeros(4)
            q[0] = (R[k, j] - R[j, k]) / s
            q[1 + i] = 0.25 * s
            q[1 + j] = (R[j, i] + R[i, j]) / s
            q[1 + k] = (R[k, i] + R[i, k]) / s
            quat[t] = q
        quat[t] /= max(np.linalg.norm(quat[t]), 1e-12)
        # Keep the sign continuous: q and -q are the same rotation, but a sign flip
        # between frames looks like a 360-degree jump to any finite difference.
        if t > 0 and quat[t] @ quat[t - 1] < 0:
            quat[t] = -quat[t]
    return quat, valid


@dataclass
class LiftedEpisode:
    """A metric 3D trajectory, plus everything needed to judge whether to trust it."""

    obj_pos: np.ndarray          # (T,3) metres, z along measured vertical
    finger_pos: np.ndarray       # (T,2,3) metres
    fps: float
    support_z: float             # plane height in the same frame
    obj_half: tuple[float, float, float]
    scale_m_per_unit: float
    scale_source: str
    plane_inlier_frac: float
    obj_valid_frac: float
    grip_valid_frac: float
    obj_quat: np.ndarray | None = None
    notes: str = ""

    def to_observation(self):
        from .proxy_scene import Observation
        return Observation(obj_pos=self.obj_pos, finger_pos=self.finger_pos,
                           fps=self.fps, obj_quat=self.obj_quat,
                           scale_source=self.scale_source, notes=self.notes)

    def to_params(self, **kw):
        from .proxy_scene import ProxyParams
        return ProxyParams(obj_half=self.obj_half, support_z=self.support_z, **kw)

    def summary(self) -> dict:
        return {"scale_m_per_unit": self.scale_m_per_unit, "scale_source": self.scale_source,
                "plane_inlier_frac": self.plane_inlier_frac,
                "obj_valid_frac": self.obj_valid_frac, "grip_valid_frac": self.grip_valid_frac,
                "obj_half_m": list(self.obj_half), "support_z": self.support_z,
                "has_orientation": self.obj_quat is not None,
                "fps": self.fps, "n_frames": int(len(self.obj_pos)), "notes": self.notes}


def _gravity_frame(normal: np.ndarray, cam_up_hint: np.ndarray | None = None) -> np.ndarray:
    """Rotation taking the point map into a frame whose +z is the measured vertical.

    The plane normal is sign-ambiguous, so it is oriented to oppose the camera's own
    down direction (+y in image space points down, hence -y is up). Getting this
    backwards would flip gravity and turn every resting object into a violation.
    """
    up = normal / np.linalg.norm(normal)
    hint = cam_up_hint if cam_up_hint is not None else np.array([0.0, -1.0, 0.0])
    if up @ hint < 0:
        up = -up
    a = np.array([1.0, 0.0, 0.0])
    if abs(up @ a) > 0.9:
        a = np.array([0.0, 0.0, 1.0])
    x = a - (a @ up) * up
    x /= np.linalg.norm(x)
    y = np.cross(up, x)
    return np.stack([x, y, up], axis=0)


def _masked_point(pts: np.ndarray, valid: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Median 3D point over a mask; None if the mask is empty or all-invalid.

    Median rather than mean: mask boundaries bleed onto background depth, and a single
    far-background pixel would drag a mean metres away.
    """
    m = mask & valid
    if not m.any():
        return None
    return np.median(pts[m], axis=0)


def _interp_gaps(traj: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Linearly fill frames where the tracker lost the entity.

    Left as NaN they would poison the second difference; filled silently they would
    fake smooth motion. The caller reports ``*_valid_frac`` so the amount of filling
    is always visible next to the residual.
    """
    out = traj.copy()
    idx = np.arange(len(traj))
    if ok.sum() < 2:
        return out
    for c in range(traj.shape[1]):
        out[:, c] = np.interp(idx, idx[ok], traj[ok, c])
    return out


def lift_episode(tracks_npz: str, which: str = "pred", view: int = 0, fps: float = 16.0,
                 device: str = "cuda", stride: int = 1, max_frames: int | None = None,
                 gt_path_len_m: float | None = None, grip_reach_slack: float = 0.05,
                 ref_obj_extent_m: float | None = None,
                 verbose: bool = True) -> LiftedEpisode:
    """Masks + video -> metric 3D trajectory in a gravity-aligned frame.

    ``gt_path_len_m`` is the paired GT end-effector path length in metres; when given it
    fixes the metric scale by matching the tracked gripper's path length. Without it the
    object's apparent size is used as the ruler and ``scale_source`` says so.

    ``grip_reach_slack`` is how far past the object's own size the gripper centroid may
    be while still counted as holding it. It exists because segmentation gives the whole
    arm, not fingertips; ``hold_frac`` is reported so its effect is auditable.
    """
    import cv2
    import torch
    from moge.model.v1 import MoGeModel

    z = np.load(tracks_npz, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    T_all = int(meta["num_frames"])
    hw = tuple(meta["mask_hw"])
    vid = meta["pred_video"] if which == "pred" else meta["gt_video"]

    m_obj = unpack_masks(z, f"v{view}_{which}_object", hw, T_all)
    m_grip = unpack_masks(z, f"v{view}_{which}_gripper", hw, T_all)

    frames = []
    cap = cv2.VideoCapture(vid)
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    T = min(T_all, len(frames))
    keep = list(range(0, T, stride))[:max_frames] if max_frames else list(range(0, T, stride))
    if verbose:
        print(f"[lift] {which}: {len(frames)} video frames, {T_all} mask frames -> "
              f"using {len(keep)} (stride {stride}), masks {hw}")

    model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device).eval()

    obj_raw, grip_raw = [], []
    plane_n, plane_d, plane_inl = [], [], []
    obj_clouds = []
    obj_cloud_seq = []
    for t in keep:
        rgb = frames[t]
        H, W = rgb.shape[:2]
        tt = torch.tensor(rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
        with torch.no_grad():
            out = model.infer(tt)
        pts = out["points"].float().cpu().numpy()
        valid = out["mask"].cpu().numpy().astype(bool)

        mo = cv2.resize(m_obj[t].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        mg = cv2.resize(m_grip[t].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

        # The support surface: everything that is neither object nor gripper, low in
        # the image. Restricting to the lower half keeps walls and ceilings out of the
        # RANSAC, which otherwise happily fits a vertical plane and calls it the floor.
        bg = valid & ~mo & ~mg
        bg[: H // 2] = False
        sub = pts[bg]
        if len(sub) > 4000:
            sub = sub[np.random.default_rng(t).choice(len(sub), 4000, replace=False)]
        nv, dd, inl = fit_plane_ransac(sub)
        plane_n.append(nv); plane_d.append(dd); plane_inl.append(inl)

        obj_raw.append(_masked_point(pts, valid, mo))
        grip_raw.append(_gripper_contact_point(pts, valid, mg, mo))
        cloud_t = None
        if mo.any():
            p = pts[mo & valid]
            if len(p) > 10:
                cloud_t = p[np.random.default_rng(t).choice(
                    len(p), min(len(p), 800), replace=False)]
                obj_clouds.append(cloud_t)
        # Kept per frame (including the Nones) so orientation can be indexed by frame;
        # ``obj_clouds`` drops empty frames and so cannot be differenced in time.
        obj_cloud_seq.append(cloud_t)

    # Orient every plane normal *before* averaging. RANSAC returns an arbitrary sign per
    # frame, so averaging first lets opposite-signed normals cancel -- measured: the GT
    # and pred of the same static episode came out with opposite verticals, which flips
    # gravity and makes a resting object read as levitating.
    hint = np.array([0.0, -1.0, 0.0])
    P = np.stack(plane_n)
    P[(P @ hint) < 0] *= -1.0
    inl = np.array(plane_inl)
    w = inl / max(inl.sum(), 1e-9)
    n_mean = (P * w[:, None]).sum(0)
    n_mean /= np.linalg.norm(n_mean)
    R = _gravity_frame(n_mean, cam_up_hint=hint)
    plane_off = float(np.average(np.array(plane_d), weights=w))
    normal_spread = float(np.median(P @ n_mean))

    ok_o = np.array([p is not None for p in obj_raw])
    ok_g = np.array([p is not None for p in grip_raw])
    fill = np.zeros(3)
    O = np.stack([p if p is not None else fill for p in obj_raw])
    Gp = np.stack([p if p is not None else fill for p in grip_raw])
    O = _interp_gaps(O, ok_o) @ R.T
    Gp = _interp_gaps(Gp, ok_g) @ R.T

    # Object extent measured *in the gravity frame*: rotate the point cloud first, then
    # take per-axis spread. Rotating a size vector instead (|R @ extent|) is not a valid
    # operation on extents and collapsed the object's height to 9 mm.
    if obj_clouds:
        cl = np.concatenate(obj_clouds) @ R.T
        ext_units = np.percentile(cl, 97.5, axis=0) - np.percentile(cl, 2.5, axis=0)
    else:
        ext_units = None

    # Metric scale: match the tracked gripper's 3D path length to the GT end-effector's.
    # One scalar, least-squares by construction, no hand-set constant.
    step = np.linalg.norm(np.diff(Gp, axis=0), axis=1)
    path_units = float(step.sum())
    if ref_obj_extent_m and ext_units is not None and float(ext_units.max()) > 1e-9:
        # Paired mode: the *same physical object* appears in GT and prediction, so its
        # largest true extent is one number shared by both. Anchoring both sides to it is
        # what makes the two grip demands comparable.
        #
        # This is not optional. Grip demand is proportional to the metric scale (measured:
        # 2.52 / 3.80 / 6.35 / 11.44 for s = 0.5 / 1 / 2 / 4), because acceleration carries
        # a factor of scale. Scaling each side independently by its own gripper path length
        # gave scale ratios from 0.20x to 2.35x between GT and pred of the same episode, and
        # the resulting "pred needs 3.6x the grip force" correlated with the scale ratio at
        # **r = +0.943** -- i.e. the headline was mostly a unit mismatch, not physics.
        scale = ref_obj_extent_m / float(ext_units.max())
        src = "paired_object_extent"
    elif gt_path_len_m and path_units > 1e-6:
        scale = gt_path_len_m / path_units
        src = "paired_gt_ee_path_len"
    elif ext_units is not None:
        # Fallback ruler: assume the manipulated object's largest extent is ~8 cm, the
        # median graspable size. Crude, and flagged as such -- but the residual for
        # severe violations is scale-invariant, so this still yields a usable verdict.
        scale = 0.08 / max(float(ext_units.max()), 1e-6)
        src = "object_size_prior_8cm"
    else:
        scale, src = 1.0, "none"
    O *= scale
    Gp *= scale

    half = (tuple(float(max(v * scale / 2.0, 0.005)) for v in ext_units)
            if ext_units is not None else (0.03, 0.03, 0.03))

    # Object orientation, in the same gravity frame. Attempted only when the object is
    # measurably non-symmetric: for a near-cubic point cloud the PCA axes are not
    # determined, and the estimator correctly reports that as unmeasurable rather than
    # inventing a rotation (verified on synthetic clouds -- an asymmetric box recovers
    # omega to 0.0% even with 2 mm of point noise, a cube gets flagged 100%).
    quat = None
    if ext_units is not None:
        srt = np.sort(ext_units)
        aspect = float(srt[2] / max(srt[0], 1e-9))
        if aspect > 1.25:
            quat, _ = object_orientation([c @ R.T * scale if c is not None else None
                                          for c in obj_cloud_seq])

    # Finger placement. The two boxes straddle the *object* centre along the axis
    # perpendicular to the gripper->object direction, and only while the gripper is close
    # enough to plausibly be holding it; otherwise they stay at the gripper centroid.
    #
    # Placing them at the gripper centroid instead (a first version did) fails on real
    # data: the gripper mask covers the whole arm, so its centroid sits far from the
    # fingertips and no contact is ever detected -- measured unsupported_frac = 0.97 with
    # the object clearly held, which reads out as levitation for the entire episode. What
    # the proxy needs is not where the arm is, but whether a grasp is geometrically
    # available, so the contact-bearing bodies are anchored to the object.
    dvec = Gp - O
    dist = np.linalg.norm(dvec, axis=1)
    reach = float(np.median(half)) + grip_reach_slack
    holding = dist < reach
    dhat = np.where(dist[:, None] > 1e-6, dvec / np.maximum(dist[:, None], 1e-9),
                    np.array([1.0, 0.0, 0.0]))
    perp = np.cross(dhat, np.array([0.0, 0.0, 1.0]))
    pn = np.linalg.norm(perp, axis=1, keepdims=True)
    perp = np.where(pn > 1e-6, perp / np.maximum(pn, 1e-9), np.array([0.0, 1.0, 0.0]))
    gap = half[1] + 0.003
    anchor = np.where(holding[:, None], O, Gp)
    F = np.stack([anchor - perp * gap, anchor + perp * gap], axis=1)
    hold_frac = float(holding.mean())

    # Support height, anchored to the object's own lowest observed position rather than
    # projected from the plane equation. Both quantities live in the gravity frame by
    # construction, so this cannot pick up the sign/frame errors that a manual projection
    # did (measured: it placed the surface 0.42 m off, and every frame read as levitating).
    # The plane is still what defines *vertical*; it is only its height that we take from
    # the object, using the assumption that the object rests on the surface at some point
    # in the episode -- true for pick-and-place, and reported so it can be checked.
    obj_bottom = O[:, 2] - half[2]
    support_z = float(np.percentile(obj_bottom, 2.0))
    rest_frac = float(np.mean(obj_bottom - support_z < 0.005))
    lift_range_m = float(obj_bottom.max() - support_z)
    if verbose:
        print(f"[lift] plane inliers {inl.mean():.3f} (coherence {normal_spread:.3f})  "
              f"vertical={np.round(n_mean,3)}  scale={scale:.4f} m/unit ({src})")
        print(f"[lift] obj_half={np.round(half,4)} m  support_z={support_z:.4f}  "
              f"rest_frac={rest_frac:.2f} lift_range={lift_range_m:.4f} m  hold_frac={hold_frac:.2f}  "
              f"obj_valid={ok_o.mean():.2f} grip_valid={ok_g.mean():.2f}")
    return LiftedEpisode(obj_pos=O, finger_pos=F, fps=fps / stride, support_z=support_z,
                         obj_half=half, scale_m_per_unit=scale, scale_source=src,
                         plane_inlier_frac=float(inl.mean()), obj_valid_frac=float(ok_o.mean()),
                         grip_valid_frac=float(ok_g.mean()), obj_quat=quat,
                         notes=(f"{which} lifted from {tracks_npz}; normal_coherence="
                                f"{normal_spread:.3f}; rest_frac={rest_frac:.2f}; "
                                f"lift_range={lift_range_m:.4f} m; hold_frac={hold_frac:.2f}"))
