"""Lift 2D masks to (pseudo-)3D for the sim2real interaction predicates.

Three backends, selected by ``--depth``:

``none`` (DEFAULT)
    Pure image plane. Every distance is divided by ``sqrt(area(object mask))`` so
    it is expressed in "object diameters" and is therefore resolution- and
    object-size-independent. This is the default on purpose: RoboWM-Bench's own
    appendix F measured that estimated depth (absolute *and* first-frame-aligned
    relative) stays far enough from ground truth that it did not improve their
    hand tracking. Our substitute for depth is the AND-gate across the two static
    exterior cameras: a contact that is only apparent in one view is rejected as
    an occlusion artifact.

``dav2``
    Depth-Anything-V2-Small via ``transformers.DepthAnythingForDepthEstimation``.
    Needs no new dependency (transformers 4.51.3 in ``models/dreamgen/.venv``
    already exposes the class). Output is *relative inverse* depth, so it is only
    used for ordinal tests (which of object/gripper is nearer) and, after
    per-frame robust normalisation, for a coarse z separation.

``moge``
    MoGe ViT-L point maps (affine-invariant metric-ish 3D + FOV). The checkpoint is
    already cached in the workspace HF cache, but the package is not installed;
    install with ``--no-deps`` (see metrics/quality/README.md for why that flag is
    mandatory here). Optional refinement only.

Whatever the backend, the metric layer receives the same interface: per-frame
scalar summaries of an entity (centroid, extent, and optionally z) plus a
``gap`` between two entities.
"""

import numpy as np


# --------------------------------------------------------------------------- #
# geometry helpers (backend-independent, pure numpy/opencv)
# --------------------------------------------------------------------------- #
def mask_centroid(mask):
    """(y, x) centroid of a boolean mask, or None when empty."""
    idx = np.flatnonzero(mask.ravel())
    if idx.size == 0:
        return None
    W = mask.shape[1]
    ys, xs = np.divmod(idx, W)
    return float(ys.mean()), float(xs.mean())


def mask_scale(mask):
    """``sqrt(area)`` in pixels - the natural length unit of the object itself."""
    a = float(mask.sum())
    return float(np.sqrt(a)) if a > 0 else 0.0


def mask_extent(mask):
    """(h, w) of the mask's tight bounding box, 0 when empty."""
    if not mask.any():
        return 0.0, 0.0
    ys, xs = np.nonzero(mask)
    return float(ys.max() - ys.min() + 1), float(xs.max() - xs.min() + 1)


class NoDepthLift:
    """Image-plane backend: distances in object-diameter units, no z."""

    name = "none"
    has_depth = False

    def frame_depth(self, frame_rgb):  # noqa: D401 - uniform interface
        return None

    def entity_z(self, mask, depth):
        return None


class DepthAnythingLift:
    """Depth-Anything-V2 relative inverse depth (no extra dependency needed)."""

    name = "dav2"
    has_depth = True

    def __init__(self, model_id="depth-anything/Depth-Anything-V2-Small-hf",
                 device="cuda"):
        import torch
        from transformers import AutoImageProcessor, DepthAnythingForDepthEstimation

        self._torch = torch
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.model = DepthAnythingForDepthEstimation.from_pretrained(model_id)
        self.model.eval().to(device)
        self.device = device

    def frame_depth(self, frame_rgb):
        """Return a (H, W) float array of *relative inverse* depth (larger = nearer)."""
        t = self.proc(images=frame_rgb, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            out = self.model(**t).predicted_depth
        out = self._torch.nn.functional.interpolate(
            out[:, None], size=frame_rgb.shape[:2], mode="bicubic", align_corners=False)
        return out[0, 0].float().cpu().numpy()

    def entity_z(self, mask, depth):
        """Median inverse depth over the mask (larger = nearer to the camera)."""
        if depth is None or not mask.any():
            return None
        m = mask
        if m.shape != depth.shape:
            import cv2
            m = cv2.resize(m.astype(np.uint8), (depth.shape[1], depth.shape[0]),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
            if not m.any():
                return None
        return float(np.median(depth[m]))


class MoGeLift:
    """MoGe ViT-L point map backend (optional; needs `pip install --no-deps moge`)."""

    name = "moge"
    has_depth = True

    def __init__(self, model_id="Ruicheng/moge-vitl", device="cuda"):
        import torch
        from moge.model.v1 import MoGeModel

        self._torch = torch
        self.model = MoGeModel.from_pretrained(model_id).to(device).eval()
        self.device = device

    def frame_depth(self, frame_rgb):
        t = self._torch.tensor(frame_rgb / 255.0, dtype=self._torch.float32,
                               device=self.device).permute(2, 0, 1)
        with self._torch.no_grad():
            out = self.model.infer(t)
        # MoGe returns metric-ish depth (larger = farther); invert so the
        # convention matches dav2 (larger = nearer) for the ordinal tests
        d = out["depth"].float().cpu().numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.where(d > 0, 1.0 / d, 0.0)
        return inv

    entity_z = DepthAnythingLift.entity_z


def build_lift(kind="none", device="cuda"):
    if kind in (None, "none"):
        return NoDepthLift()
    if kind == "dav2":
        return DepthAnythingLift(device=device)
    if kind == "moge":
        return MoGeLift(device=device)
    raise ValueError(f"unknown depth backend {kind!r} (expected none|dav2|moge)")
