"""Experimental flow-aligned subject photometric flicker proxy (RGB [0,1]).

WHY THERE ARE SEVERAL VARIANTS (E3)
-----------------------------------
The v1 statistic (`raw` below) is the mean flow-aligned photometric residual over
subject pixels. Measured on the completed v3.2 run it is badly confounded with motion
amount -- `rho(actor_motion_raw, actor_flicker_raw)` = +0.695 static / +0.473 head /
+0.364 wrist, against a <0.25 bar -- because a residual after warping grows with
displacement for reasons that are NOT flicker: flow error scales with magnitude,
bilinear resampling blurs, and fast subjects are motion-blurred in the source.

Two "obvious" fixes were tested on the stored aggregates and both fail:

* dividing the clip aggregate by the clip's motion statistic OVERSHOOTS, flipping
  rho to -0.78 / -0.79 / -0.78 (it removes real signal along with the confound);
* regressing log(flicker) on log(motion) at PAIR level and keeping the residual --
  the strongest correction available post-hoc -- fixes head (-0.071) and static
  (+0.061) but leaves wrist at -0.257, still outside the bar.

The confound is almost purely multiplicative (log-log slope 0.19-0.27), and it acts
*within* a frame: fast pixels carry more warp error than slow pixels in the same
frame. No transform of a clip-level or pair-level aggregate can undo that, so the
normalization has to happen PER PIXEL, before averaging -- which is what these
variants do. They are all computed in ONE pass (the flow and masks are what cost
GPU time; these are a few numpy ops on arrays that already exist) so a single
re-scoring run can decide between them instead of one run per candidate.
"""
import cv2
import numpy as np

GRAD_FLOOR = 0.02      # [0,1]/px; below this a pixel is flat and r/g is meaningless
COPY_FLOOR = 0.01      # [0,1]; ~2.5/255, the codec-noise floor on |I1 - I2|
FLOW_SCALE = 1.0       # px; r/(1 + m/FLOW_SCALE), the literal "local flow" normalizer
# Subtractive correction strengths. A sub-pixel misalignment of e px on a pixel with
# local gradient g produces a residual of about g*e, and a flow field is wrong by
# roughly a fixed FRACTION of the displacement it reports, so e ~ K*m and the part of
# the residual explainable by misalignment is ~ K*g*m. Subtracting that (floored at 0)
# is better founded than dividing, and unlike a division it cannot invert the
# correlation the way dividing by the clip's motion aggregate did (rho +0.70 -> -0.79).
# K is not known a priori and is probably mount-dependent, so the statistic is stored
# for a GRID of K: that makes the choice a post-hoc, zero-GPU decision, exactly as the
# E4 gate change was made reconstructible from the record instead of re-scored.
SUBTRACT_K = (0.05, 0.1, 0.2, 0.4)


def _valid_and_warp(first, second, forward, backward, mask, next_mask):
    """Shared geometry: reject out-of-bounds, occlusion (FB error), missing subject."""
    h, w = mask.shape
    if first.shape[:2] != (h, w):                      # flow/mask grid may differ
        first = cv2.resize(first, (w, h), interpolation=cv2.INTER_AREA)
        second = cv2.resize(second, (w, h), interpolation=cv2.INTER_AREA)
    y, x = np.mgrid[:h, :w].astype(np.float32)
    mx, my = x + forward[..., 0], y + forward[..., 1]
    back = cv2.remap(backward, mx, my, cv2.INTER_LINEAR)
    valid = mask & (mx >= 0) & (mx < w - 1) & (my >= 0) & (my < h - 1)
    valid &= cv2.remap(next_mask.astype(np.uint8), mx, my, cv2.INTER_NEAREST) > 0
    error = ((forward + back) ** 2).sum(-1)
    valid &= error <= 0.01 * ((forward ** 2).sum(-1) + (back ** 2).sum(-1)) + 0.5
    f1 = first.astype(np.float32) / 255
    f2 = second.astype(np.float32) / 255
    warped = cv2.remap(f2, mx, my, cv2.INTER_LINEAR)
    residual = np.abs(f1 - warped).mean(-1)
    return valid, residual, f1, f2, mx, my


def aligned_residual(first, second, forward, backward, mask, next_mask):
    """v1 statistic, unchanged: mean flow-aligned residual over valid subject pixels."""
    valid, residual, *_ = _valid_and_warp(first, second, forward, backward, mask, next_mask)
    return (float(residual[valid].mean()) if valid.any() else None,
            int(valid.sum()))


def residual_variants(first, second, forward, backward, mask, next_mask):
    """v1 plus four per-pixel motion-normalized candidates. One pass, all cheap.

    Returned keys (all None when no pixel survives the validity geometry):

    ``raw``
        v1. mean |I1 - warp(I2)|. Kept so the new run reproduces the old column.
    ``flownorm``
        mean of r / (1 + m) with m the LOCAL flow magnitude at that pixel. This is
        the literal per-pixel version of the correction that overshot at clip level;
        per pixel it only discounts the pixels that actually moved.
    ``gradnorm``
        mean of r / (|grad I1| + GRAD_FLOOR). A sub-pixel misalignment produces a
        residual of roughly |grad I| * (alignment error), so dividing by the local
        gradient removes the part of the residual that structure + misalignment can
        explain, while a true brightness flicker (gradient-free) survives.
    ``unexplained``
        sum(r) / sum(max(c, COPY_FLOOR)) with c = |I1 - I2| BEFORE warping: the
        fraction of the observed temporal change that flow cannot explain. Motion
        enters numerator and denominator together, so it should cancel structurally.
        This mirrors ActiveMotionContinuity's MSE_VFI/MSE_copy, which is the one
        motion-normalized statistic in this repo that already survives audit. Pooled
        as a ratio of sums (not a mean of ratios) so near-static pixels cannot blow up.
    ``shiftmin``
        mean over pixels of the smallest residual reachable by nudging the warp
        target within +-1 px. Any residual removable by a 1-px correction is
        alignment error by construction, so what is left is genuinely photometric.
        Strictly <= raw; the gap raw - shiftmin is itself an alignment-error estimate.
    ``sub_<K>`` (one per `SUBTRACT_K`)
        mean of max(0, r - K * g * m): the residual minus the part that a local
        misalignment of K*m px on a gradient of g would explain. Subtractive rather
        than divisive, so it cannot flip the sign of the correlation. Stored for a
        grid of K so the strength can be chosen post-hoc without re-scoring.

    Diagnostics also returned: ``copy`` (mean c), ``flow`` (mean m), ``grad``
    (mean |grad I1|) and ``align_pred`` (mean g*m, the misalignment-explainable scale)
    over the same valid pixels, so the confound can be re-analysed from the record
    without re-running the probe.
    """
    valid, residual, f1, f2, mx, my = _valid_and_warp(
        first, second, forward, backward, mask, next_mask)
    n = int(valid.sum())
    keys = ('raw', 'flownorm', 'gradnorm', 'unexplained', 'shiftmin',
            'copy', 'flow', 'grad', 'align_pred') + tuple(f'sub_{k:g}' for k in SUBTRACT_K)
    if not n:
        return {k: None for k in keys}, 0

    r = residual[valid]
    m = np.sqrt((forward ** 2).sum(-1))[valid]
    copy = np.abs(f1 - f2).mean(-1)[valid]
    gray = f1.mean(-1)
    grad = np.sqrt(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) ** 2 +
                   cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) ** 2)[valid] / 8.0

    # +-1 px search around the warp target. Nine remaps of a 256x192 frame is noise
    # next to the optical-flow forward pass this function is called after.
    best = residual
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            shifted = cv2.remap(f2, mx + dx, my + dy, cv2.INTER_LINEAR)
            best = np.minimum(best, np.abs(f1 - shifted).mean(-1))

    align = grad * m                                   # explainable-by-misalignment scale
    out = {'raw': float(r.mean()),
           'flownorm': float((r / (1.0 + m / FLOW_SCALE)).mean()),
           'gradnorm': float((r / (grad + GRAD_FLOOR)).mean()),
           'unexplained': float(r.sum() / np.maximum(copy, COPY_FLOOR).sum()),
           'shiftmin': float(best[valid].mean()),
           'copy': float(copy.mean()),
           'flow': float(m.mean()),
           'grad': float(grad.mean()),
           'align_pred': float(align.mean())}
    for k in SUBTRACT_K:
        out[f'sub_{k:g}'] = float(np.maximum(r - k * align, 0.0).mean())
    return out, n
