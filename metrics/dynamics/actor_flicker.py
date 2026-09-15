"""Experimental flow-aligned subject photometric flicker proxy (RGB [0,1])."""
import cv2
import numpy as np


def aligned_residual(first, second, forward, backward, mask, next_mask):
    """Reject out-of-bounds, occlusion (FB error), and missing subject pixels."""
    h, w = mask.shape
    y, x = np.mgrid[:h, :w].astype(np.float32)
    mx, my = x + forward[..., 0], y + forward[..., 1]
    back = cv2.remap(backward, mx, my, cv2.INTER_LINEAR)
    valid = mask & (mx >= 0) & (mx < w - 1) & (my >= 0) & (my < h - 1)
    valid &= cv2.remap(next_mask.astype(np.uint8), mx, my, cv2.INTER_NEAREST) > 0
    error = ((forward + back) ** 2).sum(-1)
    valid &= error <= 0.01 * ((forward ** 2).sum(-1) + (back ** 2).sum(-1)) + 0.5
    warped = cv2.remap(second.astype(np.float32) / 255, mx, my, cv2.INTER_LINEAR)
    residual = np.abs(first.astype(np.float32) / 255 - warped).mean(-1)
    return (float(residual[valid].mean()) if valid.any() else None,
            int(valid.sum()))
