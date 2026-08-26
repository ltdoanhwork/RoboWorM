"""Apply Farneback flow heatmap onto RGB frames (no keypoint seeds required).

Ported from thanhnv/human-assessment overlay.py + optical_flow.py so we can
run on decoded mp4 frames instead of the original view1/view2 PNG folders.
"""
from __future__ import annotations

import cv2
import numpy as np

from .optical_flow import FlowConfig, compute_flow_series, dequantize_magnitude
from .overlay import blend, flow_heatmap


def overlay_flow_rgb(frames_rgb: list[np.ndarray], cfg: FlowConfig | None = None) -> list[np.ndarray]:
    """Return copies of `frames_rgb` with a TURBO flow-magnitude heatmap blended in."""
    if not frames_rgb:
        return []
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames_rgb]
    mag_u8, scales, _stride = compute_flow_series(gray, cfg)
    out = []
    for t, f in enumerate(frames_rgb):
        h, w = f.shape[:2]
        mag = dequantize_magnitude(mag_u8[t], float(scales[t]))
        heat, alpha = flow_heatmap(mag, (h, w))
        bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        blended = blend(bgr, heat, alpha)
        rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        cv2.putText(rgb, "flow", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        out.append(rgb)
    return out
