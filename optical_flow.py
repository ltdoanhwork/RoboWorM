# """Dense optical flow (Farneback) magnitude maps."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def farneback_flow(prev_gray: np.ndarray, gray: np.ndarray, cfg: FlowConfig) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        cfg.pyr_scale,
        cfg.levels,
        cfg.winsize,
        cfg.iterations,
        cfg.poly_n,
        cfg.poly_sigma,
        0,
    )


def flow_magnitude(flow: np.ndarray) -> np.ndarray:
    return np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).astype(np.float32)


def quantize_magnitude(mag: np.ndarray) -> tuple[np.ndarray, float]:
    scale = float(np.max(mag)) if mag.size else 1.0
    if scale <= 1e-8:
        return np.zeros_like(mag, dtype=np.uint8), 1.0
    q = np.clip(mag / scale * 255.0, 0, 255).astype(np.uint8)
    return q, scale


def dequantize_magnitude(q: np.ndarray, scale: float) -> np.ndarray:
    return q.astype(np.float32) / 255.0 * float(scale)


# @dataclass
# class FlowConfig:
#     pyr_scale: float = 0.5
#     levels: int = 3
#     winsize: int = 21  
#     iterations: int = 5  
#     poly_n: int = 7  
#     poly_sigma: float = 1.5
#     downsample: int = 4

#     blur_kernel: int = 5  
#     smooth_alpha: float = (
#         0.3 
#     )
#     clip_percentile: float = (
#         99.0 
#     )


# def farneback_flow(
#     prev_gray: np.ndarray, gray: np.ndarray, cfg: FlowConfig
# ) -> np.ndarray:
#     # Dùng cờ OPTFLOW_FARNEBACK_GAUSSIAN thay cho Box Filter mặc định
#     return cv2.calcOpticalFlowFarneback(
#         prev_gray,
#         gray,
#         None,
#         cfg.pyr_scale,
#         cfg.levels,
#         cfg.winsize,
#         cfg.iterations,
#         cfg.poly_n,
#         cfg.poly_sigma,
#         cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
#     )


# def flow_magnitude(flow: np.ndarray) -> np.ndarray:
#     return np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).astype(np.float32)


# def compute_flow_series(
#     gray_frames: list[np.ndarray],
#     cfg: FlowConfig | None = None,
# ) -> tuple[np.ndarray, np.ndarray, int]:
#     """Frame-to-frame flow magnitude with spatial & temporal smoothing."""
#     cfg = cfg or FlowConfig()
#     if not gray_frames:
#         raise ValueError("gray_frames is empty")

#     stride = max(1, cfg.downsample)
#     h, w = gray_frames[0].shape[:2]
#     hs, ws = (h + stride - 1) // stride, (w + stride - 1) // stride
#     t = len(gray_frames)

#     mag_u8 = np.zeros((t, hs, ws), dtype=np.uint8)
#     scales = np.ones(t, dtype=np.float32)

#     # 1. Pre-process: Lọc nhiễu không gian bằng Gaussian Blur
#     processed_frames = []
#     for f in gray_frames:
#         if cfg.blur_kernel > 0:
#             k = cfg.blur_kernel if cfg.blur_kernel % 2 == 1 else cfg.blur_kernel + 1
#             f_blur = cv2.GaussianBlur(f, (k, k), 0)
#         else:
#             f_blur = f
#         processed_frames.append(f_blur)

#     prev = processed_frames[0]
#     prev_mag_smooth = None
#     running_scale = 1.0

#     for i in range(1, t):
#         curr = processed_frames[i]

#         # Tính flow thô
#         flow = farneback_flow(prev, curr, cfg)
#         mag_raw = flow_magnitude(flow)[::stride, ::stride]

#         # 2. Temporal Smoothing (EMA): Mượt chuyển động theo thời gian
#         if prev_mag_smooth is None:
#             mag_smooth = mag_raw
#         else:
#             mag_smooth = (
#                 cfg.smooth_alpha * mag_raw
#                 + (1.0 - cfg.smooth_alpha) * prev_mag_smooth
#             )
#         prev_mag_smooth = mag_smooth

#         # 3. Mượt Scale: Dùng Percentile 99% để loại bỏ nhiễu điểm + làm mượt EMA cho Scale
#         frame_scale = float(np.percentile(mag_smooth, cfg.clip_percentile))
#         if frame_scale <= 1e-8:
#             frame_scale = 1.0

#         # Làm mượt biến thiên Scale giữa các frame
#         running_scale = 0.2 * frame_scale + 0.8 * running_scale

#         # Quantize về uint8
#         q = np.clip(mag_smooth / running_scale * 255.0, 0, 255).astype(
#             np.uint8
#         )

#         mag_u8[i] = q
#         scales[i] = running_scale
#         prev = curr

#     return mag_u8, scales, stride

@dataclass
class FlowConfig:
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 25  # Tăng lên 25 giúp thuật toán ít nhạy với chi tiết nhỏ
    iterations: int = 5
    poly_n: int = 7
    poly_sigma: float = 1.5
    downsample: int = 4

    # --- BỘ LỌC LÀM MƯỢT ---
    blur_kernel: int = 5
    smooth_alpha: float = 0.3
    clip_percentile: float = 99.0

    # --- CÁC THAM SỐ KHỦ CHUYỂN ĐỘNG NHỎ (MỚI) ---
    min_mag_thresh: float = (
        0.8  # Bỏ qua motion nhỏ hơn X pixel (ví dụ: 0.5 - 1.5)
    )
    morph_kernel_size: int = (
        3  # Xóa các đốm nhiễu hạt nhỏ rời rạc (3x3 hoặc 5x5, 0 = tắt)
    )
    gamma: float = 1.5  # Nén phi tuyến: > 1.0 làm mờ motion nhỏ, giữ rực motion lớn


def farneback_flow(
    prev_gray: np.ndarray, gray: np.ndarray, cfg: FlowConfig
) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        cfg.pyr_scale,
        cfg.levels,
        cfg.winsize,
        cfg.iterations,
        cfg.poly_n,
        cfg.poly_sigma,
        cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    )


def flow_magnitude(flow: np.ndarray) -> np.ndarray:
    return np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).astype(np.float32)


def compute_flow_series(
    gray_frames: list[np.ndarray],
    cfg: FlowConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    cfg = cfg or FlowConfig()
    if not gray_frames:
        raise ValueError("gray_frames is empty")

    stride = max(1, cfg.downsample)
    h, w = gray_frames[0].shape[:2]
    hs, ws = (h + stride - 1) // stride, (w + stride - 1) // stride
    t = len(gray_frames)

    mag_u8 = np.zeros((t, hs, ws), dtype=np.uint8)
    scales = np.ones(t, dtype=np.float32)

    # Pre-process: Blur không gian
    processed_frames = []
    for f in gray_frames:
        if cfg.blur_kernel > 0:
            k = cfg.blur_kernel if cfg.blur_kernel % 2 == 1 else cfg.blur_kernel + 1
            f_blur = cv2.GaussianBlur(f, (k, k), 0)
        else:
            f_blur = f
        processed_frames.append(f_blur)

    prev = processed_frames[0]
    prev_mag_smooth = None
    running_scale = 1.0

    # Kernel cho Morphological Opening
    morph_kernel = None
    if cfg.morph_kernel_size > 0:
        k_s = cfg.morph_kernel_size
        morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (k_s, k_s)
        )

    for i in range(1, t):
        curr = processed_frames[i]

        flow = farneback_flow(prev, curr, cfg)
        mag_raw = flow_magnitude(flow)[::stride, ::stride]

        # 1. TRIỆT TIÊU MOTION NHỎ (Deadzone Thresholding)
        if cfg.min_mag_thresh > 0:
            mag_raw[mag_raw < cfg.min_mag_thresh] = 0.0

        # 2. LỌC HÌNH THÁI HỌC (Morphological Opening - Xóa đốm nhiễu hạt)
        if morph_kernel is not None:
            mag_raw = cv2.morphologyEx(
                mag_raw, cv2.MORPH_OPEN, morph_kernel
            )

        # 3. Làm mượt thời gian (EMA)
        if prev_mag_smooth is None:
            mag_smooth = mag_raw
        else:
            mag_smooth = (
                cfg.smooth_alpha * mag_raw
                + (1.0 - cfg.smooth_alpha) * prev_mag_smooth
            )
        prev_mag_smooth = mag_smooth

        # 4. Tính Scale & Cập nhật
        frame_scale = float(np.percentile(mag_smooth, cfg.clip_percentile))
        if frame_scale <= 1e-8:
            frame_scale = 1.0
        running_scale = 0.2 * frame_scale + 0.8 * running_scale

        norm_mag = np.clip(mag_smooth / running_scale, 0.0, 1.0)
        if cfg.gamma != 1.0:
            norm_mag = np.power(norm_mag, cfg.gamma)

        mag_u8[i] = (norm_mag * 255.0).astype(np.uint8)
        scales[i] = running_scale
        prev = curr

    return mag_u8, scales, stride