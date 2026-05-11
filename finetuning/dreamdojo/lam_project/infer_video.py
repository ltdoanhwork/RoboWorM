"""
Sliding-window LAM inference on one video: for each consecutive pair (t_i, t_{i+1}),
predict frame t_{i+1}, then stitch into a full pred sequence for side-by-side compare with GT.

Run from repo root:
  python finetuning/dreamdojo/lam_project/infer_video.py \\
    --config finetuning/dreamdojo/lam_project/config/lam_0.yaml \\
    --video_path path/to/video.mp4 \\
    --output_dir lam_video_infer
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from einops import rearrange
from PIL import Image
from tqdm import tqdm

from lam.dataset import _is_av1, _load_frames_with_av
from lam.model import LAM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LAM inference on full video (sliding 2-frame windows).")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--ckpt_path", type=str, default=None)
    p.add_argument("--video_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="lam_video_infer")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_frames", type=int, default=None, help="Cap number of frames (default: all).")
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--fps", type=float, default=None, help="Output FPS (default: from video or 30).")
    return p.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(config: Dict[str, Any], ckpt_path: str, device: torch.device) -> LAM:
    model_cfg = dict(config["model"])
    model_cfg.pop("optimizer", None)
    model = LAM(**model_cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def _spatial_preprocess(video_thwc: torch.Tensor, output_format: str) -> torch.Tensor:
    """video_thwc: [T,H,W,C] float in [0,1]. Same crop+resize as VideoDataset."""
    target_ratio = 640 / 480
    if video_thwc.shape[2] / video_thwc.shape[1] > target_ratio:
        target_height = video_thwc.shape[1]
        target_width = int(video_thwc.shape[1] * target_ratio)
    elif video_thwc.shape[2] / video_thwc.shape[1] < target_ratio:
        target_height = int(video_thwc.shape[2] / target_ratio)
        target_width = video_thwc.shape[2]
    else:
        target_height = video_thwc.shape[1]
        target_width = video_thwc.shape[2]
    h_crop = (video_thwc.shape[1] - target_height) // 2
    w_crop = (video_thwc.shape[2] - target_width) // 2
    video = video_thwc[:, h_crop : h_crop + target_height, w_crop : w_crop + target_width]
    video = rearrange(video, "t h w c -> c t h w")
    video = F.interpolate(video, (240, 320), mode="bilinear")
    video = rearrange(video, f"c t h w -> {output_format}")
    return video


def _probe_fps(path: str) -> float:
    cap = cv.VideoCapture(path, cv.CAP_FFMPEG, [int(cv.CAP_PROP_HW_ACCELERATION), 0])
    fps_m = cap.get(cv.CAP_PROP_FPS)
    cap.release()
    if fps_m and fps_m > 1e-3:
        return float(fps_m)
    return 30.0


def read_rgb_frames_consecutive(
    video_path: str,
    start_frame: int,
    max_frames: Optional[int],
) -> Tuple[torch.Tensor, float]:
    """Returns float tensor [T,H,W,C] in [0,1], and fps."""
    path = str(Path(video_path).resolve())
    fps = _probe_fps(path)

    frames: List[torch.Tensor] = []

    if _is_av1(path):
        cur = start_frame
        while True:
            remaining = None if max_frames is None else max_frames - len(frames)
            if remaining is not None and remaining <= 0:
                break
            want = 512 if remaining is None else min(512, remaining)
            chunk = _load_frames_with_av(path, want, cur)
            if not chunk:
                break
            frames.extend(chunk)
            cur += len(chunk)
            if len(chunk) < want:
                break
    else:
        cap_need = None if max_frames is None else start_frame + max_frames
        cap = cv.VideoCapture(path, cv.CAP_FFMPEG, [int(cv.CAP_PROP_HW_ACCELERATION), 0])
        if start_frame > 0:
            cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
        while True:
            if cap_need is not None and start_frame + len(frames) >= cap_need:
                break
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame))
        cap.release()

    if not frames:
        raise RuntimeError(f"No frames read from {path}")

    video = torch.stack(frames).float() / 255.0
    return video, fps


def frames_to_uint8_hwc(t: torch.Tensor) -> np.ndarray:
    """t: [H,W,C] float [0,1] -> uint8 RGB"""
    x = t.clamp(0, 1).cpu().numpy()
    return (x * 255.0).round().astype(np.uint8)


def write_png_sequence(frames_rgb: List[np.ndarray], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, fr in enumerate(frames_rgb):
        Image.fromarray(fr).save(out_dir / f"frame_{i:06d}.png")


def ffmpeg_encode_png_dir(png_dir: Path, pattern: str, out_mp4: Path, fps: float) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(png_dir / pattern),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_mp4),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise RuntimeError(f"ffmpeg failed: {r.stderr[-500:]}")


def make_side_by_side(left: List[np.ndarray], right: List[np.ndarray]) -> List[np.ndarray]:
    out = []
    for a, b in zip(left, right):
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        aa = a[:h, :w]
        bb = b[:h, :w]
        out.append(np.concatenate([aa, bb], axis=1))
    return out


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ckpt_path = args.ckpt_path or config.get("ckpt_path")
    if not ckpt_path:
        raise ValueError("Need --ckpt_path or ckpt_path in config.")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = build_model(config, ckpt_path, device)

    data_cfg = config["data"]
    output_format = data_cfg.get("output_format", "t h w c")

    raw_thwc, fps_src = read_rgb_frames_consecutive(args.video_path, args.start_frame, args.max_frames)
    video = _spatial_preprocess(raw_thwc, output_format)
    if video.shape[0] < 2:
        raise ValueError("Need at least 2 frames after preprocess.")

    fps = args.fps if args.fps is not None else fps_src

    T = video.shape[0]
    pred_frames = torch.zeros_like(video)
    pred_frames[0] = video[0]

    for i in tqdm(range(T - 1), desc="Sliding inference"):
        pair = video[i : i + 2].unsqueeze(0).to(device)
        out = model.lam({"videos": pair})
        recon = out["recon"]
        pred_frames[i + 1] = recon[0, 0].cpu()

    gt_rgb = [frames_to_uint8_hwc(video[t]) for t in range(T)]
    pr_rgb = [frames_to_uint8_hwc(pred_frames[t]) for t in range(T)]
    cmp_rgb = make_side_by_side(gt_rgb, pr_rgb)

    out_root = Path(args.output_dir)
    tmp = out_root / "_frames_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    gt_dir = tmp / "gt"
    pr_dir = tmp / "pred"
    cp_dir = tmp / "compare"
    write_png_sequence(gt_rgb, gt_dir)
    write_png_sequence(pr_rgb, pr_dir)
    write_png_sequence(cmp_rgb, cp_dir)

    out_root.mkdir(parents=True, exist_ok=True)
    ffmpeg_encode_png_dir(gt_dir, "frame_%06d.png", out_root / "gt.mp4", fps)
    ffmpeg_encode_png_dir(pr_dir, "frame_%06d.png", out_root / "pred.mp4", fps)
    ffmpeg_encode_png_dir(cp_dir, "frame_%06d.png", out_root / "compare_side_by_side.mp4", fps)

    shutil.rmtree(tmp)

    print(f"Wrote:\n  {out_root / 'gt.mp4'}\n  {out_root / 'pred.mp4'}\n  {out_root / 'compare_side_by_side.mp4'}")
    print(f"Frames: {T}, FPS: {fps}")


if __name__ == "__main__":
    main()
