"""
Roll-out (autoregressive) full-episode inference for DreamGen bimanual MULTIVIEW
(2x2 grid). Mirrors the single-view full-episode logic: instead of generating a
single ~5s clip from the first frame, we chain multiple generated segments so the
predicted video length is close to the ground-truth episode length.

Pipeline (Video2WorldPipeline, image/video conditioning, NO actions):
  segment 0 : condition on the GT first-frame grid (image)  -> generates T frames
  segment k : condition on the last frames of segment k-1 (video) -> T frames
  ... repeat until accumulated length >= GT length, then trim to GT length.

For each episode we also build the full GT grid video (every frame, 4-view 2x2)
to (a) know the true length and (b) compute PSNR/SSIM/LPIPS against the prediction.

Outputs per episode (canonical sampling_dataset layout, {model}=dreamgen):
  <save_root>/<category>/<iter>/<task>/episode_XXXXXX/
      full_pred.mp4   full_gt.mp4   full_merged.mp4   info.json
      seg_000.mp4 ...   (individual rolled-out segments)

This is a SINGLE-process script (model loads once); pass --num_gpus for context
parallel via torchrun, matching examples/video2world.py conventions.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models" / "dreamgen"))

QUAD_W, QUAD_H = 384, 216  # per-quadrant; full grid = 768x432 (matches training/prepare)

# bimanual 4-view grid layout (matches prepare_bimanual_multiview_dreamgen_batch.py --four_views)
CAM_LAYOUT = [
    "observation.images.cam_high",
    "observation.images.cam_low",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
PROMPT_4VIEW = (
    "A multi-view video of a bimanual ALOHA robot performing: {task}. "
    "The video shows four synchronized camera views arranged in a 2x2 grid: "
    "top-left shows the high overhead view, top-right shows the low front view, "
    "bottom-left shows the left wrist view, and bottom-right shows the right wrist view."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-path", required=True,
                   help="Comma-separated LeRobot task dirs (e.g. datasets/bimanual/multiview/makovian/close_toolbox).")
    p.add_argument("--save-root", required=True,
                   help="Root for outputs, e.g. sampling_dataset/dense/bimanual/output/multiview/dreamgen")
    p.add_argument("--category", default="",
                   help="makovian|non_makovian; inferred from each task path if empty.")
    p.add_argument("--dit-path", required=True, help="Path to fine-tuned DiT checkpoint .pt")
    p.add_argument("--iter-name", default="", help="Override iter dir name; else derived from --dit-path stem.")
    p.add_argument("--model-size", default="2B")
    p.add_argument("--resolution", default="480", type=str)
    p.add_argument("--fps", default=16, type=int)
    p.add_argument("--guidance", default=7.0, type=float)
    p.add_argument("--num-gpus", default=1, type=int)
    p.add_argument("--num-conditional-frames", default=1, type=int, choices=[1, 5],
                   help="Frames to condition each rolled-out segment after the first. "
                        "1 => 1 latent frame (matches image-conditioning, consistent across segments); "
                        "5 => 2 latent frames (also in train distribution, smoother but mixes with seg0).")
    p.add_argument("--max-episodes", default=None, type=int)
    p.add_argument("--max-segments", default=None, type=int,
                   help="Hard cap on rolled-out segments per episode (safety). Default: derived from GT length.")
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--save-fps", default=16, type=int)
    p.add_argument("--disable-guardrail", action="store_true", default=True)
    return p.parse_args()


def infer_category(path: Path) -> str:
    for part in path.parts:
        if part in ("makovian", "non_makovian"):
            return part
    return ""


# --------------------------------------------------------------------------- #
# GT grid video construction (all frames, 4-view 2x2)                          #
# --------------------------------------------------------------------------- #
def find_video(dataset_path: Path, cam_key: str, ep_idx: int) -> Path | None:
    videos_root = dataset_path / "videos"
    if not videos_root.exists():
        return None
    for chunk_dir in sorted(videos_root.iterdir()):
        if not chunk_dir.is_dir():
            continue
        cand = chunk_dir / cam_key / f"episode_{ep_idx:06d}.mp4"
        if cand.exists():
            return cand
    return None


def build_gt_grid_video(dataset_path: Path, ep_idx: int, out_mp4: Path, fps: int) -> bool:
    """Stitch the 4 camera mp4s into a full 2x2 grid video (every frame)."""
    inputs: list[str] = []
    scale_filters: list[str] = []
    for idx, cam_key in enumerate(CAM_LAYOUT):
        vp = find_video(dataset_path, cam_key, ep_idx)
        if vp is None:
            return False
        inputs += ["-i", str(vp)]
        scale_filters.append(f"[{idx}:v]scale={QUAD_W}:{QUAD_H}[v{idx}]")
    filter_complex = (
        ";".join(scale_filters)
        + ";[v0][v1]hstack[top];[v2][v3]hstack[bottom];[top][bottom]vstack[out]"
    )
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = (
        ["ffmpeg", "-nostdin", "-y"]
        + inputs
        + ["-filter_complex", filter_complex, "-map", "[out]",
           "-r", str(fps), "-pix_fmt", "yuv420p", str(out_mp4)]
    )
    res = subprocess.run(cmd, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return res.returncode == 0 and out_mp4.exists()


def read_episodes_jsonl(meta_dir: Path) -> dict[int, str]:
    ep_file = meta_dir / "episodes.jsonl"
    tasks_file = meta_dir / "tasks.jsonl"
    task_map: dict[int, str] = {}
    if tasks_file.exists():
        for line in tasks_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tidx = obj.get("task_index", obj.get("index"))
            task = obj.get("task", obj.get("description", ""))
            if tidx is not None and task:
                task_map[int(tidx)] = task
    episodes: dict[int, str] = {}
    if not ep_file.exists():
        return episodes
    for line in ep_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        ep_idx = obj.get("episode_index")
        if ep_idx is None:
            continue
        tasks = obj.get("tasks", [])
        if tasks:
            episodes[int(ep_idx)] = tasks[0]
        else:
            tidx = obj.get("task_index")
            episodes[int(ep_idx)] = task_map.get(int(tidx), "performing a manipulation task") if tidx is not None else "performing a manipulation task"
    return episodes


# --------------------------------------------------------------------------- #
# Video IO helpers                                                              #
# --------------------------------------------------------------------------- #
def load_video_frames(path: Path) -> np.ndarray:
    """Return (T, H, W, C) uint8."""
    import mediapy
    return np.asarray(mediapy.read_video(str(path)))


def write_video_frames(path: Path, frames: np.ndarray, fps: int) -> None:
    import mediapy
    path.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(path), frames, fps=fps)


# --------------------------------------------------------------------------- #
# Roll-out                                                                      #
# --------------------------------------------------------------------------- #
def rollout_episode(pipe, first_frame_img: Path, prompt: str, neg_prompt: str,
                    args, gt_len: int, tmp_dir: Path) -> list[np.ndarray]:
    """Generate segments autoregressively until pred length >= gt_len."""
    segments: list[np.ndarray] = []
    total = 0
    seg_idx = 0
    # First segment: condition on the single GT first frame (image input).
    cond_input = str(first_frame_img)
    cond_frames = 1

    # Derive max segments from GT length if not capped.
    # We don't know per-seg frame count up front; cap generously and break on length.
    hard_cap = args.max_segments if args.max_segments else 64

    while total < gt_len and seg_idx < hard_cap:
        video, _ = pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            aspect_ratio="16:9",
            input_path=cond_input,
            num_conditional_frames=cond_frames,
            guidance=args.guidance,
            seed=args.seed + seg_idx,
            return_prompt=True,
        )
        if video is None:
            break
        # video: tensor in [-1, 1], shape (C, T, H, W) or (1, C, T, H, W)
        import torch
        v = video[0] if video.dim() == 5 else video
        frames = ((torch.clamp((v + 1) / 2, 0, 1) * 255)
                  .to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy())  # (T,H,W,C)

        if seg_idx > 0:
            # drop the conditioning overlap frames we re-fed, keep the new content
            keep_from = min(cond_frames, frames.shape[0] - 1)
            frames_new = frames[keep_from:]
        else:
            frames_new = frames

        segments.append(frames_new)
        total += frames_new.shape[0]
        seg_idx += 1

        if total >= gt_len:
            break

        # Prepare conditioning for next segment from the tail of this segment.
        # num_conditional_frames==1 -> feed a single last frame as an image
        # (matches seg-0 image conditioning, consistent across all segments).
        # num_conditional_frames==5 -> feed last frames as a short video.
        import mediapy
        if args.num_conditional_frames == 1:
            cond_png = tmp_dir / f"_cond_{seg_idx:03d}.png"
            mediapy.write_image(str(cond_png), frames[-1])
            cond_input = str(cond_png)
            cond_frames = 1
        else:
            cond_mp4 = tmp_dir / f"_cond_{seg_idx:03d}.mp4"
            tail = frames[-max(args.num_conditional_frames, 1):]
            write_video_frames(cond_mp4, tail, args.save_fps)
            cond_input = str(cond_mp4)
            cond_frames = args.num_conditional_frames

    return segments


def main() -> None:
    args = parse_args()

    # ---- Build the pipeline once (reuse examples/video2world setup) ----
    from examples.video2world import setup_pipeline

    class _NS:
        pass

    pipe_args = _NS()
    pipe_args.model_size = args.model_size
    pipe_args.resolution = args.resolution
    pipe_args.fps = args.fps
    pipe_args.dit_path = args.dit_path
    pipe_args.load_ema = False
    pipe_args.aspect_ratio = "16:9"
    pipe_args.num_gpus = args.num_gpus
    pipe_args.seed = args.seed
    pipe_args.disable_guardrail = args.disable_guardrail
    pipe_args.offload_guardrail = False
    pipe_args.disable_prompt_refiner = True
    pipe_args.offload_prompt_refiner = False
    pipe_args.offload_text_encoder = False
    pipe_args.downcast_text_encoder = False
    pipe_args.save_path = str(Path(args.save_root) / "_pipe_init.mp4")
    pipe_args.benchmark = False
    pipe_args.use_cuda_graphs = False
    pipe_args.natten = False

    print("Building Video2World pipeline (loads model once)...")
    pipe = setup_pipeline(pipe_args)

    neg_prompt = (
        "The video captures a scene with low visual quality, blurring, jittering, "
        "distortion, static frames, or unnatural transitions."
    )

    iter_name = args.iter_name or Path(args.dit_path).stem  # e.g. iter_000010000
    task_paths = [Path(p.strip()) for p in args.dataset_path.split(",") if p.strip()]

    ep_done = 0

    for task_dir in task_paths:
        if not task_dir.exists():
            print(f"[WARN] task dir missing: {task_dir}")
            continue
        category = args.category.strip() or infer_category(task_dir)
        task_name = task_dir.name
        episodes_meta = read_episodes_jsonl(task_dir / "meta")
        if not episodes_meta:
            print(f"[WARN] no episodes in {task_dir}")
            continue

        ep_indices = sorted(episodes_meta.keys())
        save_task_dir = Path(args.save_root) / category / iter_name / task_name

        for ep_idx in ep_indices:
            if args.max_episodes is not None and ep_done >= args.max_episodes:
                break
            ep_dir = save_task_dir / f"episode_{ep_idx:06d}"
            if (ep_dir / "full_pred.mp4").exists():
                print(f"[SKIP] {category}/{task_name}/episode_{ep_idx:06d} (done)")
                ep_done += 1
                continue

            task_clean = episodes_meta[ep_idx].rstrip(".,;").strip()
            prompt = PROMPT_4VIEW.format(task=task_clean)

            with tempfile.TemporaryDirectory() as td:
                tmp_dir = Path(td)
                gt_mp4 = tmp_dir / "gt_grid.mp4"
                if not build_gt_grid_video(task_dir, ep_idx, gt_mp4, args.save_fps):
                    print(f"[WARN] could not build GT grid for episode_{ep_idx:06d}, skipping")
                    continue
                gt_frames = load_video_frames(gt_mp4)
                gt_len = gt_frames.shape[0]

                # First-frame image used to condition segment 0.
                first_png = tmp_dir / "first.png"
                import mediapy
                mediapy.write_image(str(first_png), gt_frames[0])

                print(f"[RUN] {category}/{task_name}/episode_{ep_idx:06d} "
                      f"gt_len={gt_len} ...")
                segments = rollout_episode(
                    pipe, first_png, prompt, neg_prompt, args, gt_len, tmp_dir
                )
                if not segments:
                    print("   -> no segments produced, skipping")
                    continue

                pred = np.concatenate(segments, axis=0)
                # Match resolution to GT (pred may be model-native; resize GT to pred grid).
                ph, pw = pred.shape[1:3]
                if gt_frames.shape[1:3] != (ph, pw):
                    import mediapy
                    gt_frames = np.stack([
                        np.asarray(mediapy.resize_image(f, (ph, pw))) for f in gt_frames
                    ])
                n = min(len(pred), gt_len)
                pred = pred[:n]
                gt_cut = gt_frames[:n]

                ep_dir.mkdir(parents=True, exist_ok=True)
                for si, seg in enumerate(segments):
                    write_video_frames(ep_dir / f"seg_{si:03d}.mp4", seg, args.save_fps)
                write_video_frames(ep_dir / "full_pred.mp4", pred, args.save_fps)
                write_video_frames(ep_dir / "full_gt.mp4", gt_cut, args.save_fps)
                merged = np.concatenate([gt_cut, pred], axis=2)
                write_video_frames(ep_dir / "full_merged.mp4", merged, args.save_fps)

                json.dump(
                    {
                        "num_segments": len(segments),
                        "pred_frames": int(len(pred)),
                        "gt_frames": int(gt_len),
                        "episode_index": int(ep_idx),
                        "prompt": prompt,
                    },
                    open(ep_dir / "info.json", "w"), indent=2, ensure_ascii=False,
                )
                ep_done += 1
                print(f"   -> {len(pred)} frames in {len(segments)} segments")

    print(f"\n=== Done: {ep_done} episodes ===")

    # Cleanup distributed if multi-GPU
    try:
        from examples.video2world import cleanup_distributed
        cleanup_distributed()
    except Exception:
        pass


if __name__ == "__main__":
    main()
