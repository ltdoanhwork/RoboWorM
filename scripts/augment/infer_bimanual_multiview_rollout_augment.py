"""
AUGMENT copy of scripts/infer_bimanual_multiview_rollout.py — do NOT use for the
dense/acceleration benchmark (that one must stay untouched so runs stay comparable).

Adds the three augment branches for DreamGen multiview:
  --manifest/--branch    select only the (task, episode) pairs from augment_manifest.json
                         and override the prompt with the manifest's text
  --edited-frames-root   condition segment 0 on a recolored grid first frame
                         (edit_only / combo branches)
  --prompt-refine        run DreamGen's prompt refiner (prompt_only / combo branches)
  --no-gt                augmented scenes have no ground truth: skip full_gt.mp4 /
                         full_merged.mp4 and all reference metrics
Episode dirs get a `__{branch}` suffix so they stay diffable against dense.
See .cursor/rules/augment-track-layout.mdc.

Roll-out (autoregressive) full-episode inference for DreamGen bimanual MULTIVIEW
(2x2 grid). Mirrors the single-view full-episode logic: instead of generating a
single ~5s clip from the first frame, we chain multiple generated segments so the
predicted video length is close to the ground-truth episode length.

Pipeline (Video2WorldPipeline, image/video conditioning, NO actions):
  segment 0 : condition on the (edited) first-frame grid (image) -> generates T frames
  segment k : condition on the last frames of segment k-1 (video) -> T frames
  ... repeat until accumulated length >= GT length, then trim to GT length.

The GT grid video is still built per episode to know the true target length (and,
unless --no-gt, to compute reference metrics).

Outputs per episode (canonical sampling_dataset layout, {model}=dreamgen):
  <save_root>/<category>/<iter>/<task>/episode_XXXXXX__<branch>/
      full_pred.mp4   metrics.json   augment_info.json
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "models" / "dreamgen"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import augment_paths as AP  # noqa: E402  (augment leaf/path naming)
import augment_prompt as PR  # noqa: E402  (branch prompt composition)

QUAD_W, QUAD_H = 384, 216  # per-quadrant; full grid = 768x432 (matches training/prepare)

# --------------------------------------------------------------------------- #
# 4-view grid layouts. The grid slot order is [TL, TR, BL, BR] and MUST match
# the corresponding prepare script so conditioning + GT grids line up with what
# the checkpoint was trained on.
# --------------------------------------------------------------------------- #
# bimanual ALOHA (matches prepare_bimanual_multiview_dreamgen_batch.py --four_views)
LAYOUT_BIMANUAL = {
    "cams": [
        "observation.images.cam_high",
        "observation.images.cam_low",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ],
    "prompt": (
        "A multi-view video of a bimanual ALOHA robot performing: {task}. "
        "The video shows four synchronized camera views arranged in a 2x2 grid: "
        "top-left shows the high overhead view, top-right shows the low front view, "
        "bottom-left shows the left wrist view, and bottom-right shows the right wrist view."
    ),
}
# humanoid Airbot MMK2 LAYOUT_B (matches prepare_multiview_humanoid_grid_batch.py LAYOUT_B)
LAYOUT_HUMANOID = {
    "cams": [
        "observation.images.cam_high_rgb",
        "observation.images.cam_third_view",
        "observation.images.cam_left_wrist_rgb",
        "observation.images.cam_right_wrist_rgb",
    ],
    "prompt": (
        "A multi-view video of an Airbot MMK2 robot performing: {task}. "
        "The video shows four synchronized camera views arranged in a 2x2 grid: "
        "top-left shows the high overhead view, top-right shows the third-person side view, "
        "bottom-left shows the left wrist view, and bottom-right shows the right wrist view."
    ),
}
# humanoid Airbot MMK2 LAYOUT_A (matches prepare_multiview_humanoid_grid_batch.py LAYOUT_A)
LAYOUT_HUMANOID_A = {
    "cams": [
        "observation.images.cam_front_rgb",
        "observation.images.cam_head_rgb",
        "observation.images.cam_left_wrist_rgb",
        "observation.images.cam_right_wrist_rgb",
    ],
    "prompt": (
        "A multi-view video of an Airbot MMK2 robot performing: {task}. "
        "The video shows four synchronized camera views arranged in a 2x2 grid: "
        "top-left shows the front view, top-right shows the head view, "
        "bottom-left shows the left wrist view, and bottom-right shows the right wrist view."
    ),
}


def detect_layout(task_dir: Path) -> dict:
    """Pick the grid layout by inspecting which camera dirs exist, matching the
    precedence in prepare_multiview_humanoid_grid_batch.py (LAYOUT_A takes
    priority when cam_front_rgb is present, then LAYOUT_B). Defaults to bimanual
    (backward-compatible) if neither humanoid signature is found."""
    videos_root = task_dir / "videos"
    cam_dirs: set[str] = set()
    if videos_root.exists():
        for chunk_dir in sorted(videos_root.iterdir()):
            if chunk_dir.is_dir():
                cam_dirs = {d.name for d in chunk_dir.iterdir() if d.is_dir()}
                break
    if "observation.images.cam_front_rgb" in cam_dirs:
        return LAYOUT_HUMANOID_A
    if "observation.images.cam_high_rgb" in cam_dirs or "observation.images.cam_third_view" in cam_dirs:
        return LAYOUT_HUMANOID
    return LAYOUT_BIMANUAL


# Module-level layout, set per-task in main(); functions below read these.
CAM_LAYOUT = LAYOUT_BIMANUAL["cams"]
PROMPT_4VIEW = LAYOUT_BIMANUAL["prompt"]

_LPIPS_MODEL = None


def _get_lpips(device):
    """Cache the LPIPS (VGG) model once so it is not reloaded every episode."""
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        import piq
        _LPIPS_MODEL = piq.LPIPS().to(device).eval()
    return _LPIPS_MODEL


def compute_video_metrics(pred, gt_cut, batch=32):
    """PSNR/SSIM/LPIPS on GPU (if available), batched to bound memory.

    Running these on CPU over a full-episode 2x2 grid video is the rollout's
    real bottleneck; moving to CUDA + caching LPIPS makes it near-instant.
    """
    import torch
    import piq

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.from_numpy(pred).permute(0, 3, 1, 2).float().div_(255.0).clamp_(0, 1)
    y = torch.from_numpy(gt_cut).permute(0, 3, 1, 2).float().div_(255.0).clamp_(0, 1)

    lpips_model = _get_lpips(device)
    psnr_sum = ssim_sum = lpips_sum = 0.0
    n = x.shape[0]
    with torch.no_grad():
        for i in range(0, n, batch):
            xb = x[i:i + batch].to(device, non_blocking=True)
            yb = y[i:i + batch].to(device, non_blocking=True)
            bs = xb.shape[0]
            psnr_sum += piq.psnr(xb, yb).mean().item() * bs
            ssim_sum += piq.ssim(xb, yb).mean().item() * bs
            lpips_sum += lpips_model(xb, yb).mean().item() * bs
    return psnr_sum / n, ssim_sum / n, lpips_sum / n



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
    p.add_argument("--num-conditional-frames", default=5, type=int, choices=[1, 5],
                   help="Frames to condition each rolled-out segment after the first. "
                        "1 => 1 latent frame (matches image-conditioning, consistent across segments); "
                        "5 => 2 latent frames (also in train distribution, smoother but mixes with seg0).")
    p.add_argument("--max-episodes", default=None, type=int)
    p.add_argument("--max-segments", default=None, type=int,
                   help="Hard cap on rolled-out segments per episode (safety). Default: derived from GT length.")
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--save-fps", default=16, type=int)
    p.add_argument("--disable-guardrail", action="store_true", default=True)

    # ---- AUGMENT-only options ----
    p.add_argument("--manifest", default="",
                   help="augment_manifest.json. When set, ONLY the (task, episode) pairs "
                        "listed under the selected config/category/branch are generated, "
                        "with the manifest prompt overriding the dataset instruction.")
    p.add_argument("--manifest-config", default="dreamgen_bimanual_multiview",
                   help="Manifest section to read (see scripts/augment/augment_paths.py).")
    p.add_argument("--branch", default="", choices=["", "edit_only", "prompt_only", "combo"],
                   help="Augment branch. Also becomes the episode dir's __tag suffix.")
    p.add_argument("--edited-frames-root", dest="edited_frames_root", default="",
                   help="Augment INPUT base for this config, i.e. sampling_dataset/"
                        "augment/bimanual/input/multiview/dreamgen. The recolored grid "
                        "frame is looked up at {root}/{category}/{task}/first_frames_edited/"
                        "{leaf}.png (leaf = augment_paths.leaf_name), so it always matches "
                        "what edit_dreamgen_first_frames.py wrote. Used by edit_only/combo.")
    p.add_argument("--prompt-refine", action="store_true",
                   help="Enable DreamGen's prompt refiner (prompt_only/combo branches).")
    p.add_argument("--no-gt", dest="no_gt", action="store_true",
                   help="Augmented scenes have no ground truth: skip full_gt.mp4 / "
                        "full_merged.mp4 and all reference metrics (PSNR/SSIM/LPIPS). "
                        "GT is still read internally for the target length only.")

    # Cache backends (WorldCache / DiCache / FasterCache / ScalingCache). These
    # register hyphenated flags (--worldcache-enabled, etc.) whose parsed attrs
    # (worldcache_enabled, ...) are forwarded onto the pipeline setup args so
    # examples.video2world.setup_pipeline builds the matching cache_config.
    try:
        import os as _os
        import sys as _sys
        _root = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        from methods.cache_strategy.config import add_dreamgen_cache_args
        add_dreamgen_cache_args(p)
    except Exception as _e:  # pragma: no cover - cache args are optional
        print(f"[warn] could not register cache args: {_e}")

    # Sparse-attention backends (PISA / SVG / Radial). Same forwarding trick as the
    # cache backends: register their flags (use_pisa/use_svg/use_radial + params)
    # so setup_pipeline (examples.video2world) builds the matching sparse config.
    try:
        from methods.sparse_attention.pisa.config import add_pisa_args
        from methods.sparse_attention.svg.config import add_dreamgen_svg_args
        from methods.sparse_attention.radial.config import add_dreamgen_radial_args
        add_pisa_args(p)
        add_dreamgen_svg_args(p)
        add_dreamgen_radial_args(p)
    except Exception as _e:  # pragma: no cover - sparse args are optional
        print(f"[warn] could not register sparse-attention args: {_e}")

    # Token-pruning backends (SiTo / ITM). Same forwarding trick: register their
    # flags so setup_pipeline (examples.video2world) builds the matching config.
    try:
        from methods.prunning.SiTo.config import add_dreamgen_sito_args
        from methods.prunning.importance_token_merge.config import add_dreamgen_itm_args
        add_dreamgen_sito_args(p)
        add_dreamgen_itm_args(p)
    except Exception as _e:  # pragma: no cover - pruning args are optional
        print(f"[warn] could not register pruning args: {_e}")
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
    # Augment's prompt_only/combo branches deliberately run the refiner so the
    # overridden prompt gets expanded the way DreamGen expects at train time.
    pipe_args.disable_prompt_refiner = not args.prompt_refine
    pipe_args.offload_prompt_refiner = False
    pipe_args.offload_text_encoder = False
    pipe_args.downcast_text_encoder = False
    pipe_args.save_path = str(Path(args.save_root) / "_pipe_init.mp4")
    pipe_args.benchmark = False
    pipe_args.use_cuda_graphs = False
    pipe_args.natten = False

    # Forward any cache-backend flags (worldcache_*/dicache_*/fastercache_*/
    # scalingcache_*) onto the pipeline setup args so setup_pipeline can build
    # the matching cache_config. build_cache_runtime_config reads these attrs.
    for _attr in vars(args):
        if _attr.startswith(("worldcache_", "dicache_", "fastercache_", "scalingcache_")):
            setattr(pipe_args, _attr, getattr(args, _attr))

    # Forward sparse-attention flags (use_pisa/pisa_*, use_svg/svg_*, use_radial/
    # radial_*) so setup_pipeline builds the matching sparse config.
    for _attr in vars(args):
        if _attr.startswith(("pisa_", "svg_", "radial_")) or _attr in ("use_pisa", "use_svg", "use_radial"):
            setattr(pipe_args, _attr, getattr(args, _attr))

    # Forward token-pruning flags (use_sito/sito_*, use_itm/itm_*) so setup_pipeline
    # builds the matching prune config.
    for _attr in vars(args):
        if _attr.startswith(("sito_", "itm_")) or _attr in ("use_sito", "use_itm"):
            setattr(pipe_args, _attr, getattr(args, _attr))

    print("Building Video2World pipeline (loads model once)...")
    pipe = setup_pipeline(pipe_args)

    neg_prompt = (
        "The video captures a scene with low visual quality, blurring, jittering, "
        "distortion, static frames, or unnatural transitions."
    )

    iter_name = args.iter_name or Path(args.dit_path).stem  # e.g. iter_000010000
    task_paths = [Path(p.strip()) for p in args.dataset_path.split(",") if p.strip()]

    ep_done = 0
    all_psnr: list[float] = []
    all_ssim: list[float] = []
    all_lpips: list[float] = []

    # ---- AUGMENT: manifest-driven episode selection + prompt override ----
    # manifest_sel maps (task, ep_idx) -> entry, so a task dir only generates the
    # episodes this branch asked for, with the manifest prompt replacing the
    # dataset instruction. Empty manifest => behave like the dense rollout.
    manifest_sel: dict[tuple[str, int], dict] = {}
    if args.manifest:
        if not args.branch:
            raise SystemExit("--manifest requires --branch (edit_only|prompt_only|combo)")
        _mf = json.loads(Path(args.manifest).read_text())
        _section = _mf.get(args.manifest_config, {})
        for _cat, _branches in _section.items():
            if _cat.startswith("_"):
                continue
            for _e in _branches.get(args.branch, []):
                manifest_sel[(_e["task"], int(_e["episode"]))] = _e
        if not manifest_sel:
            raise SystemExit(
                f"no entries in {args.manifest} for {args.manifest_config}/*/{args.branch}")
        print(f"[augment] branch={args.branch} entries={len(manifest_sel)} "
              f"refiner={'on' if args.prompt_refine else 'off'} no_gt={args.no_gt}")

    for task_dir in task_paths:
        if not task_dir.exists():
            print(f"[WARN] task dir missing: {task_dir}")
            continue
        category = args.category.strip() or infer_category(task_dir)
        task_name = task_dir.name

        # Select the grid layout / prompt for this embodiment (humanoid vs bimanual).
        global CAM_LAYOUT, PROMPT_4VIEW
        _layout = detect_layout(task_dir)
        CAM_LAYOUT = _layout["cams"]
        PROMPT_4VIEW = _layout["prompt"]

        episodes_meta = read_episodes_jsonl(task_dir / "meta")
        if not episodes_meta:
            print(f"[WARN] no episodes in {task_dir}")
            continue

        ep_indices = sorted(episodes_meta.keys())
        if manifest_sel:
            ep_indices = [i for i in ep_indices if (task_name, i) in manifest_sel]
            if not ep_indices:
                continue
        save_task_dir = Path(args.save_root) / category / iter_name / task_name

        # max_episodes is PER-TASK (matches single_arm balance across the benchmark).
        task_done = 0
        for ep_idx in ep_indices:
            if args.max_episodes is not None and task_done >= args.max_episodes:
                break
            entry = manifest_sel.get((task_name, ep_idx))
            # Augment leaves carry a __{branch} suffix so they never collide with
            # (and stay diffable against) the dense episode of the same id.
            ep_name = f"episode_{ep_idx:06d}"
            if args.branch:
                ep_name = f"{ep_name}__{args.branch}"
            ep_dir = save_task_dir / ep_name
            if (ep_dir / "full_pred.mp4").exists() and (ep_dir / "metrics.json").exists():
                print(f"[SKIP] {category}/{task_name}/{ep_name} (done)")
                try:
                    with open(ep_dir / "metrics.json") as f:
                        m = json.load(f)
                    if m.get("psnr") is not None:
                        all_psnr.append(m["psnr"])
                        all_ssim.append(m["ssim"])
                        all_lpips.append(m["lpips"])
                except (json.JSONDecodeError, KeyError, OSError):
                    pass
                ep_done += 1
                task_done += 1
                continue

            task_clean = episodes_meta[ep_idx].rstrip(".,;").strip()
            prompt = PROMPT_4VIEW.format(task=task_clean)
            # prompt_only / combo perturb the text. The perturbation goes INTO the
            # `{task}` slot of this layout's caption template rather than replacing
            # the whole caption: the template is what tells the model the frame is a
            # 2x2 grid of specific views, so dropping it would change the output
            # FORMAT instead of just the described behaviour. edit_only keeps the
            # dense caption byte-identical (only the image changed).
            if entry and args.branch in ("prompt_only", "combo") and entry.get("prompt"):
                prompt = PROMPT_4VIEW.format(task=PR.task_phrase(entry["prompt"]))

            with tempfile.TemporaryDirectory() as td:
                tmp_dir = Path(td)
                gt_mp4 = tmp_dir / "gt_grid.mp4"
                if not build_gt_grid_video(task_dir, ep_idx, gt_mp4, args.save_fps):
                    print(f"[WARN] could not build GT grid for episode_{ep_idx:06d}, skipping")
                    continue
                gt_frames = load_video_frames(gt_mp4)
                gt_len = gt_frames.shape[0]

                # First-frame image used to condition segment 0. edit_only/combo
                # swap in the recolored grid frame; the GT grid is then only used
                # for the target length (and, unless --no-gt, for metrics).
                first_png = tmp_dir / "first.png"
                import mediapy
                edited_png = None
                if args.edited_frames_root:
                    # Same leaf naming as everything else in the augment tree, so the
                    # PNG that edit_dreamgen_first_frames.py wrote is found by
                    # construction rather than by a hand-rolled filename guess.
                    leaf = AP.leaf_name(args.manifest_config, entry, args.branch)
                    cand = (Path(args.edited_frames_root) / category / task_name
                            / "first_frames_edited" / f"{leaf}.png")
                    if cand.exists():
                        edited_png = cand
                    else:
                        print(f"[WARN] edited frame missing: {cand}, skipping episode")
                        continue
                if edited_png is not None:
                    first_png = edited_png
                else:
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
                if not args.no_gt:
                    write_video_frames(ep_dir / "full_gt.mp4", gt_cut, args.save_fps)
                    merged = np.concatenate([gt_cut, pred], axis=2)
                    write_video_frames(ep_dir / "full_merged.mp4", merged, args.save_fps)

                # Metrics (PSNR/SSIM/LPIPS) against GT, same convention as single_arm.
                # Run on GPU + cached LPIPS (was the rollout's CPU bottleneck).
                # Skipped entirely under --no-gt: the edited scene has no GT, so a
                # reference metric would be meaningless (reference-free metrics come
                # from scripts/augment/compute_augment_metrics.py instead).
                psnr_val = ssim_val = lpips_val = None
                if not args.no_gt:
                    try:
                        psnr_val, ssim_val, lpips_val = compute_video_metrics(pred, gt_cut)
                    except (RuntimeError, ValueError, ImportError) as e:
                        print(f"   Metrics failed: {e}")

                _metrics = {
                    "total_frames_pred": int(len(pred)),
                    "total_frames_gt": int(gt_len),
                    "trajectory_id": int(ep_idx),
                    "num_segments": len(segments),
                    "mode": "autoregressive_augment_no_gt" if args.no_gt else "autoregressive",
                    "prompt": prompt,
                }
                if not args.no_gt:
                    _metrics.update({"psnr": psnr_val, "ssim": ssim_val, "lpips": lpips_val})
                json.dump(_metrics, open(ep_dir / "metrics.json", "w"),
                          indent=2, ensure_ascii=False)

                if args.branch:
                    json.dump(
                        {
                            "config": args.manifest_config,
                            "branch": args.branch,
                            "category": category,
                            "task": task_name,
                            "episode": f"{ep_idx:06d}",
                            "dense_stem": f"episode_{ep_idx:06d}",
                            "object": (entry or {}).get("object"),
                            "prompt_used": prompt,
                            "prompt_original": PROMPT_4VIEW.format(task=task_clean),
                            "edit_instruction": (entry or {}).get("edit_instruction"),
                            "edited_first_frame": str(edited_png) if edited_png else None,
                            "prompt_refiner": bool(args.prompt_refine),
                            "seed": args.seed,
                        },
                        open(ep_dir / "augment_info.json", "w"), indent=2, ensure_ascii=False,
                    )
                if psnr_val is not None:
                    all_psnr.append(psnr_val)
                    all_ssim.append(ssim_val)
                    all_lpips.append(lpips_val)
                ep_done += 1
                task_done += 1
                _m = (f"PSNR={psnr_val:.2f} SSIM={ssim_val:.4f} LPIPS={lpips_val:.4f}"
                      if psnr_val is not None
                      else ("no-gt augment mode" if args.no_gt else "metrics=NA"))
                print(f"   -> {len(pred)} frames in {len(segments)} segments | {_m}")

    if all_psnr:
        summary = {
            "mean_psnr": sum(all_psnr) / len(all_psnr),
            "mean_ssim": sum(all_ssim) / len(all_ssim),
            "mean_lpips": sum(all_lpips) / len(all_lpips),
            "num_episodes_processed": len(all_psnr),
            "mode": "autoregressive",
        }
        summary_path = Path(args.save_root) / f"{iter_name}_summary.json"
        json.dump(summary, open(summary_path, "w"), indent=2)
        print(f"\n=== Summary ({len(all_psnr)} episodes) ===")
        print(f"PSNR: {summary['mean_psnr']:.3f} | SSIM: {summary['mean_ssim']:.4f} | LPIPS: {summary['mean_lpips']:.4f}")

    print(f"\n=== Done: {ep_done} episodes ===")

    # Cleanup distributed if multi-GPU
    try:
        from examples.video2world import cleanup_distributed
        cleanup_distributed()
    except Exception:
        pass


if __name__ == "__main__":
    main()
