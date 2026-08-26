#!/usr/bin/env python3
"""Build the first 300-video MULTIVIEW labeling batch.

100 points × 3 videos (consecutive 16-frame windows).
Each video is the two STATIC cameras playing side-by-side, with a Farneback
flow heatmap overlay (thanhnv/human-assessment) so raters can see motion.

Static-camera mapping
  4-view strip 1280x192 (Ctrl-World 4-view grid): tiles 0+1 = cam_high, cam_low
  3-view strip  960x192 (single_arm Ctrl-World):   tiles 0+1 = exterior_1, exterior_2
  2x2 grid     768x432 (DreamGen multiview):      TL + TR  = two static cameras

Usage (OpenCV must work — use the DreamGen venv):
    models/dreamgen/.venv/bin/python scripts/build_assessment_multiview.py
    models/dreamgen/.venv/bin/python scripts/build_assessment_multiview.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_VENV = REPO / "models" / "dreamgen" / ".venv" / "bin" / "python"
if _VENV.is_file() and Path(sys.executable).resolve() != _VENV.resolve():
    os.execv(str(_VENV), [str(_VENV), *sys.argv])

import cv2  # noqa: E402
import numpy as np  # noqa: E402

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from methods.human_assessment.video_overlay import overlay_flow_rgb  # noqa: E402

SEG_LEN = 16
SEGS_PER_POINT = 3
NEED_FRAMES = SEG_LEN * SEGS_PER_POINT  # 48
N_POINTS = 100
HF_REPO = "doanh25032004/video_gen_physics"
HF_PREFIX = "assessment/multiview_triplets"


def read_mp4(path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 16.0)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def write_mp4(path: Path, frames: list[np.ndarray], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    w -= w % 2
    h -= h % 2
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", f"{fps:.4f}", "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert proc.stdin is not None
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f[:h, :w]).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}")


def static_pair(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """Crop the two static cameras out of a tiled multiview frame."""
    h, w = frame.shape[:2]
    if h <= 240 and w % 320 == 0:
        n = w // 320
        if n < 2:
            raise ValueError(f"strip too narrow: {w}x{h}")
        a = frame[:, 0:320]
        b = frame[:, 320:640]
        return a, b, f"strip{n}"
    # 2x2 grid (DreamGen 768x432 and similar 16:9 canvases)
    th, tw = h // 2, w // 2
    a = frame[:th, :tw]
    b = frame[:th, tw:tw * 2]
    return a, b, "grid2x2"


def dual_static_with_flow(frames: list[np.ndarray], use_flow: bool) -> list[np.ndarray]:
    left, right = [], []
    layout = None
    for f in frames:
        a, b, layout = static_pair(f)
        left.append(a)
        right.append(b)
    if use_flow:
        left = overlay_flow_rgb(left)
        right = overlay_flow_rgb(right)
    out = []
    for a, b in zip(left, right):
        if a.shape[0] != b.shape[0]:
            h = min(a.shape[0], b.shape[0])
            a, b = a[:h], b[:h]
        gap = np.full((a.shape[0], 4, 3), 32, dtype=np.uint8)
        out.append(np.concatenate([a, gap, b], axis=1))
    return out, layout


def probe_ok(path: Path) -> int | None:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return None
    s = r.stdout.strip().split(",")[-1]
    return int(s) if s.isdigit() else None


def sample_sources(items: list[dict], clips_dir: Path, n: int, seed: int) -> list[dict]:
    eligible = []
    for it in items:
        if it.get("view") != "multiview":
            continue
        nframes = probe_ok(clips_dir / f"{it['id']}.mp4")
        if nframes is None or nframes < NEED_FRAMES:
            continue
        eligible.append({**it, "_nframes": nframes})

    by_cell: dict[str, list[dict]] = defaultdict(list)
    for it in eligible:
        q = "low" if it["role"] in ("weak_augment", "strong_augment") else "high"
        by_cell[f"{it['embodiment']}/{q}"].append(it)

    rng = random.Random(seed)
    for k in by_cell:
        rng.shuffle(by_cell[k])

    # round-robin so high/low and embodiments stay mixed; prefer some low
    keys = sorted(by_cell)
    picked, i = [], 0
    while len(picked) < n and any(by_cell[k] for k in keys):
        k = keys[i % len(keys)]
        if by_cell[k]:
            picked.append(by_cell[k].pop())
        i += 1
        if i > n * 20:
            break
    return picked[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-points", type=int, default=N_POINTS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-flow", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--hf-repo", default=HF_REPO)
    args = ap.parse_args()

    dest = REPO / "sampling_dataset" / "assessment" / "multiview_triplets"
    clips_src = REPO / "sampling_dataset" / "data_for_web" / "clips"
    manifest_src = REPO / "sampling_dataset" / "data_for_web" / "manifest.json"
    items = json.loads(manifest_src.read_text())["items"]

    print("probing multiview sources (need ≥48 frames)…")
    sources = sample_sources(items, clips_src, args.n_points, args.seed)
    print(f"picked {len(sources)} sources")
    from collections import Counter
    print("  embodiment", Counter(s["embodiment"] for s in sources))
    print("  quality   ", Counter("low" if s["role"] in ("weak_augment", "strong_augment") else "high" for s in sources))
    print("  model     ", Counter(s["model"] for s in sources))
    if args.dry_run:
        return 0

    if dest.exists():
        import shutil
        shutil.rmtree(dest)
    points_dir = dest / "points"
    points_dir.mkdir(parents=True)

    catalog_items = []
    use_flow = not args.no_flow
    base_url = f"https://huggingface.co/datasets/{args.hf_repo}/resolve/main/{HF_PREFIX}/points"

    for i, src in enumerate(sources):
        pid = f"{i:05d}"
        src_mp4 = clips_src / f"{src['id']}.mp4"
        n = src["_nframes"]
        rng = random.Random(f"{args.seed}:{src['id']}")
        max_start = n - NEED_FRAMES
        n_win = max_start // SEG_LEN + 1
        start = rng.randrange(n_win) * SEG_LEN
        start = min(start, max_start)

        tmp = dest / "_tmp_window.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src_mp4),
             "-vf", f"trim=start_frame={start}:end_frame={start + NEED_FRAMES},setpts=PTS-STARTPTS",
             "-an", str(tmp)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        frames, fps = read_mp4(tmp)
        tmp.unlink(missing_ok=True)
        if len(frames) < NEED_FRAMES:
            print(f"  skip {src['id']}: got {len(frames)} frames")
            continue

        pdir = points_dir / pid
        pdir.mkdir(parents=True, exist_ok=True)
        videos = []
        layout = None
        for k in range(SEGS_PER_POINT):
            sl = k * SEG_LEN
            window = frames[sl: sl + SEG_LEN]
            if len(window) < SEG_LEN:
                print(f"  skip {src['id']} seg{k}: {len(window)} frames")
                continue
            dual, layout = dual_static_with_flow(window, use_flow)
            out = pdir / f"seg{k}.mp4"
            write_mp4(out, dual, fps if fps > 1 else 16.0)
            videos.append(f"{base_url}/{pid}/seg{k}.mp4")
        quality = "low" if src["role"] in ("weak_augment", "strong_augment") else "high"
        row = {
            "id": pid,
            "mode": "triplet",
            "videos": videos,
            "n_videos": 3,
            "seg_len": SEG_LEN,
            "start_frame": start,
            "layout": layout,
            "source_id": src["id"],
            "embodiment": src["embodiment"],
            "view": "multiview",
            "model": src["model"],
            "split": src["split"],
            "task": src["task"],
            "prompt": src.get("prompt", ""),
            "quality": quality,
        }
        catalog_items.append(row)
        (pdir / "meta.json").write_text(json.dumps(row, indent=2) + "\n")
        print(f"  [{i+1}/{len(sources)}] {pid} {src['embodiment']} {layout} start={start} q={quality}")

    catalog = {
        "mode": "triplet",
        "n": len(catalog_items),
        "n_videos": len(catalog_items) * SEGS_PER_POINT,
        "seg_len": SEG_LEN,
        "hf_repo": args.hf_repo,
        "hf_prefix": HF_PREFIX,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "items": catalog_items,
    }
    (dest / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    (dest / "manifest.json").write_text(json.dumps(catalog, indent=2) + "\n")
    (dest / "README.md").write_text(
        f"""# Multiview triplet batch (first 300 videos)

- **{len(catalog_items)} points × 3 videos = {len(catalog_items) * SEGS_PER_POINT} clips**
- Each clip is **{SEG_LEN} frames** of the two static cameras, side by side
- Flow heatmap overlay from `methods/human_assessment/` (thanhnv/human-assessment)
- Score **1 / 2 / 3** per video (see webpage/labeling rubric placeholders)

```bash
models/dreamgen/.venv/bin/python scripts/build_assessment_multiview.py
```
"""
    )
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
