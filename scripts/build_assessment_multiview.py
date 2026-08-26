#!/usr/bin/env python3
"""Build the first 300-video MULTIVIEW labeling batch.

Each item is the FULL original clip, cropped to the two STATIC cameras only,
with two mp4s written side-by-side for the rater:

  raw.mp4   — two static views, no overlay
  flow.mp4  — same frames + Farneback heatmap (thanhnv/human-assessment)

16-frame segment scoring is Mode B (remaining 300, mixed MV+SV) — not here.

Static-camera mapping
  4-view strip 1280x192 (Ctrl-World 4-view grid): tiles 0+1 = cam_high, cam_low
  3-view strip  960x192 (single_arm Ctrl-World):   tiles 0+1 = exterior_1, exterior_2
  2x2 grid     768x432 (DreamGen multiview):      TL + TR  = two static cameras

Usage (OpenCV must work — use the DreamGen venv):
    models/dreamgen/.venv/bin/python scripts/build_assessment_multiview.py
    models/dreamgen/.venv/bin/python scripts/build_assessment_multiview.py --limit 2
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import shutil
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

N_VIDEOS = 300
HF_REPO = "doanh25032004/video_gen_physics"
HF_PREFIX = "assessment/multiview_static2"
SAMPLING = REPO / "sampling_dataset"
DEST = SAMPLING / "assessment" / "multiview_static2"


def _load_script(name: str):
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


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
        "-loglevel", "error",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f[:h, :w]).tobytes())
    proc.stdin.close()
    err = proc.stderr.read() if proc.stderr else b""
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}: {err[-400:]}")


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
    th, tw = h // 2, w // 2
    a = frame[:th, :tw]
    b = frame[:th, tw:tw * 2]
    return a, b, "grid2x2"


def hstack_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape[0] != b.shape[0]:
        h = min(a.shape[0], b.shape[0])
        a, b = a[:h], b[:h]
    gap = np.full((a.shape[0], 4, 3), 32, dtype=np.uint8)
    return np.concatenate([a, gap, b], axis=1)


def dual_static(frames: list[np.ndarray], with_flow: bool) -> tuple[list[np.ndarray], str]:
    left, right = [], []
    layout = "unknown"
    for f in frames:
        a, b, layout = static_pair(f)
        left.append(a)
        right.append(b)
    if with_flow:
        left = overlay_flow_rgb(left)
        right = overlay_flow_rgb(right)
    return [hstack_pair(a, b) for a, b in zip(left, right)], layout


def collect_sources(n: int, seed: int) -> list[dict]:
    """200 curated MV from data_for_web, then fill from dense to reach n."""
    man = json.loads((SAMPLING / "data_for_web" / "manifest.json").read_text())
    clips_dir = SAMPLING / "data_for_web" / "clips"
    web = []
    used_ep: set[str] = set()
    for it in man["items"]:
        if it.get("view") != "multiview":
            continue
        src = clips_dir / f"{it['id']}.mp4"
        if not src.is_file():
            continue
        ep = it.get("episode") or it["id"]
        used_ep.add(ep)
        web.append(
            {
                "src": src,
                "embodiment": it["embodiment"],
                "split": it.get("split", ""),
                "task": it.get("task", ""),
                "prompt": it.get("prompt") or it.get("task", ""),
                "episode": ep,
                "pool": "data_for_web",
            }
        )

    extras: list[dict] = []
    if len(web) < n:
        bdw = _load_script("build_data_for_web")
        dense = SAMPLING / "dense"
        cands = [c for c in bdw.discover(dense) if c.get("view") == "multiview"]
        rng = random.Random(seed)
        rng.shuffle(cands)
        by_emb: dict[str, list[dict]] = defaultdict(list)
        for c in cands:
            if c.get("episode") in used_ep:
                continue
            pred = bdw.pred_file(c["src"], c["kind"])
            if not pred.is_file():
                continue
            c = {**c, "pred": pred}
            by_emb[c["embodiment"]].append(c)
        keys = sorted(by_emb)
        i = 0
        need = n - len(web)
        while len(extras) < need and any(by_emb[k] for k in keys):
            k = keys[i % len(keys)]
            i += 1
            if not by_emb[k]:
                continue
            c = by_emb[k].pop()
            used_ep.add(c["episode"])
            extras.append(
                {
                    "src": c["pred"],
                    "embodiment": c["embodiment"],
                    "split": c.get("split", ""),
                    "task": c.get("task", ""),
                    "prompt": (c.get("task") or "").replace("_", " "),
                    "episode": c["episode"],
                    "pool": "dense",
                }
            )

    picked = web + extras
    rng = random.Random(seed)
    rng.shuffle(picked)
    return picked[:n]


def public_row(pid: str, src: dict, layout: str, n_frames: int, fps: float, base_url: str) -> dict:
    return {
        "id": pid,
        "mode": "full_static2",
        "video_raw": f"{base_url}/{pid}/raw.mp4",
        "video_flow": f"{base_url}/{pid}/flow.mp4",
        "n_frames": n_frames,
        "fps": round(float(fps), 4),
        "layout": layout,
        "embodiment": src["embodiment"],
        "view": "multiview",
        "split": src["split"],
        "task": src["task"],
        "prompt": src["prompt"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=N_VIDEOS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="process only the first N (smoke test)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--hf-repo", default=HF_REPO)
    ap.add_argument("--keep-existing", action="store_true", help="do not wipe DEST; skip ids that already have raw+flow")
    args = ap.parse_args()

    n = args.n
    print(f"collecting {n} unique full-length multiview sources…")
    sources = collect_sources(n, args.seed)
    print(f"picked {len(sources)}")
    from collections import Counter
    print("  embodiment", Counter(s["embodiment"] for s in sources))
    print("  pool      ", Counter(s["pool"] for s in sources))
    if args.dry_run:
        return 0

    if not args.keep_existing and DEST.exists():
        shutil.rmtree(DEST)
    clips_dir = DEST / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    base_url = f"https://huggingface.co/datasets/{args.hf_repo}/resolve/main/{HF_PREFIX}/clips"
    catalog_items = []
    todo = sources[: args.limit] if args.limit else sources

    for i, src in enumerate(todo):
        pid = f"{i:05d}"
        out_dir = clips_dir / pid
        raw_p, flow_p = out_dir / "raw.mp4", out_dir / "flow.mp4"
        if args.keep_existing and raw_p.is_file() and flow_p.is_file():
            print(f"  [{i+1}/{len(todo)}] {pid} skip existing")
            continue
        try:
            frames, fps = read_mp4(Path(src["src"]))
        except Exception as e:
            print(f"  skip {src['src']}: {e}")
            continue
        if len(frames) < 8:
            print(f"  skip {src['src']}: {len(frames)} frames")
            continue
        try:
            raw, layout = dual_static(frames, with_flow=False)
            flow, _ = dual_static(frames, with_flow=True)
        except Exception as e:
            print(f"  skip {src['src']} crop/flow: {e}")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        fps_out = fps if fps > 1 else 16.0
        write_mp4(raw_p, raw, fps_out)
        write_mp4(flow_p, flow, fps_out)
        row = public_row(pid, src, layout, len(frames), fps_out, base_url)
        catalog_items.append(row)
        (out_dir / "meta.json").write_text(json.dumps({**row, "episode": src["episode"], "pool": src["pool"]}, indent=2) + "\n")
        print(f"  [{i+1}/{len(todo)}] {pid} {src['embodiment']} {layout} {len(frames)}f {src['pool']}")

    catalog = {
        "mode": "full_static2",
        "n": len(catalog_items),
        "n_videos": len(catalog_items),
        "hf_repo": args.hf_repo,
        "hf_prefix": HF_PREFIX,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": "Full-length multiview, two static cameras only. Each item has raw.mp4 + flow.mp4.",
        "items": catalog_items,
    }
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    public = {**catalog, "items": [{k: it[k] for k in (
        "id", "mode", "video_raw", "video_flow", "n_frames", "fps",
        "layout", "embodiment", "view", "split", "task", "prompt",
    )} for it in catalog_items]}
    (DEST / "catalog.public.json").write_text(json.dumps(public, indent=2) + "\n")
    (DEST / "README.md").write_text(
        f"""# Multiview static-pair batch (first 300 videos)

- **{len(catalog_items)} full-length videos** (not 16-frame cuts)
- Each video shows **two static cameras** only
- `raw.mp4` + `flow.mp4` (optical-flow overlay from `methods/human_assessment/`)
- Score **1 / 2 / 3** once per video

```bash
models/dreamgen/.venv/bin/python scripts/build_assessment_multiview.py
```
"""
    )
    print(f"\nwrote {DEST}  ({len(catalog_items)} items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
