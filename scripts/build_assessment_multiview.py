#!/usr/bin/env python3
"""Build the curated 300-video MULTIVIEW labeling batch.

Each item is the FULL original clip, cropped to the two STATIC cameras only,
with two mp4s written side-by-side for the rater:

  raw.mp4       — two static views, no overlay
  tracking.mp4  — same frames + CoTracker trajectories from assets/seed_points.json

Videos without usable seeds fall back to a Farneback flow overlay. Source order
comes from assets/all_videos.md. The 3×3 reference examples in
assets/examples.md are exported as examples.public.json.

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
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

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
ASSETS = REPO / "assets"
ALL_VIDEOS = ASSETS / "all_videos.md"
EXAMPLES = ASSETS / "examples.md"
SEED_POINTS = ASSETS / "seed_points.json"


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


def read_path_list(path: Path) -> list[str]:
    """Read one relative POSIX video path per non-comment line."""
    rows = [line.strip() for line in path.read_text().splitlines()]
    return [row for row in rows if row and not row.startswith("#")]


def source_key(source_path: str) -> str:
    """Return seed_points.json key for a source video path."""
    path = PurePosixPath(source_path)
    if path.name in {"full_pred.mp4", "pred_all_views.mp4"}:
        return path.parent.as_posix()
    return path.with_suffix("").as_posix()


def _task_from_path(parts: tuple[str, ...], mv_idx: int, model: str) -> str:
    if model.startswith("ctrlworld") and len(parts) > mv_idx + 3:
        episode = parts[mv_idx + 3]
        match = re.match(r"episode_(.+?)__\d+", episode, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    if model == "dreamgen" and len(parts) > mv_idx + 4:
        candidate = parts[mv_idx + 3]
        if candidate.startswith("iter_"):
            return parts[mv_idx + 4]
    return "multiview_robot_action"


def _manifest_metadata() -> dict[str, dict]:
    path = SAMPLING / "data_for_web" / "manifest.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    return {
        item["source_path"]: item
        for item in data.get("items", [])
        if item.get("source_path")
    }


def source_record(source_path: str, video_root: Path, metadata: dict[str, dict]) -> dict:
    """Resolve path plus blind catalog metadata for one curated source."""
    parts = PurePosixPath(source_path).parts
    item = metadata.get(source_path, {})
    if len(parts) >= 3 and parts[:2] == ("tmp", "lowquality_multiview"):
        embodiment, model, split = "single_arm", "dreamgen", ""
        task = "multiview_robot_action"
    else:
        try:
            mv_idx = parts.index("multiview")
        except ValueError as exc:
            raise ValueError(f"not a multiview source: {source_path}") from exc
        if mv_idx < 2 or len(parts) <= mv_idx + 2:
            raise ValueError(f"cannot parse multiview source: {source_path}")
        embodiment = parts[mv_idx - 2]
        model = parts[mv_idx + 1]
        split = parts[mv_idx + 2]
        task = _task_from_path(parts, mv_idx, model)

    task = item.get("task") or task
    return {
        "src": str(video_root / PurePosixPath(source_path)),
        "source_path": source_path,
        "seed_key": source_key(source_path),
        "embodiment": item.get("embodiment") or embodiment,
        "model": item.get("model") or model,
        "split": item.get("split") or split,
        "task": task,
        "prompt": item.get("prompt") or task.replace("_", " "),
        "episode": item.get("episode") or PurePosixPath(source_key(source_path)).name,
        "pool": parts[0],
    }


def collect_sources(n: int, video_root: Path = SAMPLING, videos_path: Path = ALL_VIDEOS) -> list[dict]:
    """Load deterministic curated sources from assets/all_videos.md."""
    paths = read_path_list(videos_path)
    if len(paths) != len(set(paths)):
        raise ValueError(f"duplicate paths in {videos_path}")
    if len(paths) != N_VIDEOS:
        raise ValueError(f"expected {N_VIDEOS} paths in {videos_path}, found {len(paths)}")
    metadata = _manifest_metadata()
    limit = min(n, len(paths)) if n > 0 else len(paths)
    return [source_record(path, video_root, metadata) for path in paths[:limit]]


def collect_examples(video_root: Path = SAMPLING, examples_path: Path = EXAMPLES) -> list[dict]:
    """Load Bad/Medium/Good × embodiment reference examples."""
    rating = ""
    score_by_rating = {"bad": 1, "medium": 2, "good": 3}
    metadata = _manifest_metadata()
    rows = []
    for raw_line in examples_path.read_text().splitlines():
        line = raw_line.strip()
        if line.endswith(":") and not line.startswith("-"):
            rating = line[:-1].strip().lower()
            continue
        if not line.startswith("- "):
            continue
        embodiment, source_path = line[2:].split(": ", 1)
        if rating not in score_by_rating:
            raise ValueError(f"example has no valid rating: {raw_line}")
        record = source_record(source_path, video_root, metadata)
        record.update(
            rating=rating,
            score=score_by_rating[rating],
            embodiment=embodiment.strip().lower(),
            example_id=f"{rating}-{embodiment.strip().lower().replace('_', '-')}",
        )
        rows.append(record)
    if len(rows) != 9:
        raise ValueError(f"expected 9 examples in {examples_path}, found {len(rows)}")
    return rows


def tracking_overlay_rgb(
    left: list[np.ndarray],
    right: list[np.ndarray],
    seeds: dict,
    *,
    show_flow: bool,
) -> tuple[list[np.ndarray], list[np.ndarray], str]:
    """Track configured points in each static view and render short trails."""
    if not seeds.get("view1") or not seeds.get("view2"):
        return overlay_flow_rgb(left), overlay_flow_rgb(right), "flow_fallback"

    from methods.human_assessment.keypoints import track_keypoints
    from methods.human_assessment.overlay import draw_trajectories

    output = []
    for view_name, frames in (("view1", left), ("view2", right)):
        bgr_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]
        points = np.asarray(seeds[view_name], dtype=np.float32)
        trajectory, alive = track_keypoints(bgr_frames, points)
        bases = overlay_flow_rgb(frames) if show_flow else frames
        rendered = []
        for frame_idx, base in enumerate(bases):
            bgr = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
            drawn = draw_trajectories(bgr, trajectory, alive, frame_idx)
            rgb = cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)
            cv2.putText(
                rgb,
                "CoTracker",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            rendered.append(rgb)
        output.append(rendered)
    kind = "tracking+flow" if show_flow else "tracking"
    return output[0], output[1], kind


def process_one(job: dict) -> dict | None:
    """Decode one source, write raw + diagnostic overlay, return catalog row."""
    pid, src, out_dir, base_url = job["pid"], job["src"], Path(job["out_dir"]), job["base_url"]
    overlay_mode = job.get("overlay_mode", "tracking")
    overlay_name = "flow.mp4" if overlay_mode == "flow" else "tracking.mp4"
    raw_p, overlay_p = out_dir / "raw.mp4", out_dir / overlay_name
    try:
        frames, fps = read_mp4(Path(src["src"]))
    except Exception as e:
        print(f"  skip {src['src']}: {e}", flush=True)
        return None
    if len(frames) < 8:
        print(f"  skip {src['src']}: {len(frames)} frames", flush=True)
        return None
    try:
        left, right, layout = [], [], "unknown"
        for f in frames:
            a, b, layout = static_pair(f)
            left.append(a)
            right.append(b)
        raw = [hstack_pair(a, b) for a, b in zip(left, right)]
        if overlay_mode == "flow":
            overlay_left, overlay_right = overlay_flow_rgb(left), overlay_flow_rgb(right)
            overlay_kind = "flow"
        else:
            overlay_left, overlay_right, overlay_kind = tracking_overlay_rgb(
                left,
                right,
                job.get("seeds", {}),
                show_flow=overlay_mode == "tracking+flow",
            )
        overlay = [hstack_pair(a, b) for a, b in zip(overlay_left, overlay_right)]
    except Exception as e:
        print(f"  skip {src['src']} crop/overlay: {e}", flush=True)
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    fps_out = fps if fps > 1 else 16.0
    write_mp4(raw_p, raw, fps_out)
    write_mp4(overlay_p, overlay, fps_out)
    row = public_row(
        pid,
        src,
        layout,
        len(frames),
        fps_out,
        base_url,
        overlay_name=overlay_name,
        overlay_kind=overlay_kind,
    )
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                **row,
                "episode": src["episode"],
                "pool": src["pool"],
                "source_path": src["source_path"],
                "seed_key": src["seed_key"],
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"  {pid} {src['embodiment']} {layout} {len(frames)}f {src['pool']} {overlay_kind}",
        flush=True,
    )
    return row


def public_row(
    pid: str,
    src: dict,
    layout: str,
    n_frames: int,
    fps: float,
    base_url: str,
    *,
    overlay_name: str,
    overlay_kind: str,
) -> dict:
    return {
        "id": pid,
        "mode": "full_static2",
        "video_raw": f"{base_url}/{pid}/raw.mp4",
        "video_overlay": f"{base_url}/{pid}/{overlay_name}",
        "overlay_kind": overlay_kind,
        "n_frames": n_frames,
        "fps": round(float(fps), 4),
        "layout": layout,
        "embodiment": src["embodiment"],
        "model": src["model"],
        "view": "multiview",
        "split": src["split"],
        "task": src["task"],
        "prompt": src["prompt"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=N_VIDEOS)
    ap.add_argument("--limit", type=int, default=0, help="process only the first N (smoke test)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--hf-repo", default=HF_REPO)
    ap.add_argument("--video-root", type=Path, default=SAMPLING)
    ap.add_argument("--videos", type=Path, default=ALL_VIDEOS)
    ap.add_argument("--examples", type=Path, default=EXAMPLES)
    ap.add_argument("--seed-points", type=Path, default=SEED_POINTS)
    ap.add_argument(
        "--overlay",
        choices=("tracking", "tracking+flow", "flow"),
        default="tracking",
        help="diagnostic shown beside raw video",
    )
    ap.add_argument(
        "--keep-existing",
        action="store_true",
        help="do not wipe DEST; reuse rows with matching raw + overlay outputs",
    )
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument(
        "--sync-web",
        action="store_true",
        help="copy public catalog + examples into webpage/labeling",
    )
    args = ap.parse_args()

    n = args.n
    print(f"collecting {n} unique full-length multiview sources…")
    sources = collect_sources(n, args.video_root, args.videos)
    examples = collect_examples(args.video_root, args.examples)
    seed_db = json.loads(args.seed_points.read_text())
    print(f"picked {len(sources)}")
    from collections import Counter
    print("  embodiment", Counter(s["embodiment"] for s in sources))
    print("  pool      ", Counter(s["pool"] for s in sources))
    usable_seeds = sum(
        bool(seed_db.get(src["seed_key"], {}).get("view1"))
        and bool(seed_db.get(src["seed_key"], {}).get("view2"))
        for src in sources
    )
    missing_files = [src["source_path"] for src in sources if not Path(src["src"]).is_file()]
    print(f"  seeds     {usable_seeds}/{len(sources)} usable for both views")
    print(f"  examples  {len(examples)}")
    print(f"  missing   {len(missing_files)} source files under {args.video_root}")
    if usable_seeds < len(sources) and args.overlay != "flow":
        print(f"  fallback  {len(sources) - usable_seeds} videos will use optical flow")
    if args.dry_run:
        return 0

    if not args.keep_existing and DEST.exists():
        shutil.rmtree(DEST)
    clips_dir = DEST / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    base_url = f"https://huggingface.co/datasets/{args.hf_repo}/resolve/main/{HF_PREFIX}/clips"
    todo = sources[: args.limit] if args.limit else sources
    jobs = []
    catalog_items = []
    overlay_name = "flow.mp4" if args.overlay == "flow" else "tracking.mp4"
    for i, src in enumerate(todo):
        pid = f"{i:05d}"
        out_dir = clips_dir / pid
        meta_path = out_dir / "meta.json"
        if (
            args.keep_existing
            and (out_dir / "raw.mp4").is_file()
            and (out_dir / overlay_name).is_file()
            and meta_path.is_file()
        ):
            row = json.loads(meta_path.read_text())
            if row.get("source_path") == src["source_path"] and row.get("video_overlay"):
                catalog_items.append({k: v for k, v in row.items() if k not in {"episode", "pool", "source_path", "seed_key"}})
                print(f"  {pid} reuse existing")
                continue
        jobs.append(
            {
                "pid": pid,
                "src": src,
                "out_dir": str(out_dir),
                "base_url": base_url,
                "overlay_mode": args.overlay,
                "seeds": seed_db.get(src["seed_key"], {}),
            }
        )

    workers = max(1, args.workers)
    if args.overlay != "flow" and workers != 1:
        print("CoTracker uses one GPU model; forcing --workers 1")
        workers = 1
    print(f"processing {len(jobs)} clips with {workers} workers…", flush=True)
    if workers == 1:
        for job in jobs:
            row = process_one(job)
            if row:
                catalog_items.append(row)
    else:
        ctx = __import__("multiprocessing").get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = {ex.submit(process_one, job): job["pid"] for job in jobs}
            for fut in as_completed(futs):
                try:
                    row = fut.result()
                except Exception as e:
                    print(f"  fail {futs[fut]}: {e}", flush=True)
                    continue
                if row:
                    catalog_items.append(row)
    catalog_items.sort(key=lambda r: r["id"])

    # Eight references are also evaluation items; reuse their rendered files.
    # Process only examples outside this batch (currently Good/single_arm).
    source_to_id = {src["source_path"]: f"{i:05d}" for i, src in enumerate(todo)}
    catalog_by_id = {row["id"]: row for row in catalog_items}
    example_items = []
    if not args.limit:
        example_base_url = (
            f"https://huggingface.co/datasets/{args.hf_repo}/resolve/main/"
            f"{HF_PREFIX}/examples"
        )
        for example in examples:
            reused = catalog_by_id.get(source_to_id.get(example["source_path"], ""))
            if reused:
                row = dict(reused)
                row["id"] = example["example_id"]
            else:
                row = process_one(
                    {
                        "pid": example["example_id"],
                        "src": example,
                        "out_dir": str(DEST / "examples" / example["example_id"]),
                        "base_url": example_base_url,
                        "overlay_mode": args.overlay,
                        "seeds": seed_db.get(example["seed_key"], {}),
                    }
                )
            if row:
                example_items.append(
                    {
                        **row,
                        "score": example["score"],
                        "rating": example["rating"],
                        "embodiment": example["embodiment"],
                    }
                )

    catalog = {
        "mode": "full_static2",
        "n": len(catalog_items),
        "n_videos": len(catalog_items),
        "hf_repo": args.hf_repo,
        "hf_prefix": HF_PREFIX,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": "Curated full-length multiview videos. Two static cameras; raw + diagnostic overlay.",
        "items": catalog_items,
    }
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    public = dict(catalog)
    (DEST / "catalog.public.json").write_text(json.dumps(public, indent=2) + "\n")
    examples_public = {
        "n": len(example_items),
        "scale": ["bad", "medium", "good"],
        "items": example_items,
    }
    (DEST / "examples.public.json").write_text(json.dumps(examples_public, indent=2) + "\n")
    if args.sync_web:
        shutil.copy2(DEST / "catalog.public.json", REPO / "webpage" / "labeling" / "catalog.json")
        shutil.copy2(DEST / "examples.public.json", REPO / "webpage" / "labeling" / "examples.json")
    (DEST / "README.md").write_text(
        f"""# Multiview static-pair batch (curated 300 videos)

- **{len(catalog_items)} full-length videos** (not 16-frame cuts)
- Each video shows **two static cameras** only
- `raw.mp4` + `{overlay_name}` ({args.overlay}; empty seeds fall back to flow)
- 9 reference examples: Bad / Medium / Good × 3 embodiments
- Score **Bad (1) / Medium (2) / Good (3)** once per video

```bash
models/dreamgen/.venv/bin/python scripts/build_assessment_multiview.py --sync-web
```
"""
    )
    print(f"\nwrote {DEST}  ({len(catalog_items)} items, {len(example_items)} examples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
