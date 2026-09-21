#!/usr/bin/env python3
"""Build a REAL (ground-truth) video gallery for the paper / supplementary drive.

Samples the source LeRobot datasets under ``datasets/`` and lays out a browsable
tree grouped by

    embodiment  x  {makovian, non_makovian}  x  {singleview, multiview}

with ``--per-cell`` (default 5) DIFFERENT tasks per cell, chosen greedily for
diversity (distinct task strings, spread across the available tasks).

Output (default ``assets/drive/real_videos``):

    <embodiment>/<track>/<split>/<NN>_<task_slug>/
        grid.mp4              multi-view 2x2 grid (multiview only, canonical layout)
        <view>.mp4            one file per camera view
        frames/
            <view>_fNNNNN.png     pre-cut frames, evenly spaced, named by frame index
            <view>_strip.png      horizontal filmstrip of those frames (figure-ready)
        info.json             task text, source paths, resolution / fps / #frames

    manifest.json         every clip, machine readable
    INDEX.md              human readable index (task text per cell)

The per-view files are re-encoded (h264, height capped at ``--max-height``, default
480) so the whole gallery stays a few hundred MB instead of tens of GB; use
``--copy`` for byte-identical dataset mp4s. ``grid.mp4`` uses the SAME quadrant
layout as the corresponding ``prepare_*_multiview_*`` script so what reviewers see
matches what the models are fed.

``--frames N`` (default 8) also dumps N evenly-spaced PNGs per video — the same
frame indices across every view of a clip, so a row of stills stays synchronised —
plus a ready-to-drop-in filmstrip per view.

Usage:
    python3 assets/scripts/make_real_video_gallery.py                    # all 12 cells, 5 tasks each
    python3 assets/scripts/make_real_video_gallery.py --per-cell 3
    python3 assets/scripts/make_real_video_gallery.py --frames 12         # more stills per clip
    python3 assets/scripts/make_real_video_gallery.py --frames 0          # videos only
    python3 assets/scripts/make_real_video_gallery.py --only humanoid     # one embodiment
    python3 assets/scripts/make_real_video_gallery.py --no-grid           # skip the 2x2 grids
    python3 assets/scripts/make_real_video_gallery.py --dry-run           # just print the picks
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DS = REPO / "datasets"
OUT_DEFAULT = REPO / "assets" / "drive" / "real_videos"

EMBODIMENTS = ["single_arm", "bimanual", "humanoid"]
TRACKS = ["singleview", "multiview"]
SPLITS = ["makovian", "non_makovian"]

# ---------------------------------------------------------------------------
# Camera layouts (mirror the canonical prepare_* scripts)
# ---------------------------------------------------------------------------

# name -> (label, grid slot)  slot order = TL, TR, BL, BR
LAYOUTS: dict[str, list[tuple[str, str]]] = {
    # single_arm multiview (prepare_single_arm_multiview_dreamgen_batch.py)
    "single_arm/multiview": [
        ("observation.images.exterior_1_left", "exterior 1 (left)"),
        ("observation.images.exterior_2_left", "exterior 2 (left)"),
        ("observation.images.wrist_left", "wrist"),
    ],
    "single_arm/singleview": [
        ("observation.images.global", "global static"),
    ],
    # bimanual multiview (prepare_bimanual_multiview_dreamgen_batch.py, --four_views)
    "bimanual/multiview": [
        ("observation.images.cam_high", "overhead"),
        ("observation.images.cam_low", "low front"),
        ("observation.images.cam_left_wrist", "left wrist"),
        ("observation.images.cam_right_wrist", "right wrist"),
    ],
    "bimanual/singleview": [
        ("observation.images.cam_high", "overhead"),
    ],
    "humanoid/singleview": [
        ("observation.images.ego_view_freq20", "ego view"),
    ],
}

# humanoid multiview has two layouts (prepare_multiview_humanoid_grid_batch.py)
HU_LAYOUT_A = [
    ("observation.images.cam_front_rgb", "front"),
    ("observation.images.cam_head_rgb", "head"),
    ("observation.images.cam_left_wrist_rgb", "left wrist"),
    ("observation.images.cam_right_wrist_rgb", "right wrist"),
]
HU_LAYOUT_B = [
    ("observation.images.cam_high_rgb", "high overhead"),
    ("observation.images.cam_third_view", "third-person side"),
    ("observation.images.cam_left_wrist_rgb", "left wrist"),
    ("observation.images.cam_right_wrist_rgb", "right wrist"),
]

# quadrant size used for the 2x2 grid, per embodiment (matches prepare scripts)
QUAD = {"single_arm": (320, 180), "bimanual": (384, 216), "humanoid": (384, 216)}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Clip:
    embodiment: str
    track: str
    split: str
    task: str  # human readable instruction / task name
    task_key: str  # key used for de-duplication
    episode: int
    root: Path  # dataset root that owns meta/ + videos/
    views: list = field(default_factory=list)  # [(cam_key, label, Path)]
    length: int = 0


def slug(s: str, maxlen: int = 48) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s.strip().lower()).strip("_")
    return (s[:maxlen].rstrip("_") or "task")


def read_meta(root: Path) -> dict[int, dict]:
    f = root / "meta" / "episodes.jsonl"
    if not f.exists():
        return {}
    out = {}
    for line in f.read_text().splitlines():
        if line.strip():
            o = json.loads(line)
            out[int(o["episode_index"])] = o
    return out


def episode_task(meta: dict) -> str:
    t = meta.get("source_task")
    if t:
        return str(t).replace("_", " ")
    tasks = meta.get("tasks") or []
    for cand in tasks:
        if cand and cand.strip().lower() not in {"valid", "invalid"}:
            return str(cand)
    return meta.get("description") or ""


def find_videos(root: Path, cam: str) -> dict[int, Path]:
    out = {}
    for p in sorted((root / "videos").glob(f"chunk-*/{cam}/episode_*.mp4")):
        out[int(p.stem.split("_")[1])] = p
    return out


def probe(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
        "-of", "json", str(path),
    ]
    try:
        st = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
        s = st["streams"][0]
        num, den = s["r_frame_rate"].split("/")
        return {
            "width": s.get("width"),
            "height": s.get("height"),
            "fps": round(float(num) / float(den), 3),
            "frames": int(s.get("nb_frames") or 0),
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Candidate collection
# ---------------------------------------------------------------------------


def hu_mv_layout(task_dir: Path) -> list:
    cams = {d.name for d in (task_dir / "videos").glob("chunk-*/*") if d.is_dir()}
    return HU_LAYOUT_A if "observation.images.cam_front_rgb" in cams else HU_LAYOUT_B


def collect(embodiment: str, track: str, split: str) -> list:
    """Return every usable episode of a cell as a Clip (views resolved)."""
    base = DS / embodiment / track / split
    if not base.exists():
        return []

    task_dirs = [
        d for d in sorted(base.iterdir())
        if d.is_dir() and (d / "meta").is_dir() and (d / "videos").is_dir()
    ]
    clips: list[Clip] = []

    if task_dirs:  # one directory per task (bimanual sv/mv, humanoid mv)
        for td in task_dirs:
            layout = (
                hu_mv_layout(td) if (embodiment == "humanoid" and track == "multiview")
                else LAYOUTS[f"{embodiment}/{track}"]
            )
            meta = read_meta(td)
            per_cam = {cam: find_videos(td, cam) for cam, _ in layout}
            primary = per_cam[layout[0][0]]
            for ep in sorted(primary):
                views = [
                    (cam, lbl, per_cam[cam][ep]) for cam, lbl in layout if ep in per_cam[cam]
                ]
                if len(views) != len(layout):
                    continue
                m = meta.get(ep, {})
                clips.append(Clip(
                    embodiment, track, split,
                    task=episode_task(m) or td.name.replace("_", " "),
                    task_key=td.name, episode=ep, root=td, views=views,
                    length=int(m.get("length") or 0),
                ))
    else:  # flat dataset, tasks live in meta/episodes.jsonl
        layout = LAYOUTS[f"{embodiment}/{track}"]
        meta = read_meta(base)
        per_cam = {cam: find_videos(base, cam) for cam, _ in layout}
        primary = per_cam[layout[0][0]]
        for ep in sorted(primary):
            views = [(cam, lbl, per_cam[cam][ep]) for cam, lbl in layout if ep in per_cam[cam]]
            if len(views) != len(layout):
                continue
            m = meta.get(ep, {})
            task = episode_task(m)
            if not task or task.strip().lower() in {"invalid", "no action", ""}:
                continue
            task = re.sub(r"^(un)?locked waist:\s*", "", task).strip()
            low = task.lower()
            if not task or low.startswith("grid_") or low.startswith("fail"):
                continue
            clips.append(Clip(
                embodiment, track, split, task=task,
                task_key=slug(task, 80), episode=ep, root=base, views=views,
                length=int(m.get("length") or 0),
            ))
    return clips


def pick_diverse(clips: list, n: int, phase: int = 0) -> list:
    """Pick n clips, preferring n DIFFERENT tasks spread over the task list.

    If the cell has fewer than n distinct tasks (e.g. single_arm singleview has
    only 2), the remaining slots are filled round-robin with other episodes of
    those tasks so every cell still ships n clips.

    ``phase`` rotates the spread so two cells drawn from the same underlying task
    pool (bimanual singleview vs multiview) don't select the identical tasks.
    """
    by_task: dict[str, list] = {}
    for c in clips:
        by_task.setdefault(c.task_key, []).append(c)
    keys = sorted(by_task)
    if not keys:
        return []
    for k in keys:  # longest-first, deterministic
        by_task[k] = sorted(by_task[k], key=lambda c: (-c.length, c.episode))

    if len(keys) <= n:
        chosen_keys = keys
    else:
        step = len(keys) / n
        off = (phase * max(1, int(step / 2))) % len(keys)  # half-step shift per phase
        chosen_keys = [keys[(int(i * step) + off) % len(keys)] for i in range(n)]

    # how many slots each chosen task gets (balanced round-robin)
    quota = {k: 0 for k in chosen_keys}
    for i in range(n):
        quota[chosen_keys[i % len(chosen_keys)]] += 1

    out = []
    for k in chosen_keys:
        cands = sorted(by_task[k], key=lambda c: c.episode)
        want = min(quota[k], len(cands))
        if want == 1:  # single representative: a mid-length episode
            best = sorted(cands, key=lambda c: (-c.length, c.episode))
            out.append(best[len(best) // 2])
        else:  # several demos of the same task: spread across the episode list
            step = len(cands) / want
            out += [cands[min(len(cands) - 1, int(i * step))] for i in range(want)]
    return sorted(out, key=lambda c: (c.task.lower(), c.episode))


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def duration_of(path: Path) -> float:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=duration", "-of", "csv=p=0", str(path)]
    try:
        return float(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip())
    except Exception:
        return 0.0


def even(x: int) -> int:
    return x - (x % 2)


def transcode(src: Path, dst: Path, max_h: int, crf: int) -> bool:
    """Re-encode one view: cap height, h264 yuv420p. Keeps the drive small."""
    vf = f"scale=-2:'min({max_h},ih)':flags=lanczos,setsar=1" if max_h else "setsar=1"
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(src),
           "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst)]
    return subprocess.run(cmd).returncode == 0 and dst.exists()


def build_grid(clip: Clip, dst: Path, quad: tuple, crf: int) -> bool:
    """2x2 grid mp4 (all frames) in the canonical slot order; BR black if 3 views.

    The black filler is given an explicit -t (the shortest real view) — an
    unbounded lavfi source would make hstack/vstack run forever.
    """
    qw, qh = quad
    slots = [v[2] for v in clip.views][:4]
    dur = min((duration_of(p) for p in slots), default=0.0)
    if dur <= 0:
        return False
    fps = probe(slots[0]).get("fps") or 30
    while len(slots) < 4:
        slots.append(None)

    inputs, filters = [], []
    for i, p in enumerate(slots):
        if p is None:
            inputs += ["-f", "lavfi", "-t", f"{dur:.3f}",
                       "-i", f"color=c=black:s={qw}x{qh}:r={fps}"]
        else:
            inputs += ["-i", str(p)]
        # force a common frame rate + SAR: mixing rates makes hstack explode the
        # output timebase (and produce a multi-GB file)
        filters.append(f"[{i}:v]fps={fps},scale={qw}:{qh}:flags=lanczos,setsar=1[v{i}]")
    fc = (";".join(filters)
          + ";[v0][v1]hstack=shortest=1[t];[v2][v3]hstack=shortest=1[b]"
          + ";[t][b]vstack=shortest=1[out]")
    cmd = (["ffmpeg", "-nostdin", "-y", "-loglevel", "error"] + inputs
           + ["-filter_complex", fc, "-map", "[out]", "-t", f"{dur:.3f}",
              "-r", str(fps), "-vsync", "cfr",
              "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
              "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst)])
    return subprocess.run(cmd).returncode == 0 and dst.exists()


def frame_indices(total: int, n: int) -> list:
    """n frame indices spread over [0, total-1], always including first and last."""
    if total <= 0:
        return []
    if total <= n:
        return list(range(total))
    if n == 1:
        return [0]
    return [round(i * (total - 1) / (n - 1)) for i in range(n)]


def extract_frames(video: Path, out_dir: Path, n: int, prefix: str) -> list:
    """Dump n evenly-spaced PNG frames from `video` as <prefix>_f<idx>.png.

    One ffmpeg pass with a select filter — much faster than n seeks, and the
    indices are exact frame numbers of the mp4 (so they line up across views).
    """
    total = probe(video).get("frames") or 0
    if not total:
        return []
    fps = probe(video).get("fps") or 0
    idxs = frame_indices(total, n)
    out_dir.mkdir(parents=True, exist_ok=True)
    expr = "+".join(f"eq(n\\,{i})" for i in idxs)
    tmpl = out_dir / f"{prefix}_%02d.png"
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(video),
           "-vf", f"select='{expr}'", "-vsync", "0", str(tmpl)]
    if subprocess.run(cmd).returncode != 0:
        return []
    out = []
    for k, idx in enumerate(idxs, 1):  # rename to the real frame number
        src = out_dir / f"{prefix}_{k:02d}.png"
        if not src.exists():
            continue
        dst = out_dir / f"{prefix}_f{idx:05d}.png"
        src.replace(dst)
        out.append({"file": f"{out_dir.name}/{dst.name}", "frame": idx,
                    "t": round(idx / fps, 3) if fps else None})
    return out


def build_filmstrip(frames: list, base: Path, dst: Path, height: int = 200,
                    gap: int = 4) -> bool:
    """Horizontal contact sheet of the already-extracted frames (paper-ready).

    A thin white gutter is inserted between stills, otherwise consecutive frames
    of a slow manipulation task visually bleed into one another.
    """
    paths = [base / Path(f["file"]).name for f in frames]
    paths = [p for p in paths if p.exists()]
    if len(paths) < 2:
        return False
    inputs = []
    for p in paths:
        inputs += ["-i", str(p)]
    n = len(paths)
    scale = []
    for i in range(n):
        f = f"[{i}:v]scale=-2:{height}:flags=lanczos,setsar=1"
        if i < n - 1:  # white gutter on the right of every tile but the last
            f += f",pad=iw+{gap}:ih:0:0:color=white"
        scale.append(f + f"[v{i}]")
    fc = (";".join(scale) + ";" + "".join(f"[v{i}]" for i in range(n))
          + f"hstack=inputs={n}[out]")
    cmd = (["ffmpeg", "-nostdin", "-y", "-loglevel", "error"] + inputs
           + ["-filter_complex", fc, "-map", "[out]", "-frames:v", "1", str(dst)])
    return subprocess.run(cmd).returncode == 0 and dst.exists()


def write_clip(clip: Clip, out_root: Path, rank: int, args) -> dict:
    name = f"{rank:02d}_{slug(clip.task_key)}"
    d = out_root / clip.embodiment / clip.track / clip.split / name
    d.mkdir(parents=True, exist_ok=True)

    views_meta = []
    for cam, label, src in clip.views:
        short = cam.replace("observation.images.", "")
        dst = d / f"{short}.mp4"
        if not dst.exists():
            if args.copy:
                shutil.copyfile(src, dst)
            elif not transcode(src, dst, args.max_height, args.crf):
                print(f"    [WARN] transcode failed: {src}")
                continue
        views_meta.append({
            "camera": short, "label": label, "file": dst.name,
            "source": str(src.relative_to(REPO)), **probe(dst),
        })

    grid = None
    if not args.no_grid and clip.track == "multiview":
        g = d / "grid.mp4"
        if g.exists() or build_grid(clip, g, QUAD[clip.embodiment], args.crf):
            grid = g.name

    # ---- pre-cut frames -------------------------------------------------------
    frames_meta: dict = {}
    filmstrips: dict = {}
    if args.frames > 0:
        fdir = d / "frames"
        targets = [(v["camera"], d / v["file"]) for v in views_meta]
        if grid:
            targets.append(("grid", d / grid))
        for name, vid in targets:
            fr = extract_frames(vid, fdir, args.frames, name)
            if not fr:
                continue
            frames_meta[name] = fr
            strip = fdir / f"{name}_strip.png"
            if strip.exists() or build_filmstrip(fr, fdir, strip, args.strip_height):
                filmstrips[name] = f"frames/{strip.name}"

    info = {
        "embodiment": clip.embodiment,
        "track": clip.track,
        "split": clip.split,
        "task": clip.task,
        "task_key": clip.task_key,
        "episode_index": clip.episode,
        "num_frames": clip.length,
        "dataset_root": str(clip.root.relative_to(REPO)),
        "grid": grid,
        "views": views_meta,
        "frames": frames_meta,
        "filmstrips": filmstrips,
        "dir": str(d.relative_to(out_root)),
    }
    (d / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def write_index(manifest: list, out: Path):
    lines = [
        "# Real (ground-truth) video gallery",
        "",
        "Sampled from `datasets/<embodiment>/<track>/<split>/` (LeRobot format) and",
        "re-encoded to h264 with the height capped for size. `grid.mp4` is the 2x2",
        "multi-view grid in the SAME slot order the models are conditioned on (mirrors",
        "`scripts/prepare_*_multiview_*_batch.py`); the per-camera files are the same",
        "content unstitched. Each clip dir also has `info.json` (task text, episode",
        "index, source dataset path, resolution / fps / #frames per view).",
        "",
        "Layout: `<embodiment>/<track>/<split>/<NN>_<task>/`, i.e. one cell per",
        "embodiment x {makovian, non_makovian} x {singleview, multiview}.",
        "",
        "Each clip dir also carries pre-cut stills in `frames/`:",
        "`<view>_fNNNNN.png` are evenly-spaced frames named by their real frame index",
        "(the SAME indices for every view and for the grid, so a row of stills stays",
        "synchronised across cameras), and `<view>_strip.png` is a ready-to-include",
        "filmstrip of those frames.",
        "",
        "Where a cell reports fewer distinct tasks than clips, the source dataset itself",
        "has no more tasks in that cell (e.g. single_arm singleview has 2 tasks total,",
        "humanoid multiview makovian has 2 recorded scenarios) — the remaining slots are",
        "different episodes of those tasks, spread across the episode list.",
        "",
        "Regenerate with `python3 assets/scripts/make_real_video_gallery.py`.",
        "",
    ]
    for emb in EMBODIMENTS:
        for track in TRACKS:
            for split in SPLITS:
                cell = [m for m in manifest
                        if m["embodiment"] == emb and m["track"] == track and m["split"] == split]
                if not cell:
                    continue
                ntask = len({m["task_key"] for m in cell})
                lines += [f"## {emb} / {track} / {split}  "
                          f"({len(cell)} clips, {ntask} distinct tasks)", ""]
                lines += ["| # | task | episode | frames | views | stills | dir |",
                          "|---|------|---------|--------|-------|--------|-----|"]
                for i, m in enumerate(cell, 1):
                    vs = ", ".join(v["camera"] for v in m["views"])
                    ns = sum(len(v) for v in (m.get("frames") or {}).values())
                    lines.append(
                        f"| {i} | {m['task']} | {m['episode_index']:06d} | "
                        f"{m['num_frames']} | {vs} | {ns} | `{m['dir']}` |"
                    )
                lines.append("")
    out.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--per-cell", type=int, default=5)
    ap.add_argument("--only", nargs="*", default=EMBODIMENTS, choices=EMBODIMENTS)
    ap.add_argument("--tracks", nargs="*", default=TRACKS, choices=TRACKS)
    ap.add_argument("--no-grid", action="store_true")
    ap.add_argument("--copy", action="store_true",
                    help="byte-copy the dataset mp4s instead of re-encoding (much bigger)")
    ap.add_argument("--max-height", type=int, default=480,
                    help="cap the height of each view (0 = native); default 480")
    ap.add_argument("--crf", type=int, default=23, help="x264 quality, lower = better")
    ap.add_argument("--frames", type=int, default=8,
                    help="pre-cut this many evenly-spaced PNG frames per video (0 = off)")
    ap.add_argument("--strip-height", type=int, default=200,
                    help="row height of the <view>_strip.png filmstrip")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest = []
    for emb in args.only:
        for track in args.tracks:
            for split in SPLITS:
                pool = collect(emb, track, split)
                picks = pick_diverse(pool, args.per_cell, phase=TRACKS.index(track))
                print(f"[{emb}/{track}/{split}] pool={len(pool)} eps, picked {len(picks)}")
                for i, c in enumerate(picks, 1):
                    print(f"    {i}. ep{c.episode:06d}  {len(c.views)}v  {c.length:5d}f  {c.task[:80]}")
                    if args.dry_run:
                        continue
                    manifest.append(write_clip(c, args.out, i, args))

    if args.dry_run:
        return
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_index(manifest, args.out / "INDEX.md")

    def rel(p: Path) -> str:
        try:
            return str(p.relative_to(REPO))
        except ValueError:
            return str(p)

    print(f"\n[ok] {len(manifest)} clips -> {rel(args.out)}")
    print(f"[ok] {rel(args.out / 'manifest.json')}")
    print(f"[ok] {rel(args.out / 'INDEX.md')}")


if __name__ == "__main__":
    main()
