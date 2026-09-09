#!/usr/bin/env python3
"""Build blind public catalogs for hallucinate-bench 300vid labeling.

Source of truth (HF bucket twanghcmut/hallucinate-bench, revision web):
  300vid/{robot}/key.json                              — videos + segments + machine bands
  300vid/{robot}/{markov,non_markov}/{high,medium,low}/{n}.mp4
  300vid/example/key.json                              — 18 walkthrough clips
  example/{robot}/{low,medium,high}/{1..5}.mp4         — 45 rubric reference clips
  rubric_{vi,en}.md

One catalog item = one VIDEO. A video carries its 16-frame segments; the rater
watches the clip, it pauses at every segment boundary and they score that
segment (key.json expects a flag per segment plus one per video).

Public catalog stays blind (no band / ratio / bundle / violated).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO / "sampling_dataset" / "data_for_web" / "hallucinate_bench"
WEB_ROOT = REPO / "webpage" / "labeling"
BUCKET_BASE = "https://huggingface.co/buckets/twanghcmut/hallucinate-bench/resolve/web"

ROBOTS = ("bimanual", "humanoid", "single_arm")
SPLITS = ("markov", "non_markov")
BANDS = ("high", "medium", "low")
# example folders: low=cleanest→Good, high=extreme→Bad (see example/manifest.json)
RATING = {"low": ("good", 3), "medium": ("medium", 2), "high": ("bad", 1)}
FPS = 16.0
SEG_LEN = 16
# repo-wide vocabulary for the markov / non-markov split
SPLIT_ALIAS = {"markov": "makovian", "non_markov": "non_makovian"}
BAND_TO_SCORE = {"high": 1, "medium": 2, "middle": 2, "low": 3, "clean": 3, "control": 3}
SCORE_RATING = {1: "bad", 2: "medium", 3: "good"}


def bucket_url(relative_path: str) -> str:
    return f"{BUCKET_BASE}/{quote(relative_path, safe='/')}"


def fetch_json(url: str):
    req = urllib.request.Request(
        url, headers={"User-Agent": "vgp-labeling-builder/2.0", "Cache-Control": "no-cache"}
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "vgp-labeling-builder/2.0"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return resp.read()


def write_json(path: Path, value: dict, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=indent) + "\n", encoding="utf-8")


def load_keys(local: Path | None) -> dict:
    """{robot: key.json} for the three 300vid pools."""
    keys = {}
    for robot in ROBOTS:
        if local:
            keys[robot] = json.loads((local / robot / "key.json").read_text(encoding="utf-8"))
        else:
            keys[robot] = fetch_json(bucket_url(f"300vid/{robot}/key.json"))
    return keys


def segments_of(key: dict) -> dict[str, list[dict]]:
    """video path -> segments sorted by index (validated contiguous from 0)."""
    grouped: dict[str, list[dict]] = {}
    for row in key["segments"].values():
        grouped.setdefault(row["video"], []).append(row)
    for path, rows in grouped.items():
        rows.sort(key=lambda r: r["index"])
        for i, row in enumerate(rows):
            if row["index"] != i:
                raise ValueError(f"{path}: segment index gap at {i} (got {row['index']})")
    return grouped


FRAME_CACHE = DATA_ROOT / "300vid" / "frames.json"


def probe_frames(path: str) -> int | None:
    """Real frame count, read over HTTP range requests (no full download)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames,duration,r_frame_rate", "-of", "json",
         bucket_url(f"300vid/{path}")],
        capture_output=True, text=True, timeout=180,
    )
    if out.returncode != 0:
        return None
    try:
        stream = json.loads(out.stdout)["streams"][0]
    except (KeyError, IndexError, json.JSONDecodeError):
        return None
    if stream.get("nb_frames"):
        return int(stream["nb_frames"])
    if stream.get("duration"):
        return round(float(stream["duration"]) * FPS)
    return None


def load_frame_counts(paths: list[str], reprobe: bool = False) -> dict[str, int]:
    """Cached {path: n_frames}. key.json over-reports segments on a few clips."""
    cache = {}
    if FRAME_CACHE.exists() and not reprobe:
        cache = json.loads(FRAME_CACHE.read_text(encoding="utf-8"))
    todo = [p for p in paths if p not in cache]
    if todo:
        print(f"ffprobe {len(todo)} videos …")
        with ThreadPoolExecutor(max_workers=12) as pool:
            for path, frames in zip(todo, pool.map(probe_frames, todo)):
                if frames:
                    cache[path] = frames
        write_json(FRAME_CACHE, cache, indent=None)
    missing = [p for p in paths if p not in cache]
    if missing:
        print(f"  warning: no frame count for {len(missing)} videos, trusting key.json ({missing[:3]})")
    return cache


def clip_segments(segs: list[dict], n_frames: int | None) -> list[dict]:
    """Drop segments past the real end of the file and clamp the last one."""
    if not n_frames:
        return segs
    kept = [dict(s) for s in segs if s["start"] < n_frames]
    if kept:
        kept[-1]["end"] = min(kept[-1]["end"], n_frames - 1)
    return kept


def sort_key(path: str) -> tuple:
    """{robot}/{split}/{band}/{n}.mp4 → stable ordering."""
    _, split, band, name = path.split("/")
    return (SPLITS.index(split), BANDS.index(band), int(name.removesuffix(".mp4")))


def ordered_videos(keys: dict, frames: dict[str, int] | None = None) -> list[tuple[str, str, dict, list[dict]]]:
    """[(robot, path, video_meta, segments)] in catalog order, clipped to the real file length."""
    out = []
    for robot in ROBOTS:
        key = keys[robot]
        by_video = segments_of(key)
        for path in sorted(key["videos"], key=sort_key):
            segs = clip_segments(by_video.get(path) or [], (frames or {}).get(path))
            if not segs:
                raise ValueError(f"{path}: no segments in key.json")
            out.append((robot, path, key["videos"][path], segs))
    return out


def public_segments(segs: list[dict]) -> list[dict]:
    return [
        {
            "index": s["index"],
            "start_frame": s["start"],
            "end_frame": s["end"],
            "n_frames": s["end"] - s["start"] + 1,
            "rateable": True,
        }
        for s in segs
    ]


def build_catalog(keys: dict, frames: dict) -> dict:
    items = []
    for idx, (robot, path, meta, segs) in enumerate(ordered_videos(keys, frames)):
        split = SPLIT_ALIAS[meta["split"]]
        # a handful of clips carry one trailing frame the key does not segment
        covered = segs[-1]["end"] + 1
        n_frames = frames.get(path) or covered
        items.append({
            "id": f"hv{idx:03d}",
            "index": idx,
            "clip_id": path.removesuffix(".mp4").replace("/", "_"),
            "video": bucket_url(f"300vid/{path}"),
            "video_path": path,
            "embodiment": robot,
            "split": split,
            "view": "mixed",
            "task": "Robot motion quality assessment",
            "fps": FPS,
            "n_frames": n_frames,
            "n_frames_segmented": covered,
            "n_segments": len(segs),
            "segment_len_frames": SEG_LEN,
            "segments": public_segments(segs),
            "check": "hallucinate",
        })
    n_segments = sum(it["n_segments"] for it in items)
    per_emb = {r: sum(1 for it in items if it["embodiment"] == r) for r in ROBOTS}
    per_emb_segs = {r: sum(it["n_segments"] for it in items if it["embodiment"] == r) for r in ROBOTS}
    return {
        "mode": "300vid_segments",
        "source": "300vid/{robot}/key.json",
        "n": len(items),
        "n_videos": len(items),
        "n_segments": n_segments,
        "segment_len_frames": SEG_LEN,
        "fps": FPS,
        "videos_per_embodiment": per_emb,
        "segments_per_embodiment": per_emb_segs,
        "check": "hallucinate",
        "note": (
            f"{len(items)} videos / {n_segments} segments of {SEG_LEN} frames. "
            "The clip pauses at every segment boundary; the rater scores that segment."
        ),
        "items": items,
    }


def build_metrics(keys: dict, frames: dict) -> dict:
    """Admin-only machine key: one row per segment, plus per-video rollup."""
    videos = []
    segments = []
    for idx, (robot, path, meta, segs) in enumerate(ordered_videos(keys, frames)):
        vid = f"hv{idx:03d}"
        band = meta.get("band")
        machine_score = BAND_TO_SCORE.get(band or "")
        videos.append({
            "id": vid,
            "index": idx,
            "check": "hallucinate",
            "embodiment": robot,
            "split": SPLIT_ALIAS[meta["split"]],
            "split_src": meta["split"],
            "video_path": path,
            "clip_id": path.removesuffix(".mp4").replace("/", "_"),
            "band": band,
            "machine_score": machine_score,
            "rating": SCORE_RATING.get(machine_score),
            "bundle": meta.get("bundle"),
            "source_video": meta.get("video"),
            "n_segments": len(segs),
            "n_segments_key": meta.get("n_segments"),
            "n_violated": sum(1 for s in segs if s.get("violated")),
            "score_mean": meta.get("score_mean"),
            "score_p90": meta.get("score_p90"),
            "score_max": meta.get("score_max"),
            "frac_violating": meta.get("frac_violating"),
            "real_multiple": meta.get("real_multiple"),
        })
        for s in segs:
            segments.append({
                "id": f"{vid}_s{s['index']:03d}",
                "video_id": vid,
                "video_path": path,
                "segment_index": s["index"],
                "check": "hallucinate",
                "embodiment": robot,
                "split": SPLIT_ALIAS[meta["split"]],
                "band": band,
                "bundle": meta.get("bundle"),
                "start_frame": s["start"],
                "end_frame": s["end"],
                "ratio": s.get("ratio"),
                "ratio_rigidity": s.get("ratio_rigidity"),
                "ratio_jerk": s.get("ratio_jerk"),
                "violated": bool(s.get("violated")),
            })
    thresholds = {r: keys[r].get("segment_threshold") for r in ROBOTS}
    return {
        "mode": "300vid_key",
        "source": "300vid/{robot}/key.json",
        "n": len(segments),
        "n_videos": len(videos),
        "n_segments": len(segments),
        "n_violated": sum(1 for s in segments if s["violated"]),
        "segment_len_frames": SEG_LEN,
        "segment_threshold": thresholds,
        "detectors_used": {r: keys[r].get("detectors_used") for r in ROBOTS},
        "real_median_score": {r: keys[r].get("real_median_score") for r in ROBOTS},
        "check": "hallucinate",
        "note": (
            "Segment machine signal is `violated` (ratio >= 1 vs the real-data p95 threshold). "
            "Video machine label is `band` (high→Bad, medium→Medium, low→Good)."
        ),
        "videos": videos,
        "items": segments,
    }


def build_examples() -> dict:
    """45 rubric reference clips (unchanged pool) + the 18 300vid walkthroughs."""
    manifest = fetch_json(bucket_url("example/manifest.json"))
    declared = set(manifest.get("videos", {}).keys())
    items = []
    for robot in ("bimanual", "humanoid", "single_arm_a1x"):
        for level, (rating, score) in RATING.items():
            for index in range(1, 6):
                relative = f"{robot}/{level}/{index}.mp4"
                if declared and relative not in declared:
                    raise ValueError(f"Example missing from manifest: {relative}")
                items.append({
                    "id": f"example_{robot}_{rating}_{index}",
                    "video": bucket_url(f"example/{relative}"),
                    "embodiment": "single_arm" if robot == "single_arm_a1x" else robot,
                    "rating": rating,
                    "score": score,
                    "level": level,
                    "index": index,
                })
    if len(items) != 45:
        raise ValueError(f"Expected 45 examples (3×3×5), got {len(items)}")
    return {
        "n": len(items),
        "note": "low=cleanest→good, high=extreme→bad (example/manifest.json). 5 clips per embodiment×level.",
        "items": items,
    }


def sync_rubrics() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for lang in ("vi", "en"):
        data = fetch_bytes(bucket_url(f"rubric_{lang}.md"))
        (DATA_ROOT / f"rubric_{lang}.md").write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync-web", action="store_true", help="Copy catalogs and rubrics into webpage/labeling.")
    parser.add_argument("--keys", type=Path, default=None, help="Local dir holding {robot}/key.json (default: fetch).")
    parser.add_argument("--skip-rubrics", action="store_true", help="Do not re-download the rubric markdown.")
    parser.add_argument("--reprobe", action="store_true", help="Re-run ffprobe for every video instead of using the cache.")
    args = parser.parse_args()

    keys = load_keys(args.keys)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for robot in ROBOTS:
        write_json(DATA_ROOT / "300vid" / robot / "key.json", keys[robot], indent=None)

    frames = load_frame_counts([p for _, p, _, _ in ordered_videos(keys)], reprobe=args.reprobe)
    catalog = build_catalog(keys, frames)
    examples = build_examples()
    metrics = build_metrics(keys, frames)
    if not args.skip_rubrics:
        sync_rubrics()

    catalog_path = DATA_ROOT / "catalog.segment.json"
    examples_path = DATA_ROOT / "examples.segment.json"
    metrics_path = DATA_ROOT / "metrics.segment.json"
    write_json(catalog_path, catalog)
    write_json(examples_path, examples)
    write_json(metrics_path, metrics, indent=None)

    if args.sync_web:
        shutil.copy2(catalog_path, WEB_ROOT / "catalog.segment.json")
        shutil.copy2(examples_path, WEB_ROOT / "examples.segment.json")
        shutil.copy2(metrics_path, WEB_ROOT / "metrics.segment.json")
        if not args.skip_rubrics:
            shutil.copy2(DATA_ROOT / "rubric_vi.md", WEB_ROOT / "rubric.segment.vi.md")
            shutil.copy2(DATA_ROOT / "rubric_en.md", WEB_ROOT / "rubric.segment.en.md")

    print(
        f"Built {catalog['n_videos']} hallucinate videos / {catalog['n_segments']} segments "
        f"({metrics['n_violated']} violating) / {examples['n']} examples"
    )
    for robot in ROBOTS:
        print(f"  {robot:11} {catalog['videos_per_embodiment'][robot]:3} videos "
              f"{catalog['segments_per_embodiment'][robot]:5} segments")


if __name__ == "__main__":
    main()
