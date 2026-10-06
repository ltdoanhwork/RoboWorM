#!/usr/bin/env python3
"""Build blind public catalogs for hallucinate-bench segment labeling."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from urllib.parse import quote


REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "sampling_dataset" / "data_for_web" / "hallucinate_bench"
WEB_ROOT = REPO / "webpage" / "labeling"
BUCKET_BASE = "https://huggingface.co/buckets/twanghcmut/hallucinate-bench/resolve/web"
EMBODIMENTS = ("bimanual", "humanoid", "single_arm")
RATING = {"low": ("bad", 1), "medium": ("medium", 2), "high": ("good", 3)}
DETECTORS = ("rigidity", "jerk", "teleport", "joint_limit", "self_collision")


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def bucket_url(relative_path: str) -> str:
    return f"{BUCKET_BASE}/{quote(relative_path, safe='/')}"


def task_from_source(source: str | None) -> str:
    """Create useful prompt without leaking source bundle or quality stratum."""
    if not source:
        return "Robot motion quality assessment"
    stem = Path(source).stem.split("__episode_", 1)[0]
    stem = re.sub(r"^\d+_", "", stem)
    if re.fullmatch(r"episode_?\d+", stem, flags=re.IGNORECASE):
        return "Robot motion quality assessment"
    task = re.sub(r"[_-]+", " ", stem).strip()
    return task or "Robot motion quality assessment"


def detector_fields(segment: dict) -> tuple[dict, bool]:
    """Copy calibrated detector outputs. Public catalog stays blind."""
    out = {}
    any_violated = False
    for name in DETECTORS:
        raw = segment.get(name)
        if not isinstance(raw, dict):
            continue
        rec = {
            "value": raw.get("value"),
            "ratio": raw.get("ratio"),
            "violated": bool(raw.get("violated")),
        }
        out[name] = rec
        if rec["violated"]:
            any_violated = True
    return out, any_violated


def public_segments(raw_segments: list[dict]) -> list[dict]:
    result = []
    for segment in raw_segments:
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        result.append({
            "start_frame": start,
            "end_frame": end,
            "n_frames": int(segment.get("n_frames", end - start + 1)),
        })
    return result


def build_catalog() -> dict:
    batch_root = DATA_ROOT / "300-prune"
    manifest = read_json(batch_root / "manifest.json")
    manifest_videos = manifest.get("videos", {})
    items = []

    for embodiment in EMBODIMENTS:
        metadata = read_json(batch_root / embodiment / "segments.json")
        videos = metadata.get("videos", metadata)
        for number in range(1, 101):
            filename = f"{number}.mp4"
            local_video = batch_root / embodiment / filename
            if not local_video.is_file():
                raise FileNotFoundError(local_video)
            raw = videos.get(filename) or videos.get(str(number))
            if raw is None:
                raise KeyError(f"Missing segment metadata: {embodiment}/{filename}")
            segments = public_segments(raw.get("segments", []))
            if not segments:
                raise ValueError(f"No segments: {embodiment}/{filename}")
            manifest_row = manifest_videos.get(f"{embodiment}/{filename}", {})
            source = raw.get("source") or manifest_row.get("source_video")
            items.append({
                "id": f"hallucination_{embodiment}_{number:03d}",
                "video": bucket_url(f"300-prune/{embodiment}/{filename}"),
                "embodiment": embodiment,
                "view": raw.get("view", "singleview"),
                "task": task_from_source(source),
                "fps": float(raw["fps"]),
                "n_frames": int(raw["n_frames"]),
                "segments": segments,
            })

    if len(items) != 300:
        raise ValueError(f"Expected 300 videos, got {len(items)}")
    return {
        "mode": "segment_hallucination",
        "n": len(items),
        "n_segments": sum(len(item["segments"]) for item in items),
        "items": items,
    }


def build_metrics() -> dict:
    """Admin-only machine scores, same ids as human labels (`…_s000`)."""
    batch_root = DATA_ROOT / "300-prune"
    items = []
    for embodiment in EMBODIMENTS:
        metadata = read_json(batch_root / embodiment / "segments.json")
        videos = metadata.get("videos", metadata)
        for number in range(1, 101):
            raw = videos.get(f"{number}.mp4") or videos.get(str(number))
            if raw is None:
                raise KeyError(f"Missing segment metadata: {embodiment}/{number}")
            video_id = f"hallucination_{embodiment}_{number:03d}"
            for index, segment in enumerate(raw.get("segments", [])):
                detectors, violated = detector_fields(segment)
                start = int(segment["start_frame"])
                end = int(segment["end_frame"])
                items.append({
                    "id": f"{video_id}_s{index:03d}",
                    "video_id": video_id,
                    "embodiment": embodiment,
                    "index": index,
                    "start_frame": start,
                    "end_frame": end,
                    "n_frames": int(segment.get("n_frames", end - start + 1)),
                    "violated": violated,
                    "n_violated_video": int(raw.get("n_violated") or 0),
                    "detectors": detectors,
                })
    if not items:
        raise ValueError("No metric segments")
    return {
        "mode": "segment_metrics",
        "n": len(items),
        "detectors": list(DETECTORS),
        "n_violated": sum(1 for item in items if item["violated"]),
        "items": items,
    }


def build_examples() -> dict:
    example_root = DATA_ROOT / "example"
    manifest = read_json(example_root / "manifest.json")
    declared = manifest.get("videos", manifest)
    declared_paths = set(declared if isinstance(declared, dict) else declared)
    items = []

    for path in sorted(example_root.glob("*/*/*.mp4")):
        relative = path.relative_to(example_root).as_posix()
        embodiment, level = relative.split("/")[:2]
        if level not in RATING:
            continue
        if declared_paths and relative not in declared_paths:
            raise ValueError(f"Example missing from manifest: {relative}")
        rating, score = RATING[level]
        items.append({
            "id": f"example_{embodiment}_{rating}_{path.stem}",
            "video": bucket_url(f"example/{relative}"),
            "embodiment": "single_arm" if embodiment == "singlearm" else embodiment,
            "rating": rating,
            "score": score,
            "index": int(path.stem),
        })

    if len(items) != 45:
        raise ValueError(f"Expected 45 examples, got {len(items)}")
    return {"n": len(items), "items": items}


def write_json(path: Path, value: dict, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=indent) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync-web", action="store_true",
                        help="Copy catalogs and rubrics into webpage/labeling.")
    args = parser.parse_args()
    catalog_path = DATA_ROOT / "catalog.segment.json"
    examples_path = DATA_ROOT / "examples.segment.json"
    metrics_path = DATA_ROOT / "metrics.segment.json"
    write_json(catalog_path, build_catalog())
    write_json(examples_path, build_examples())
    write_json(metrics_path, build_metrics(), indent=None)

    if args.sync_web:
        shutil.copy2(catalog_path, WEB_ROOT / "catalog.segment.json")
        shutil.copy2(examples_path, WEB_ROOT / "examples.segment.json")
        shutil.copy2(metrics_path, WEB_ROOT / "metrics.segment.json")
        shutil.copy2(DATA_ROOT / "rubric_vi.md", WEB_ROOT / "rubric.segment.vi.md")
        shutil.copy2(DATA_ROOT / "rubric_en.md", WEB_ROOT / "rubric.segment.en.md")

    catalog = read_json(catalog_path)
    examples = read_json(examples_path)
    metrics = read_json(metrics_path)
    print(f"Built {catalog['n']} videos / {catalog['n_segments']} segments / "
          f"{examples['n']} examples / {metrics['n']} metrics "
          f"({metrics['n_violated']} machine-violated)")


if __name__ == "__main__":
    main()
