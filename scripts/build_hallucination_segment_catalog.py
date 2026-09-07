#!/usr/bin/env python3
"""Build blind public catalogs for hallucinate-bench rate900 labeling.

Source of truth (HF bucket twanghcmut/hallucinate-bench, revision web):
  rate900/{group}/{1..100}.mp4         — 48 frames (3×16), rate middle segment
  rate900/key.json                     — machine bands / ratios (admin only)
  example/{group}/{low,medium,high}/{1..3}.mp4
  rubric_{vi,en}.md

Public catalog stays blind (no band / ratios / bundle).
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path
from urllib.parse import quote

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO / "sampling_dataset" / "data_for_web" / "hallucinate_bench"
WEB_ROOT = REPO / "webpage" / "labeling"
BUCKET_BASE = "https://huggingface.co/buckets/twanghcmut/hallucinate-bench/resolve/web"
BUCKET_API = "https://huggingface.co/api/buckets/twanghcmut/hallucinate-bench/tree/web"

# rate900 groups (folder names on HF)
EMBODIMENTS = ("bimanual", "humanoid", "single_arm_a1x")
# example folders: low=cleanest→Good, high=extreme→Bad (see example/manifest.json)
RATING = {"low": ("good", 3), "medium": ("medium", 2), "high": ("bad", 1)}
FPS = 16.0
N_FRAMES = 48
SEG_LEN = 16
MIDDLE_SEG = 1  # 0-based; raters judge the middle of three


def bucket_url(relative_path: str) -> str:
    return f"{BUCKET_BASE}/{quote(relative_path, safe='/')}"


def fetch_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "vgp-labeling-builder/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "vgp-labeling-builder/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def write_json(path: Path, value: dict, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=indent) + "\n", encoding="utf-8")


def middle_segment() -> dict:
    start = MIDDLE_SEG * SEG_LEN
    end = start + SEG_LEN - 1
    return {
        "start_frame": start,
        "end_frame": end,
        "n_frames": SEG_LEN,
        "rateable": True,
        "role": "middle",
    }


def all_segments() -> list[dict]:
    """Full clip context; only middle is scored (see rate_target)."""
    segs = []
    for i in range(3):
        start = i * SEG_LEN
        end = start + SEG_LEN - 1
        segs.append({
            "start_frame": start,
            "end_frame": end,
            "n_frames": SEG_LEN,
            "rateable": i == MIDDLE_SEG,
            "role": ("lead", "middle", "trail")[i],
        })
    return segs


def norm_embodiment(embodiment: str) -> str:
    return "single_arm" if embodiment == "single_arm_a1x" else embodiment


def build_catalog(key: dict) -> dict:
    items = []
    idx = 0
    segs = all_segments()
    for embodiment in EMBODIMENTS:
        for number in range(1, 101):
            key_id = f"{embodiment}/{number}"
            if key_id not in key:
                raise KeyError(f"Missing key entry: {key_id}")
            clip_id = f"rate900_{embodiment}_{number:03d}"
            video = bucket_url(f"rate900/{embodiment}/{number}.mp4")
            for seg_i, seg in enumerate(segs):
                items.append({
                    "id": f"h{idx:03d}",
                    "index": idx,
                    "clip_id": clip_id,
                    "video": video,
                    "embodiment": norm_embodiment(embodiment),
                    "source_embodiment": embodiment,
                    "clip": number,
                    "segment_index": seg_i,
                    "view": "mixed",
                    "task": "Robot motion quality assessment",
                    "fps": FPS,
                    "n_frames": N_FRAMES,
                    "rate_target": seg["role"],
                    "segments": [{**row, "rateable": i == seg_i} for i, row in enumerate(segs)],
                    "score_segment_index": seg_i,
                    "check": "hallucinate",
                })
                idx += 1
    if len(items) != 900:
        raise ValueError(f"Expected 900 segments, got {len(items)}")
    return {
        "mode": "rate900_segments",
        "n": len(items),
        "n_segments": len(items),
        "n_clips": 300,
        "check": "hallucinate",
        "note": (
            "900 segments, 300 per embodiment. Each 48-frame clip has lead / middle / trail. "
            "Raters watch the clip and score the highlighted segment."
        ),
        "items": items,
    }


def band_to_score(band: str) -> int | None:
    """Map machine stratum → rater scale (not shown to raters)."""
    return {
        "high": 1,      # Bad
        "middle": 2,    # Medium
        "medium": 2,    # Medium
        "low": 3,       # Good
        "clean": 3,     # Good
        "control": 3,   # teleop / clean reference
    }.get(band)


def build_metrics(key: dict) -> dict:
    items = []
    idx = 0
    for embodiment in EMBODIMENTS:
        for number in range(1, 101):
            row = key[f"{embodiment}/{number}"]
            clip_id = f"rate900_{embodiment}_{number:03d}"
            machine_score = band_to_score(row.get("band") or "")
            for seg_i in range(3):
                items.append({
                    "id": f"h{idx:03d}",
                    "index": idx,
                    "video_id": f"h{idx:03d}",
                    "clip_id": clip_id,
                    "check": "hallucinate",
                    "embodiment": norm_embodiment(embodiment),
                    "source_embodiment": embodiment,
                    "clip": number,
                    "segment_index": seg_i,
                    "band": row.get("band"),
                    "machine_score": machine_score,
                    "rating": {1: "bad", 2: "medium", 3: "good"}.get(machine_score),
                    "selected_by": row.get("selected_by"),
                    "peak_ratio": row.get("peak_ratio"),
                    "ratios": row.get("ratios") or {},
                    "control": bool(row.get("control")),
                    "reference": bool(row.get("reference")),
                    "bundle": row.get("bundle"),
                    "source_video": row.get("video"),
                    "segment_index_src": row.get("segment_index"),
                    "violated": (row.get("band") in ("high", "middle")),
                })
                idx += 1
    return {
        "mode": "rate900_key",
        "source": "rate900/key.json",
        "n": len(items),
        "n_clips": 300,
        "n_violated": sum(1 for it in items if it["violated"]),
        "check": "hallucinate",
        "items": items,
    }


def build_examples() -> dict:
    manifest = fetch_json(bucket_url("example/manifest.json"))
    declared = set(manifest.get("videos", {}).keys())
    items = []
    for embodiment in EMBODIMENTS:
        for level, (rating, score) in RATING.items():
            for index in range(1, 6):
                relative = f"{embodiment}/{level}/{index}.mp4"
                if declared and relative not in declared:
                    raise ValueError(f"Example missing from manifest: {relative}")
                items.append({
                    "id": f"example_{embodiment}_{rating}_{index}",
                    "video": bucket_url(f"example/{relative}"),
                    "embodiment": embodiment,
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
    parser.add_argument(
        "--sync-web",
        action="store_true",
        help="Copy catalogs and rubrics into webpage/labeling.",
    )
    parser.add_argument(
        "--key",
        type=Path,
        default=None,
        help="Optional local key.json (default: fetch from HF).",
    )
    args = parser.parse_args()

    if args.key:
        key = json.loads(args.key.read_text(encoding="utf-8"))
    else:
        key = fetch_json(bucket_url("rate900/key.json"))

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "rate900").mkdir(parents=True, exist_ok=True)
    write_json(DATA_ROOT / "rate900" / "key.json", key)

    catalog = build_catalog(key)
    examples = build_examples()
    metrics = build_metrics(key)
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
        shutil.copy2(DATA_ROOT / "rubric_vi.md", WEB_ROOT / "rubric.segment.vi.md")
        shutil.copy2(DATA_ROOT / "rubric_en.md", WEB_ROOT / "rubric.segment.en.md")

    print(
        f"Built {catalog['n']} hallucinate segments / {examples['n']} examples / "
        f"{metrics['n']} key metrics ({metrics['n_violated']} non-clean bands)"
    )


if __name__ == "__main__":
    main()
