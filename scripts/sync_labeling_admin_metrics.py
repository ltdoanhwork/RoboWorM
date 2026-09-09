#!/usr/bin/env python3
"""Align labeling catalogs + admin metrics with the videos raters actually see.

Geometry clips follow assets/all_videos.md order (id 00000..). Catalog metadata
must come from those paths — not a stale manifest — so admin embodiment pies
and human↔machine joins match the HF clip behind each id.

Hallucinate comes from the 300vid pool and is only validated here (the builder
owns it): one catalog item per video, one metrics row per 16-frame segment.
Machine bands from key.json: high → Bad (1), medium → Medium (2), low → Good (3).
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "webpage" / "labeling"
ALL_VIDEOS = REPO / "assets" / "all_videos.md"

SCORE_TO_RATING = {1: "bad", 2: "medium", 3: "good"}
BAND_TO_SCORE = {
    "high": 1,
    "middle": 2,
    "medium": 2,
    "low": 3,
    "clean": 3,
    "control": 3,
}


def parse_source(path: str) -> dict:
    parts = path.strip().split("/")
    if len(parts) >= 2 and parts[:2] == ["tmp", "lowquality_multiview"]:
        return {
            "embodiment": "single_arm",
            "model": "dreamgen",
            "split": "",
            "task": "multiview_robot_action",
            "source_path": path.strip(),
        }
    mv = parts.index("multiview")
    embodiment = parts[mv - 2]
    model = parts[mv + 1]
    split = parts[mv + 2]
    task = "multiview_robot_action"
    if model.startswith("ctrlworld") and len(parts) > mv + 3:
        m = re.match(r"episode_(.+?)__\d+", parts[mv + 3], flags=re.I)
        if m:
            task = m.group(1)
    elif model == "dreamgen" and len(parts) > mv + 4 and parts[mv + 3].startswith("iter_"):
        task = parts[mv + 4]
    return {
        "embodiment": embodiment,
        "model": model,
        "split": split,
        "task": task,
        "source_path": path.strip(),
    }


def write_json(path: Path, payload: dict, indent: int | None = 2) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=indent)
    if indent is not None:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def sync_geometry() -> tuple[int, dict]:
    paths = [ln.strip() for ln in ALL_VIDEOS.read_text().splitlines() if ln.strip()]
    catalog = json.loads((WEB / "catalog.json").read_text())
    metrics = json.loads((WEB / "metrics.multiview.json").read_text())
    if len(paths) != len(catalog["items"]) or len(paths) != len(metrics["items"]):
        raise SystemExit(
            f"count mismatch: all_videos={len(paths)} "
            f"catalog={len(catalog['items'])} metrics={len(metrics['items'])}"
        )
    emb_n = {}
    for i, (path, cat, met) in enumerate(zip(paths, catalog["items"], metrics["items"])):
        meta = parse_source(path)
        pid = f"{i:05d}"
        if cat["id"] != pid or met["id"] != pid:
            raise SystemExit(f"id drift at {i}: catalog={cat['id']} metrics={met['id']}")
        cat["embodiment"] = meta["embodiment"]
        cat["index"] = i
        cat["split"] = meta["split"]
        cat["task"] = meta["task"]
        cat["view"] = "multiview"
        met["embodiment"] = meta["embodiment"]
        met["index"] = i
        met["robot_type"] = meta["embodiment"]
        met["task"] = meta["task"]
        met["split"] = meta["split"]
        met["check"] = "geometry"
        met["machine_score"] = int(met["score"])
        met["rating"] = SCORE_TO_RATING[int(met["score"])]
        emb_n[meta["embodiment"]] = emb_n.get(meta["embodiment"], 0) + 1
    catalog["n"] = len(catalog["items"])
    catalog["n_videos"] = catalog["n"]
    catalog["check"] = "geometry"
    catalog["note"] = (
        "Geometry pool raters see. Two static cameras. "
        "Metadata follows assets/all_videos.md (same order as clip ids)."
    )
    metrics["n"] = len(metrics["items"])
    metrics["check"] = "geometry"
    metrics["created_at"] = datetime.now(timezone.utc).isoformat()
    metrics["note"] = (
        "Machine geometry scores for the same 300 catalog.json ids raters label. "
        "score/machine_score: 1 Bad, 2 Medium, 3 Good."
    )
    write_json(WEB / "catalog.json", catalog)
    write_json(WEB / "metrics.multiview.json", metrics)
    return catalog["n"], emb_n


def sync_hallucinate() -> tuple[int, int, dict]:
    """Validate the 300vid pair written by build_hallucination_segment_catalog.py.

    Catalog item = one video; metrics `items` = one row per 16-frame segment and
    metrics `videos` = the per-video machine rollup. Nothing is rewritten here —
    the builder owns the content; this only fails loudly on drift.
    """
    catalog = json.loads((WEB / "catalog.segment.json").read_text())
    metrics = json.loads((WEB / "metrics.segment.json").read_text())
    cat_by_id = {it["id"]: it for it in catalog["items"]}
    met_videos = {it["id"]: it for it in metrics.get("videos", [])}
    if set(cat_by_id) != set(met_videos):
        raise SystemExit(
            f"hallu video id drift: catalog={len(cat_by_id)} metrics={len(met_videos)}"
        )
    seg_per_video: dict[str, int] = {}
    for seg in metrics["items"]:
        if seg["video_id"] not in cat_by_id:
            raise SystemExit(f"segment {seg['id']} has no catalog video")
        seg_per_video[seg["video_id"]] = seg_per_video.get(seg["video_id"], 0) + 1
    bands = {}
    for vid, cat in cat_by_id.items():
        met = met_videos[vid]
        if seg_per_video.get(vid, 0) != cat["n_segments"]:
            raise SystemExit(
                f"{vid}: catalog {cat['n_segments']} segments, metrics {seg_per_video.get(vid, 0)}"
            )
        band = met.get("band") or "unknown"
        if BAND_TO_SCORE.get(band) is None:
            raise SystemExit(f"unknown band {band!r} on {vid}")
        if met["machine_score"] != BAND_TO_SCORE[band]:
            raise SystemExit(f"{vid}: band {band} vs machine_score {met['machine_score']}")
        if met["rating"] != SCORE_TO_RATING[met["machine_score"]]:
            raise SystemExit(f"{vid}: rating drift")
        if any(k in cat for k in ("band", "bundle", "ratio", "violated")):
            raise SystemExit(f"{vid}: machine fields leaked into the public catalog")
        bands[band] = bands.get(band, 0) + 1
    return len(cat_by_id), len(metrics["items"]), bands


def main() -> None:
    geo_n, geo_emb = sync_geometry()
    hallu_v, hallu_s, hallu_bands = sync_hallucinate()
    print(f"geometry {geo_n}  {geo_emb}")
    print(f"hallucinate {hallu_v} videos / {hallu_s} segments  bands {hallu_bands}")


if __name__ == "__main__":
    main()
