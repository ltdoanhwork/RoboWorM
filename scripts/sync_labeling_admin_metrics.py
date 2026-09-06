#!/usr/bin/env python3
"""Align labeling catalogs + admin metrics with the videos raters actually see.

Geometry clips follow assets/all_videos.md order (id 00000..). Catalog metadata
must come from those paths — not a stale manifest — so admin embodiment pies
and human↔machine joins match the HF clip behind each id.

Hallucinate clips are rate900_{emb}_{nnn}. Machine bands from key.json:
  high → Bad (1), medium → Medium (2), low → Good (3).
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
        cat["split"] = meta["split"]
        cat["task"] = meta["task"]
        cat["view"] = "multiview"
        met["embodiment"] = meta["embodiment"]
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


def sync_hallucinate() -> tuple[int, dict]:
    catalog = json.loads((WEB / "catalog.segment.json").read_text())
    metrics = json.loads((WEB / "metrics.segment.json").read_text())
    cat_ids = [it["id"] for it in catalog["items"]]
    met_by_id = {it["id"]: it for it in metrics["items"]}
    missing = [i for i in cat_ids if i not in met_by_id]
    extra = [i for i in met_by_id if i not in set(cat_ids)]
    if missing or extra:
        raise SystemExit(f"hallu id mismatch missing={len(missing)} extra={len(extra)}")
    aligned = []
    bands = {}
    for cat in catalog["items"]:
        met = met_by_id[cat["id"]]
        band = met.get("band") or "unknown"
        score = BAND_TO_SCORE.get(band)
        if score is None:
            raise SystemExit(f"unknown band {band!r} on {cat['id']}")
        met["check"] = "hallucinate"
        met["machine_score"] = score
        met["rating"] = SCORE_TO_RATING[score]
        met["violated"] = band in ("high", "middle")
        met["embodiment"] = cat["embodiment"]
        aligned.append(met)
        bands[band] = bands.get(band, 0) + 1
        cat["check"] = "hallucinate"
        cat["view"] = "mixed"
    catalog["n"] = len(catalog["items"])
    catalog["n_segments"] = catalog["n"]
    catalog["check"] = "hallucinate"
    catalog["note"] = (
        "Hallucinate pool raters see. 48-frame clips (3×16). "
        "Score applies to the middle 16 frames."
    )
    metrics["items"] = aligned
    metrics["n"] = len(aligned)
    metrics["check"] = "hallucinate"
    metrics["n_violated"] = sum(1 for it in aligned if it["violated"])
    metrics["created_at"] = datetime.now(timezone.utc).isoformat()
    metrics["note"] = (
        "Machine hallucinate scores for the same catalog.segment.json ids. "
        "high→Bad(1), medium→Medium(2), low→Good(3)."
    )
    write_json(WEB / "catalog.segment.json", catalog)
    write_json(WEB / "metrics.segment.json", metrics, indent=None)
    return catalog["n"], bands


def main() -> None:
    geo_n, geo_emb = sync_geometry()
    hallu_n, hallu_bands = sync_hallucinate()
    print(f"geometry {geo_n}  {geo_emb}")
    print(f"hallucinate {hallu_n}  bands {hallu_bands}")


if __name__ == "__main__":
    main()
