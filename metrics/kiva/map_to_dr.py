#!/usr/bin/env python3
"""Resolve the 291 human-rated hallucinate-bench clips to their LOCAL source
prediction, then join the human label with the DR metrics already computed in
metrics/WorldScore/results/dr_full.

Why this script has to exist
----------------------------
The clips humans rated live on the HF bucket (twanghcmut/hallucinate-bench, rev
`web`) as 512x288 h264 re-renders; `sampling_dataset/data_for_web/
hallucinate_bench/300vid/` holds ONLY key.json, no mp4. So a label cannot be
joined to a metric by path -- the rated file does not exist on disk.

The join that DOES work uses two fields in 300vid/{robot}/key.json that identify
the source exactly:

    bundle  -> acceleration method  (= top-level sampling_dataset/<method>/)
    video   -> episode id, path components joined by '__'

plus `n_frames` from catalog.segment.json as the discriminator, because `video`
alone is ambiguous (a bare `episode_000048` exists under dreamgen singleview,
dreamdojo singleview AND two multiview task dirs, with different lengths).

Matching on (bundle, embodiment, all tokens of `video` present in path, exact
frame count) resolves 258/291 uniquely. Verified by pixel comparison, not just
metadata: web vs local is MAE ~2.7/255 flat across the clip (codec + downscale
noise only), i.e. the same generation.

The 33 that do NOT resolve are genuinely absent, not mis-mapped. Their web
length is 93 frames while only 185/277-frame versions exist locally. 93/185/277
are 1/2/3 DreamGen chunks (93 frames, 1 overlapping). A trim was ruled out by
pixel check: frame 0 matches (MAE 3.9 -- it is the shared GT conditioning frame)
but frame 46 and 92 do not (MAE ~40), so the web clip is a SEPARATE 1-chunk
inference run whose output was not kept. One case is 2301 frames, longer than
any local file. Scoring these requires re-running 1-chunk inference.

Coverage of the resolved 258 against dr_full: 244 have vs_dense PSNR/SSIM/LPIPS.
The other 14 are all `dense` clips, which by definition have no vs_dense pair --
dense IS the reference. They need vs_gt to carry a number.

Usage:
    python3 metrics/kiva/map_to_dr.py                  # write both files
    python3 metrics/kiva/map_to_dr.py --map-only        # skip the join
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "sampling_dataset/data_for_web/hallucinate_bench"
LABELS = REPO / "sampling_dataset/data_for_web/hallucinate_final_labels.json"
DR_RUN = REPO / "metrics/WorldScore/results/dr_full"
SAMPLING = REPO / "sampling_dataset"
OUT_MAP = REPO / "results/reports/hallucinate_source_map.json"
OUT_JOIN = REPO / "results/reports/hallucinate_human_vs_dr.csv"

ROBOTS = ("bimanual", "humanoid", "single_arm")
# never a model's own prediction: GT, side-by-side, per-segment, per-view renders
DERIVED_NAMES = {"full_merged.mp4", "full_gt.mp4", "gt_all_views.mp4", "pred_all_views.mp4"}
DERIVED_PREFIX = ("seg_", "view_")


def probe_frames(path: str) -> int | None:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    try:
        return int(out.stdout.strip().split(",")[0])
    except (ValueError, IndexError):
        return None


def load_targets() -> list[dict]:
    """One entry per rated video: web path + bundle + source episode id."""
    n_frames = {
        it["video_path"]: it["n_frames"]
        for it in json.loads((BENCH / "catalog.segment.json").read_text())["items"]
    }
    targets = []
    for robot in ROBOTS:
        key = json.loads((BENCH / "300vid" / robot / "key.json").read_text())
        for web_path, meta in key["videos"].items():
            targets.append({
                "web_path": web_path,
                "embodiment": robot,
                "split": meta.get("split"),
                "band": meta.get("band"),
                "bundle": meta.get("bundle"),
                "video_id": meta.get("video"),
                "n_frames": n_frames.get(web_path),
            })
    return targets


def index_bundles(bundles: set[str]) -> dict[str, list[str]]:
    """All candidate prediction mp4s under each sampling_dataset/<bundle>/."""
    idx: dict[str, list[str]] = collections.defaultdict(list)
    for bundle in sorted(b for b in bundles if b):
        base = SAMPLING / bundle
        if not base.is_dir():
            continue
        for dirpath, _dirnames, filenames in os.walk(base):
            for name in filenames:
                if not name.endswith(".mp4"):
                    continue
                if name in DERIVED_NAMES or name.startswith(DERIVED_PREFIX):
                    continue
                idx[bundle].append(os.path.join(dirpath, name))
    return idx


def resolve(targets: list[dict]) -> tuple[dict, list[dict]]:
    idx = index_bundles({t["bundle"] for t in targets})
    for t in targets:
        vid = t["video_id"]
        t["candidates"] = [] if not vid else [
            p for p in idx.get(t["bundle"], [])
            if f'/{t["embodiment"]}/' in p and all(tok in p for tok in vid.split("__"))
        ]

    every = sorted({p for t in targets for p in t["candidates"]})
    print(f"ffprobe {len(every)} candidates ...", flush=True)
    with ThreadPoolExecutor(32) as pool:
        frames = dict(zip(every, pool.map(probe_frames, every)))

    resolved, unresolved = {}, []
    for t in targets:
        exact = [p for p in t["candidates"] if frames.get(p) == t["n_frames"]]
        if len(exact) == 1:
            resolved[t["web_path"]] = {
                "path": os.path.relpath(exact[0], REPO),
                "bundle": t["bundle"],
                "embodiment": t["embodiment"],
                "split": t["split"],
                "band": t["band"],
                "video_id": t["video_id"],
                "n_frames": t["n_frames"],
            }
        else:
            unresolved.append({
                "web_path": t["web_path"],
                "bundle": t["bundle"],
                "embodiment": t["embodiment"],
                "video_id": t["video_id"],
                "web_n_frames": t["n_frames"],
                "reason": "ambiguous" if exact else "no_local_clip_with_that_length",
                "candidates": [
                    {"path": os.path.relpath(p, REPO), "n_frames": frames.get(p)}
                    for p in t["candidates"][:4]
                ],
            })
    return resolved, unresolved


def join_dr(resolved: dict) -> list[dict]:
    """Attach human label + DR metrics per resolved clip."""
    labels = {
        v["video_path"]: v
        for v in json.loads(LABELS.read_text())["videos"]
    }
    manifest = json.loads((DR_RUN / "manifest.json").read_text())
    by_pred = collections.defaultdict(list)
    for row in manifest:
        by_pred[os.path.relpath(row["pred"], REPO)].append(row)

    rows = []
    for web_path, info in sorted(resolved.items()):
        human = labels.get(web_path, {})
        rec = {
            "web_path": web_path,
            "embodiment": info["embodiment"],
            "split": info["split"],
            "band": info["band"],
            "method": info["bundle"],
            "local_path": info["path"],
            "n_frames": info["n_frames"],
            "human_score": human.get("human_score"),
            "human_rating": human.get("human_rating"),
            "human_consensus": human.get("consensus"),
            "human_conflict": human.get("conflict"),
            "ref_kind": None, "psnr": None, "ssim": None, "lpips": None,
            "dr_status": "no_dr_pair",
        }
        for dr in by_pred.get(info["path"], []):
            result = DR_RUN / "results" / f'{dr["id"]}.json'
            if not result.exists():
                rec["dr_status"] = "pair_not_scored"
                continue
            payload = json.loads(result.read_text())
            if payload.get("skipped"):
                rec["dr_status"] = f'skipped:{payload["skipped"]}'
                continue
            rec.update({
                "ref_kind": dr["ref_kind"],
                "psnr": payload.get("psnr"),
                "ssim": payload.get("ssim"),
                "lpips": payload.get("lpips"),
                "dr_status": "ok",
            })
            break
        if rec["dr_status"] == "no_dr_pair" and info["bundle"] == "dense":
            # dense is the vs_dense reference, so a vs_dense pair cannot exist
            rec["dr_status"] = "dense_is_reference"
        rows.append(rec)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-only", action="store_true", help="resolve paths, skip the DR join")
    args = ap.parse_args()

    targets = load_targets()
    resolved, unresolved = resolve(targets)
    OUT_MAP.parent.mkdir(parents=True, exist_ok=True)
    OUT_MAP.write_text(json.dumps({
        "source": "300vid/{robot}/key.json (bundle + video) x catalog.segment.json (n_frames)",
        "note": ("Rated mp4s are 512x288 web re-renders on the HF bucket; only key.json is "
                 "local. Resolution is by bundle + episode id tokens + EXACT frame count, "
                 "pixel-verified at MAE ~2.7/255 (codec noise only)."),
        "n_rated": len(targets),
        "n_resolved": len(resolved),
        "n_unresolved": len(unresolved),
        "resolved": resolved,
        "unresolved": unresolved,
    }, indent=2) + "\n")
    print(f"rated      : {len(targets)}")
    print(f"resolved   : {len(resolved)}")
    print(f"unresolved : {len(unresolved)}")
    print(f"wrote {OUT_MAP.relative_to(REPO)}")
    if args.map_only:
        return

    rows = join_dr(resolved)
    fields = list(rows[0])
    with OUT_JOIN.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    status = collections.Counter(r["dr_status"] for r in rows)
    print()
    for key, count in status.most_common():
        print(f"  {key:24s} {count}")
    print(f"wrote {OUT_JOIN.relative_to(REPO)}")


if __name__ == "__main__":
    main()
