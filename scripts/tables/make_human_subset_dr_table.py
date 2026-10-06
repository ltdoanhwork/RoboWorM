#!/usr/bin/env python3
"""Results table for the two human-rated subsets, scored with the adapted DR metrics.

Subsets (both are human-labelled; they are the correlation study material):

    geometry      300 clips  sampling_dataset/data_for_web/geometry_final_labels.json
    hallucinate   291 clips  sampling_dataset/data_for_web/hallucinate_final_labels.json

Metrics come from the finished DR run (metrics/WorldScore/results/dr_full) -- nothing
is recomputed on GPU. PSNR/SSIM/LPIPS are per clip. FVD is NOT: it is a distance
between two *distributions* of I3D features, so it is undefined for one clip. The DR
run stores each clip's I3D feature windows, so FVD here is aggregated per
human-score band, which is the finest grouping the estimator supports.

Reference kind differs per subset, and that is a property of the data, not a choice:

    geometry     -> vs_gt is available for 213/300 (dense + worldcache multiview
                    predictions keep a rendered GT sibling).
    hallucinate  -> vs_dense only. These are DreamGen singleview flat mp4s and NO
                    GT mp4 exists anywhere for them: the mirrored input/ tree holds
                    only batch_input_*.json and first-frame PNGs. Getting vs_gt for
                    this subset requires rendering GT from the LeRobot datasets first.

Consequence for reading the output: correlations are computed WITHIN a subset, so the
reference kind is constant inside each correlation. Do not compare an absolute PSNR
from geometry against one from hallucinate.

Clips with no metric, and why (all reported in the CSV `status` column):

    geometry     24 augment/ + 39 tmp/lowquality_multiview/ live outside the method
                 trees the DR inventory walks, and have no GT sibling; 18 dense
                 single_arm multiview flat mp4s have no GT either.
    hallucinate  33 never resolved to a local clip (see map_hallucinate_to_dr.py),
                 14 are `dense` which cannot have a vs_dense pair (dense IS the
                 reference), 1 is a corrupt mp4.

Usage:
    python3 scripts/tables/make_human_subset_dr_table.py
"""

from __future__ import annotations

import collections
import csv
import json
import math
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
WEB = REPO / "sampling_dataset/data_for_web"
DR_RUN = REPO / "metrics/WorldScore/results/dr_full"
HALLU_MAP = REPO / "metrics/reports/hallucinate_source_map.json"
OUT_DIR = REPO / "metrics/reports"

# A 400-d covariance estimated from a handful of windows is rank-deficient and its
# FVD reflects that, not the videos. Same guard the DR run uses.
FVD_MIN_WINDOWS = 64
SCORE_LABEL = {3: "good", 2: "medium", 1: "bad"}


# --------------------------------------------------------------------------- #
# DR lookup
# --------------------------------------------------------------------------- #

def load_dr_index() -> dict[str, list[dict]]:
    """rel_pred_path -> [manifest rows], one per reference kind."""
    manifest = json.loads((DR_RUN / "manifest.json").read_text())
    index = collections.defaultdict(list)
    for row in manifest:
        index[os.path.relpath(row["pred"], REPO)].append(row)
    return index


def read_result(pair_id: str) -> dict | None:
    path = DR_RUN / "results" / f"{pair_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def pick_metrics(rows: list[dict], prefer: tuple[str, ...]) -> dict:
    """Best available reference for a clip, in `prefer` order."""
    by_kind = {}
    for row in rows:
        payload = read_result(row["id"])
        if payload is None:
            by_kind.setdefault(row["ref_kind"], {"status": "pair_not_scored"})
            continue
        if payload.get("skipped"):
            by_kind.setdefault(row["ref_kind"], {"status": f'skipped:{payload["skipped"]}'})
            continue
        by_kind[row["ref_kind"]] = {
            "status": "ok",
            "ref_kind": row["ref_kind"],
            "psnr": payload.get("psnr"),
            "ssim": payload.get("ssim"),
            "lpips": payload.get("lpips"),
            "i3d_pred": payload.get("i3d_pred"),
            "i3d_ref": payload.get("i3d_ref"),
        }
    for kind in prefer:
        if by_kind.get(kind, {}).get("status") == "ok":
            return by_kind[kind]
    for kind in prefer:
        if kind in by_kind:
            return by_kind[kind]
    return {"status": "no_dr_pair"}


# --------------------------------------------------------------------------- #
# subsets
# --------------------------------------------------------------------------- #

def geometry_rows(dr: dict) -> list[dict]:
    labels = json.loads((WEB / "geometry_final_labels.json").read_text())
    rows = []
    for clip in labels["clips"]:
        rel = os.path.join("sampling_dataset", clip["video_path"])
        bucket = clip["video_path"].split("/")[0]
        m = pick_metrics(dr.get(rel, []), prefer=("vs_gt", "vs_dense"))
        status = m["status"]
        if status == "no_dr_pair":
            status = {
                "augment": "no_gt_rendered(augment tree)",
                "tmp": "no_gt_rendered(lowquality tree)",
            }.get(bucket, "no_gt_rendered")
        rows.append({
            "subset": "geometry",
            "clip_id": clip["id"],
            "local_path": clip["video_path"],
            "embodiment": clip["embodiment"],
            "model": clip["model"],
            "view": clip["view"],
            "split": clip["split"],
            "method": bucket,
            "human_score": clip.get("human_score"),
            "human_rating": clip.get("human_rating"),
            "human_conflict": clip.get("conflict"),
            "machine_score": clip.get("machine_score"),
            "geometric_consistency": clip.get("geometric_consistency"),
            "pose_stability": clip.get("pose_stability"),
            "ref_kind": m.get("ref_kind"),
            "psnr": m.get("psnr"),
            "ssim": m.get("ssim"),
            "lpips": m.get("lpips"),
            "status": status,
            "_i3d": (m.get("i3d_pred"), m.get("i3d_ref")) if status == "ok" else None,
        })
    return rows


def hallucinate_rows(dr: dict) -> list[dict]:
    labels = {
        v["video_path"]: v
        for v in json.loads((WEB / "hallucinate_final_labels.json").read_text())["videos"]
    }
    mapping = json.loads(HALLU_MAP.read_text())
    resolved, unresolved = mapping["resolved"], mapping["unresolved"]
    unresolved_by_path = {u["web_path"]: u for u in unresolved}

    rows = []
    for web_path, human in labels.items():
        info = resolved.get(web_path)
        base = {
            "subset": "hallucinate",
            "clip_id": human.get("id"),
            "local_path": info["path"] if info else "",
            "embodiment": human.get("embodiment"),
            "model": "dreamgen",
            "view": "singleview",
            "split": human.get("split"),
            "method": info["bundle"] if info else (unresolved_by_path.get(web_path, {}).get("bundle") or ""),
            "human_score": human.get("human_score"),
            "human_rating": human.get("human_rating"),
            "human_conflict": human.get("conflict"),
            "machine_score": None,
            "geometric_consistency": None,
            "pose_stability": None,
            "ref_kind": None, "psnr": None, "ssim": None, "lpips": None,
            "_i3d": None,
        }
        if info is None:
            reason = unresolved_by_path.get(web_path, {}).get("reason", "unresolved")
            base["status"] = f"unresolved:{reason}"
            rows.append(base)
            continue
        m = pick_metrics(dr.get(info["path"], []), prefer=("vs_dense", "vs_gt"))
        status = m["status"]
        if status == "no_dr_pair" and info["bundle"] == "dense":
            status = "dense_is_reference"
        base.update({
            "ref_kind": m.get("ref_kind"),
            "psnr": m.get("psnr"),
            "ssim": m.get("ssim"),
            "lpips": m.get("lpips"),
            "status": status,
            "_i3d": (m.get("i3d_pred"), m.get("i3d_ref")) if status == "ok" else None,
        })
        rows.append(base)
    return rows


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #

def frechet(x: np.ndarray, y: np.ndarray) -> float:
    from scipy import linalg
    mx, my = x.mean(0), y.mean(0)
    cx, cy = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    covmean, _ = linalg.sqrtm(cx.dot(cy), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mx - my) ** 2).sum() + np.trace(cx) + np.trace(cy)
                 - 2 * np.trace(covmean))


def spearman(xs: list[float], ys: list[float]) -> float:
    def rank(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def finite(rows, key):
    out = []
    for r in rows:
        v = r.get(key)
        if isinstance(v, (int, float)) and math.isfinite(v):
            out.append((r, float(v)))
    return out


def summarise(name: str, rows: list[dict]) -> dict:
    scored = [r for r in rows if r["status"] == "ok" and r["human_score"]]
    bands = []
    for score in (3, 2, 1):
        group = [r for r in scored if int(r["human_score"]) == score]
        if not group:
            continue
        pred = [w for r in group if r["_i3d"] for w in (r["_i3d"][0] or [])]
        ref = [w for r in group if r["_i3d"] for w in (r["_i3d"][1] or [])]
        enough = len(pred) >= FVD_MIN_WINDOWS and len(ref) >= FVD_MIN_WINDOWS
        cell = {
            "human_score": score,
            "human_rating": SCORE_LABEL[score],
            "n_clips": len(group),
            "ref_kind": sorted({r["ref_kind"] for r in group if r["ref_kind"]}),
            "n_windows": len(pred),
            "fvd": frechet(np.asarray(pred), np.asarray(ref)) if enough else None,
            "fvd_reason": None if enough else f"<{FVD_MIN_WINDOWS} windows",
        }
        for key in ("psnr", "ssim", "lpips"):
            vals = [v for _, v in finite(group, key)]
            cell[key] = sum(vals) / len(vals) if vals else None
            cell[f"n_{key}"] = len(vals)
        bands.append(cell)

    corr = {}
    for key in ("psnr", "ssim", "lpips"):
        pairs = [(v, int(r["human_score"])) for r, v in finite(scored, key)]
        corr[key] = {
            "spearman_vs_human": spearman([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else None,
            "n": len(pairs),
        }
    return {
        "subset": name,
        "n_clips": len(rows),
        "n_scored": len(scored),
        "status_breakdown": dict(collections.Counter(r["status"] for r in rows).most_common()),
        "by_human_band": bands,
        "correlation": corr,
    }


# --------------------------------------------------------------------------- #

def main() -> None:
    dr = load_dr_index()
    rows = geometry_rows(dr) + hallucinate_rows(dr)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "human_subset_dr_metrics.csv"
    fields = [k for k in rows[0] if not k.startswith("_")]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "protocol": "dr-v1 (PSNR/SSIM/LPIPS per clip; FVD per human-score band)",
        "dr_run": str(DR_RUN.relative_to(REPO)),
        "note": ("FVD is a distribution distance, so it cannot exist per clip; it is "
                 "aggregated over the I3D windows of each human-score band. Reference "
                 "kind differs per subset (geometry: vs_gt, hallucinate: vs_dense) "
                 "because no GT mp4 was ever rendered for the DreamGen singleview "
                 "flat outputs -- compare within a subset only."),
        "subsets": [
            summarise("geometry", [r for r in rows if r["subset"] == "geometry"]),
            summarise("hallucinate", [r for r in rows if r["subset"] == "hallucinate"]),
        ],
    }
    json_path = OUT_DIR / "human_subset_dr_metrics.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n")

    for sub in report["subsets"]:
        print(f'=== {sub["subset"]}: {sub["n_scored"]}/{sub["n_clips"]} scored ===')
        for key, count in sub["status_breakdown"].items():
            print(f'    {key:34s} {count}')
        print(f'    {"human":<10} {"n":>4} {"PSNR":>8} {"SSIM":>8} {"LPIPS":>8} {"FVD":>9}')
        for band in sub["by_human_band"]:
            fvd = f'{band["fvd"]:9.2f}' if band["fvd"] is not None else f'{band["fvd_reason"]:>9}'
            label = '{}({})'.format(band["human_rating"], band["human_score"])
            print(f'    {label:<10} {band["n_clips"]:>4} '
                  f'{band["psnr"]:>8.3f} {band["ssim"]:>8.4f} {band["lpips"]:>8.4f} {fvd}')
        print("    spearman vs human: " + "  ".join(
            f'{k}={v["spearman_vs_human"]:+.4f}' for k, v in sub["correlation"].items()
            if v["spearman_vs_human"] is not None))
        print()
    print(f"wrote {csv_path.relative_to(REPO)}")
    print(f"wrote {json_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
