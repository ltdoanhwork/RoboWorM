#!/usr/bin/env python3
"""Import the geometry arbiter pass (thanh_check) into the labels table.

Thanh scored all 300 geometry clips offline; that pass is the `score` column of
sampling_dataset/data_for_web/data_for_web.csv. Because it covers the whole
catalog rather than only the conflicts a checker happened to reach, vgp.js uses
it as the arbiter: when the two assigned raters disagree, this score settles the
item (see GEO_ARBITER / resolveUnit).

Row i of the CSV describes catalog id f"{i:05d}". That mapping is not assumed
here — it is re-verified against webpage/labeling/metrics.multiview.json, which
build_multiview_metrics_json.py only writes after checking each rendered clip's
meta.json source_path equals the CSV video_path. If the two disagree we abort,
because a wrong mapping would silently attach a score to a different video.

Usage:
  python3 scripts/import_geo_arbiter_labels.py --dry-run
  python3 scripts/import_geo_arbiter_labels.py
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CSV_REL = Path("sampling_dataset") / "data_for_web" / "data_for_web.csv"
LABELING = REPO / "webpage" / "labeling"
ARBITER = "thanh_check"


def find_csv(explicit: str | None) -> Path:
    """Locate data_for_web.csv.

    sampling_dataset/ is gitignored, so it exists only in the main working tree.
    Running this from a linked worktree or a nested clone (e.g. .main-edit) must
    still find it, hence the parent-directory and worktree candidates.
    """
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"--csv {path} not found")
        return path
    candidates = [REPO / CSV_REL, REPO.parent / CSV_REL]
    try:
        out = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=REPO, capture_output=True, text=True, check=True,
        ).stdout
        candidates += [
            Path(line.split(" ", 1)[1]) / CSV_REL
            for line in out.splitlines()
            if line.startswith("worktree ")
        ]
    except (OSError, subprocess.CalledProcessError):
        pass
    for path in candidates:
        if path.is_file():
            return path
    raise SystemExit(f"{CSV_REL} not found in any worktree; pass --csv PATH")


def supabase_config() -> tuple[str, str]:
    txt = (LABELING / "config.js").read_text()
    url = re.search(r'SUPABASE_URL:\s*"([^"]+)"', txt)
    key = re.search(r'SUPABASE_PUBLISHABLE_KEY:\s*"([^"]+)"', txt)
    if not url or not key:
        raise SystemExit("could not parse SUPABASE_URL / KEY from webpage/labeling/config.js")
    return url.group(1).rstrip("/"), key.group(1)


def arbiter_scores(csv_path: Path) -> list[dict]:
    """CSV score per catalog id, cross-checked against metrics.multiview.json."""
    rows = list(csv.DictReader(csv_path.open()))
    catalog = json.loads((LABELING / "catalog.json").read_text())["items"]
    metrics = json.loads((LABELING / "metrics.multiview.json").read_text())["items"]
    if not (len(rows) == len(catalog) == len(metrics) == 300):
        raise SystemExit(
            f"expected 300 rows everywhere; got csv={len(rows)} "
            f"catalog={len(catalog)} metrics={len(metrics)}"
        )

    out = []
    for i, (row, item, metric) in enumerate(zip(rows, catalog, metrics)):
        item_id = f"{i:05d}"
        if item["id"] != item_id or metric["id"] != item_id:
            raise SystemExit(f"row {i}: id mismatch catalog={item['id']} metrics={metric['id']}")
        # metrics.multiview.json is built only after each clip's rendered
        # meta.json source_path was checked against this same video_path.
        if metric["video_path"] != row["video_path"]:
            raise SystemExit(
                f"row {i}: CSV scores {row['video_path']!r} but the verified "
                f"mapping says id {item_id} is {metric['video_path']!r} — "
                "rebuild metrics.multiview.json"
            )
        if item["embodiment"] != row["robot_type"]:
            raise SystemExit(
                f"row {i}: embodiment {item['embodiment']!r} != robot_type {row['robot_type']!r}"
            )
        score = int(float(row["score"]))
        if score not in (1, 2, 3):
            raise SystemExit(f"row {i}: score {score} outside 1..3")
        out.append({"rater_name": ARBITER, "item_id": item_id, "score": score})
    return out


def post(url: str, key: str, payload: list[dict]) -> None:
    req = urllib.request.Request(
        f"{url}/rest/v1/labels",
        method="POST",
        data=json.dumps(payload).encode(),
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates,return=minimal",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            resp.read()
    except urllib.error.HTTPError as e:
        raise SystemExit(f"upsert failed: {e.code} {e.read().decode()[:500]}") from e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="validate and report, write nothing")
    ap.add_argument("--csv", help="path to data_for_web.csv (default: auto-detect)")
    args = ap.parse_args()

    csv_path = find_csv(args.csv)
    print(f"csv: {csv_path}")
    labels = arbiter_scores(csv_path)
    dist = {s: sum(1 for r in labels if r["score"] == s) for s in (1, 2, 3)}
    print(f"{len(labels)} arbiter labels for {ARBITER!r}; score distribution {dist}")
    print("mapping verified against metrics.multiview.json (clip-level checked)")

    if args.dry_run:
        print("--dry-run: nothing written.")
        return

    url, key = supabase_config()
    for start in range(0, len(labels), 200):
        post(url, key, labels[start : start + 200])
    print(f"upserted {len(labels)} rows into labels as {ARBITER!r}")


if __name__ == "__main__":
    sys.exit(main())
