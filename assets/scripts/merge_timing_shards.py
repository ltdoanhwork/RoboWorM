#!/usr/bin/env python3
"""Merge every timing shard into one table: model x view x method.

WHY THIS EXISTS
---------------
The timing pass is spread over a local GPU plus N Determined shells, and a cell's
10 methods may be split across several shells, so the results live in many
`results.jsonl` files (`dist/<cell>/`, `dist/<cell>__shardK/`, and the local
scratch). Reading any single one of them gives a partial table and — worse — a
speedup computed against a `dense` that may live in a different shard.

This merges by (cell, method), keeps the newest record per pair, and only then
divides by that cell's dense. Records the harness flagged `suspect_*` are dropped
from the table and listed separately, so a number that failed its own sanity
check can never silently become a paper cell.

    python3 assets/scripts/merge_timing_shards.py
    python3 assets/scripts/merge_timing_shards.py --csv metrics/reports/speedup_perchunk.csv
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

METHOD_ORDER = ["dense", "worldcache", "dicache", "fastercache",
                "sito", "itm", "pisa", "radial", "svg1", "svg2"]
METHOD_DISPLAY = {"dense": "dense", "worldcache": "WorldCache", "dicache": "DiCache",
                  "fastercache": "FasterCache", "sito": "SiTo", "itm": "ITM",
                  "pisa": "PISA", "radial": "Radial", "svg1": "SVG1", "svg2": "SVG2"}

# cell id -> (model family, embodiment, view) for the paper axes
CELL_META = {
    "ctrlworld_single_arm_mv":  ("Ctrl-World (3-view)", "single_arm", "MV"),
    "ctrlworld4_bimanual_mv":   ("Ctrl-World (4-view grid)", "bimanual", "MV"),
    "ctrlworld4_humanoid_mv":   ("Ctrl-World (4-view grid)", "humanoid", "MV"),
    "dreamgen_single_arm_mv":   ("DreamGen (14B)", "single_arm", "MV"),
    "dreamgen_bimanual_mv":     ("DreamGen (2B)", "bimanual", "MV"),
    "dreamgen_humanoid_mv":     ("DreamGen (2B)", "humanoid", "MV"),
    "dreamgen_single_arm_sv":   ("DreamGen (2B)", "single_arm", "SV"),
    "dreamgen_bimanual_sv":     ("DreamGen (2B)", "bimanual", "SV"),
    "dreamgen_humanoid_sv":     ("DreamGen (2B)", "humanoid", "SV"),
    "dreamdojo_single_arm_sv":  ("DreamDojo (2B)", "single_arm", "SV"),
    "dreamdojo_bimanual_sv":    ("DreamDojo (2B)", "bimanual", "SV"),
    "dreamdojo_humanoid_sv":    ("DreamDojo (2B)", "humanoid", "SV"),
}


def recompute_from_log(log: str, model: str):
    """Re-derive the per-chunk median from the log.

    A shell may still be running an older copy of the harness (the copies are
    snapshotted per dispatch round), so the stored `generation_s` can come from
    superseded logic. The log itself is the source of truth, so recompute and
    prefer that — this is what caught a cell reporting 1772.6 s (the SUM of 15
    chunks) as its per-chunk time.
    """
    try:
        txt = open(log, errors="replace").read()
    except Exception:
        return None
    if model == "dreamgen":
        vals = [float(m.group(1)) for m in
                re.finditer(r"(?<!LVG )(?<!GR00T LVG )Total inference time:\s*([0-9.]+)", txt)]
        if len(vals) > 1:
            vals = vals[1:]
    else:
        pairs = [(int(n), float(v)) for n, v in
                 re.findall(r"\[ChunkTiming\] chunk (\d+): ([0-9.]+)s", txt)]
        # chunk numbering restarts per segment; every "chunk 1" is a warm-up
        vals = [v for n, v in pairs if n > 1] or [v for _, v in pairs]
    return round(statistics.median(vals), 4) if vals else None


def load_records():
    paths = [str(ROOT / "exp_logs/timing_pass/dist" / "*" / "results.jsonl"),
             "/tmp/uniform_timing/results.jsonl"]
    files = []
    for p in paths:
        files += glob.glob(p) if "*" in p else ([p] if os.path.isfile(p) else [])

    best, suspect = {}, []
    for f in files:
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if str(r.get("status", "")).startswith("suspect"):
                suspect.append(r)
                continue
            if r.get("status") != "ok" or not r.get("generation_s"):
                continue
            fixed = recompute_from_log(r.get("log", ""), r.get("model", ""))
            if fixed:
                if abs(fixed - r["generation_s"]) / fixed > 0.05:
                    r["stale_recorded_s"] = r["generation_s"]
                r["generation_s"] = fixed
            key = (r["cell"], r["method"])
            # keep the record whose log is newest
            prev = best.get(key)
            if prev is None or os.path.getmtime(r["log"]) >= os.path.getmtime(prev["log"]):
                best[key] = r
    return best, suspect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "metrics/reports/speedup_perchunk.csv"))
    args = ap.parse_args()

    best, suspect = load_records()
    cells = sorted({c for c, _ in best}, key=lambda c: (CELL_META.get(c, ("zz",))[0], c))

    rows = []
    print(f"{'model':26s} {'emb':11s} {'view':5s} " +
          " ".join(f"{METHOD_DISPLAY[m][:9]:>9s}" for m in METHOD_ORDER[1:]))
    print("-" * 128)
    for cell in cells:
        model, emb, view = CELL_META.get(cell, (cell, "?", "?"))
        dense = best.get((cell, "dense"))
        line = f"{model:26s} {emb:11s} {view:5s} "
        for m in METHOD_ORDER[1:]:
            r = best.get((cell, m))
            if not r or not dense:
                line += f"{'—':>9s} "
                continue
            sp = dense["generation_s"] / r["generation_s"]
            line += f"{sp:8.2f}x "
            rows.append(dict(cell=cell, model=model, embodiment=emb, view=view,
                             method=METHOD_DISPLAY[m],
                             dense_chunk_s=round(dense["generation_s"], 3),
                             method_chunk_s=round(r["generation_s"], 3),
                             speedup=round(sp, 3),
                             n_chunks=len(r.get("unit_times_s") or []),
                             stale_fixed=("yes" if "stale_recorded_s" in r else "")))
        n_done = sum(1 for m in METHOD_ORDER if (cell, m) in best)
        print(line + f"  [{n_done}/10]")

    if rows:
        import csv
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows -> {args.csv}")

    dense_missing = [c for c in cells if (c, "dense") not in best]
    if dense_missing:
        print(f"\nNO DENSE (speedup not computable): {', '.join(dense_missing)}")
    if suspect:
        print(f"\n{len(suspect)} record(s) FLAGGED by the harness and excluded:")
        for r in suspect:
            print(f"  {r['cell']}/{r['method']}: {r.get('suspect_reason','')}")
    stale = [r for r in rows if r["stale_fixed"]]
    if stale:
        print(f"\n{len(stale)} row(s) recomputed from log (shell ran an older harness):")
        for r in stale[:10]:
            print(f"  {r['cell']}/{r['method']}")


if __name__ == "__main__":
    main()
