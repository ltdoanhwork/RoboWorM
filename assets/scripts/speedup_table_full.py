#!/usr/bin/env python3
"""Full speedup table: model x view x method, with EXPLICIT provenance per cell.

Three sources, in priority order, because they are NOT the same measurement and
must never be silently averaged:

  1. `chunk`   -- this pass: median per-chunk time, dense control re-run in the
                  same batch, idle GPU. The only source comparable across cells.
  2. `episode` -- earlier runs today, per-EPISODE totals. Superseded: an episode
                  bundles a variable NUMBER of chunks, so the ratio also measures
                  episode length (measured: the same WorldCache config scored
                  0.64x and 1.71x on two episodes of one cell).
  3. `note`    -- numbers transcribed from .cursor/rules/*-insights.mdc, each from
                  its own ad-hoc sweep (own episode count, split, step count).

Every cell is tagged so a reader can see which rows are provisional. Use
`--only chunk` for the publishable subset.

    python3 assets/scripts/speedup_table_full.py
    python3 assets/scripts/speedup_table_full.py --only chunk
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "tables"))

METHODS = ["worldcache", "dicache", "fastercache", "sito", "itm",
           "pisa", "radial", "svg1", "svg2"]
DISP = {"worldcache": "WorldCache", "dicache": "DiCache", "fastercache": "FasterCache",
        "sito": "SiTo", "itm": "ITM", "pisa": "PISA", "radial": "Radial",
        "svg1": "SVG1", "svg2": "SVG2"}

CELLS = [
    ("ctrlworld_single_arm_mv",  "Ctrl-World 3-view", "single_arm", "MV", "ctrlworld", "single_arm", "multiview"),
    ("ctrlworld4_bimanual_mv",   "Ctrl-World grid",   "bimanual",   "MV", "ctrlworld_4view_grid", "bimanual", "multiview"),
    ("ctrlworld4_humanoid_mv",   "Ctrl-World grid",   "humanoid",   "MV", "ctrlworld_4view_grid", "humanoid", "multiview"),
    ("dreamgen_single_arm_mv",   "DreamGen 14B",      "single_arm", "MV", "dreamgen", "single_arm", "multiview"),
    ("dreamgen_bimanual_mv",     "DreamGen 2B",       "bimanual",   "MV", "dreamgen", "bimanual", "multiview"),
    ("dreamgen_humanoid_mv",     "DreamGen 2B",       "humanoid",   "MV", "dreamgen", "humanoid", "multiview"),
    ("dreamgen_single_arm_sv",   "DreamGen 2B",       "single_arm", "SV", "dreamgen", "single_arm", "singleview"),
    ("dreamgen_bimanual_sv",     "DreamGen 2B",       "bimanual",   "SV", "dreamgen", "bimanual", "singleview"),
    ("dreamgen_humanoid_sv",     "DreamGen 2B",       "humanoid",   "SV", "dreamgen", "humanoid", "singleview"),
    ("dreamdojo_single_arm_sv",  "DreamDojo 2B",      "single_arm", "SV", "dreamdojo", "single_arm", "singleview"),
    ("dreamdojo_bimanual_sv",    "DreamDojo 2B",      "bimanual",   "SV", "dreamdojo", "bimanual", "singleview"),
    ("dreamdojo_humanoid_sv",    "DreamDojo 2B",      "humanoid",   "SV", "dreamdojo", "humanoid", "singleview"),
]


def per_chunk_from_log(log, model):
    try:
        txt = open(log, errors="replace").read()
    except Exception:
        return None
    if model == "dreamgen":
        v = [float(m.group(1)) for m in
             re.finditer(r"(?<!LVG )(?<!GR00T LVG )Total inference time:\s*([0-9.]+)", txt)]
        if len(v) > 1:
            v = v[1:]
    else:
        pairs = [(int(n), float(x)) for n, x in
                 re.findall(r"\[ChunkTiming\] chunk (\d+): ([0-9.]+)s", txt)]
        v = [x for n, x in pairs if n > 1] or [x for _, x in pairs]
    return round(statistics.median(v), 4) if v else None


def load_jsonl(paths):
    out = []
    for p in paths:
        for f in (glob.glob(p) if "*" in p else ([p] if os.path.isfile(p) else [])):
            for line in open(f):
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        pass
    return out


def build():
    # ---- source 1: per-chunk (this pass) ----------------------------------
    chunk = {}
    for r in load_jsonl([str(ROOT / "logs/exp_logs/timing_pass/dist/*/results.jsonl"),
                         "/tmp/uniform_timing/results.jsonl"]):
        if str(r.get("status", "")).startswith("suspect") or r.get("status") != "ok":
            continue
        g = per_chunk_from_log(r.get("log", ""), r.get("model", "")) or r.get("generation_s")
        if g:
            chunk[(r["cell"], r["method"])] = g

    # ---- source 2: per-episode (earlier today, superseded) ----------------
    episode = {}
    for r in load_jsonl([str(ROOT / "logs/exp_logs/timing_pass/results_*.jsonl")]):
        if r.get("status") != "ok" or not r.get("generation_s"):
            continue
        episode.setdefault((r["cell"], r["method"]), r["generation_s"])

    # ---- source 3: documented tuning notes --------------------------------
    note = {}
    try:
        # make_speedup_summary.py runs argparse at import time, so a plain import
        # dies on our argv. Execute just its SPEEDUP_LOOKUP construction: read the
        # file and exec the prefix up to the first `def `/`@dataclass`, which is
        # exactly the _add(...) block.
        src_txt = (ROOT / "scripts/tables/make_speedup_summary.py").read_text()
        cut = min((i for i in (src_txt.find("\n@dataclass"), src_txt.find("\ndef parse_cell"))
                   if i > 0), default=len(src_txt))
        ns: dict = {"__file__": str(ROOT / "scripts/tables/make_speedup_summary.py"),
                    "__name__": "_mss_lookup"}
        exec(compile(src_txt[:cut], "make_speedup_summary.py", "exec"), ns)
        note = ns.get("SPEEDUP_LOOKUP", {})
    except Exception as exc:
        print(f"[warn] could not load tuning-note lookup: {exc}", file=sys.stderr)
    return chunk, episode, note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["chunk", "all"], default="all")
    ap.add_argument("--csv", default=str(ROOT / "results/reports/speedup_table_full.csv"))
    args = ap.parse_args()
    chunk, episode, note = build()

    hdr = f"{'model':18s} {'emb':11s} {'view':4s} " + " ".join(f"{DISP[m][:10]:>10s}" for m in METHODS) + "  src"
    print(hdr)
    print("-" * len(hdr))
    csv_rows = []
    for cell, model, emb, view, nmodel, nemb, nview in CELLS:
        dense_c = chunk.get((cell, "dense"))
        dense_e = episode.get((cell, "dense"))
        line = f"{model:18s} {emb:11s} {view:4s} "
        srcs = set()
        for m in METHODS:
            val = src = None
            if dense_c and (cell, m) in chunk:
                val, src = dense_c / chunk[(cell, m)], "chunk"
            elif args.only == "all" and dense_e and (cell, m) in episode:
                val, src = dense_e / episode[(cell, m)], "episode"
            elif args.only == "all" and (m, nmodel, nemb, nview) in note:
                val, src = note[(m, nmodel, nemb, nview)]["speedup"], "note"
            if val is None:
                line += f"{'—':>10s} "
                continue
            mark = {"chunk": "", "episode": "*", "note": "†"}[src]
            srcs.add(src)
            line += f"{val:9.2f}x{mark:1s}"
            csv_rows.append(dict(cell=cell, model=model, embodiment=emb, view=view,
                                 method=DISP[m], speedup=round(val, 3), source=src))
        tag = "+".join(sorted(srcs)) if srcs else "none"
        print(line + f"  {tag}")

    print("\n(no mark) = per-chunk, dense control same batch  |  * = per-EPISODE (superseded)"
          "  |  † = tuning note (own protocol)")
    n = {k: sum(1 for r in csv_rows if r["source"] == k) for k in ("chunk", "episode", "note")}
    print(f"cells: chunk={n['chunk']}  episode={n['episode']}  note={n['note']}"
          f"  total={len(csv_rows)}/{len(CELLS)*len(METHODS)}")

    if csv_rows:
        import csv as _csv
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
            w.writeheader(); w.writerows(csv_rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
