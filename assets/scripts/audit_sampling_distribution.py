#!/usr/bin/env python3
"""Audit the sampling_dataset episode distribution against the benchmark design scope.

Design scope (the "excel"): per (embodiment, view, model) cell, counted as
makovian + non_makovian.

    bimanual   MV : DreamGen 150+150 | DreamDojo   N/a  | Ctrl-World 150+150
    bimanual   SV : DreamGen 150+150 | DreamDojo 150+150| Ctrl-World   N/a
    humanoid   MV : DreamGen  98+150 | DreamDojo   N/a  | Ctrl-World  98+150
    humanoid   SV : DreamGen 130+130 | DreamDojo 118+98 | Ctrl-World   N/a
    single_arm MV : DreamGen 128+129 | DreamDojo   N/a  | Ctrl-World 128+129
    single_arm SV : DreamGen 120+120 | DreamDojo 120+120| Ctrl-World   N/a

  per model : DreamGen 1605 | DreamDojo 756 | Ctrl-World 805  -> dense 3166
  per method: itm 2410, sito 2410 (no DreamDojo)
              fastercache 2361 (no Ctrl-World)
              worldcache/dicache/svg1/svg2/radial/pisa 3166 (full)
  groups    : Dense 3166 | caching 8693 | sparse attn 12664 | token reduction 4820
              TOTAL 29343

Counting rules
  - An episode output is either a directory containing `full_pred.mp4`, or a flat
    per-episode `*.mp4` (DreamGen multiview/singleview gr00t batch layout).
  - Auxiliary videos (full_gt / full_merged / gt_all_views / pred_all_views /
    seg_* / view_*) are never counted as episodes.
  - Ctrl-World has two tree variants (`ctrlworld`, `ctrlworld_4view_grid`); the
    design scope has ONE Ctrl-World column, so a cell is satisfied by whichever
    variant a method actually produced (the larger one is reported).
  - Known non-benchmark leftovers are excluded and reported separately, so the
    totals reflect benchmark content only:
      * `_fps_compare/`                      debug artifacts
      * `.../singleview/dreamgen/*/iter_000090000/`
            stale tree from the old wrong output tag (canonical is
            `iter_000090000_static16fps`)
      * `.../single_arm/output/singleview/dreamdojo/*/iter_000100000/`
            bimanual's iter tag sitting in the single_arm tree

Usage:  python3 scripts/audit_sampling_distribution.py [--show-extras]
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "sampling_dataset",
)

AUX_EXACT = {
    "full_gt.mp4", "full_merged.mp4", "gt_all_views.mp4",
    "pred_all_views.mp4", "full_pred.mp4",
}
AUX_PREFIX = ("seg_", "view_")
VIEW_TAG = {"multiview": "MV", "singleview": "SV"}

SCOPE = {
    ("bimanual", "MV", "dreamgen"): (150, 150),
    ("bimanual", "MV", "ctrlworld"): (150, 150),
    ("bimanual", "SV", "dreamgen"): (150, 150),
    ("bimanual", "SV", "dreamdojo"): (150, 150),
    ("humanoid", "MV", "dreamgen"): (98, 150),
    ("humanoid", "MV", "ctrlworld"): (98, 150),
    ("humanoid", "SV", "dreamgen"): (130, 130),
    ("humanoid", "SV", "dreamdojo"): (118, 98),
    ("single_arm", "MV", "dreamgen"): (128, 129),
    ("single_arm", "MV", "ctrlworld"): (128, 129),
    ("single_arm", "SV", "dreamgen"): (120, 120),
    ("single_arm", "SV", "dreamdojo"): (120, 120),
}
MODEL_TOTAL = {"dreamgen": 1605, "dreamdojo": 756, "ctrlworld": 805}
METHOD_TOTAL = {
    "itm": 2410, "sito": 2410, "worldcache": 3166, "fastercache": 2361,
    "dicache": 3166, "svg2": 3166, "svg1": 3166, "radial": 3166, "pisa": 3166,
}
GROUPS = [
    ("Dense (no acceleration)", ["dense"], 3166),
    ("Acceleration caching", ["worldcache", "dicache", "fastercache"], 8693),
    ("Sparse attention", ["svg1", "svg2", "radial", "pisa"], 12664),
    ("Token reduction", ["sito", "itm"], 4820),
]
EXCLUDE_SUBSTR = (
    "/_fps_compare/",
    "/singleview/dreamgen/makovian/iter_000090000/",
    "/singleview/dreamgen/non_makovian/iter_000090000/",
    "/single_arm/output/singleview/dreamdojo/makovian/iter_000100000/",
    "/single_arm/output/singleview/dreamdojo/non_makovian/iter_000100000/",
)


def is_aux(name: str) -> bool:
    return name in AUX_EXACT or name.startswith(AUX_PREFIX)


def family(model_dir: str) -> str:
    return "ctrlworld" if model_dir.startswith("ctrlworld") else model_dir


def split_of(parts) -> str:
    for p in parts:
        if p in ("makovian", "non_makovian"):
            return p
    return "?"


def scan(method: str):
    """-> (counts, extras) where
    counts[(emb, view, model_dir)][split] = n episodes
    extras = n episode-like files excluded as non-benchmark leftovers
    """
    counts = defaultdict(lambda: defaultdict(set))
    extras = 0
    base = os.path.join(ROOT, method)
    if not os.path.isdir(base):
        return counts, extras
    for emb in sorted(os.listdir(base)):
        outdir = os.path.join(base, emb, "output")
        if not os.path.isdir(outdir):
            continue
        for view in sorted(os.listdir(outdir)):
            if view not in VIEW_TAG:
                continue
            vdir = os.path.join(outdir, view)
            if not os.path.isdir(vdir):
                continue
            for model_dir in sorted(os.listdir(vdir)):
                mdir = os.path.join(vdir, model_dir)
                if not os.path.isdir(mdir):
                    continue
                for dirpath, dirnames, filenames in os.walk(mdir):
                    rel = os.path.relpath(dirpath, mdir)
                    parts = [] if rel == "." else rel.split(os.sep)
                    if "full_pred.mp4" in filenames:
                        hits, dirnames[:] = [rel], []
                    else:
                        hits = [
                            os.path.join(rel, f) for f in filenames
                            if f.endswith(".mp4") and not is_aux(f)
                        ]
                    if not hits:
                        continue
                    probe = dirpath.replace(os.sep, "/") + "/"
                    if any(s in probe for s in EXCLUDE_SUBSTR):
                        extras += len(hits)
                        continue
                    key = (emb, VIEW_TAG[view], model_dir)
                    counts[key][split_of(parts)].update(hits)
    return counts, extras


def best_variant(counts, emb, view, fam):
    """Pick the tree variant that actually holds this cell's episodes."""
    best, best_n = None, -1
    for key in counts:
        if key[0] == emb and key[1] == view and family(key[2]) == fam:
            n = sum(len(v) for v in counts[key].values())
            if n > best_n:
                best, best_n = key, n
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-extras", action="store_true",
                    help="list the excluded non-benchmark leftovers per method")
    ap.add_argument("--json", type=str, default=None,
                    help="also write the audited counts to this path, so paper "
                         "figures read the SAME numbers this audit verifies")
    args = ap.parse_args()

    methods = ["dense"] + list(METHOD_TOTAL)
    data = {m: scan(m) for m in methods}
    ok = True

    # ---------- 1. dense per-cell grid ----------
    print("=" * 84)
    print("1) DENSE — embodiment x view x model  (makovian + non_makovian)")
    print("=" * 84)
    counts, _ = data["dense"]
    print(f"{'embodiment':<12}{'view':<5}{'DreamGen':>14}{'DreamDojo':>14}{'Ctrl-World':>14}")
    print("-" * 84)
    model_sum = defaultdict(int)
    dense_cells = {}
    for emb in ("bimanual", "humanoid", "single_arm"):
        for view in ("MV", "SV"):
            cells = []
            for fam in ("dreamgen", "dreamdojo", "ctrlworld"):
                exp = SCOPE.get((emb, view, fam))
                if exp is None:
                    cells.append("N/a")
                    continue
                key = best_variant(counts, emb, view, fam)
                a = len(counts[key].get("makovian", ())) if key else 0
                b = len(counts[key].get("non_makovian", ())) if key else 0
                model_sum[fam] += a + b
                dense_cells[f"{emb}|{view}|{fam}"] = {
                    "embodiment": emb, "view": view, "model": fam,
                    "makovian": a, "non_makovian": b, "total": a + b,
                    "scope_makovian": exp[0], "scope_non_makovian": exp[1],
                }
                mark = "" if (a, b) == exp else " !"
                if (a, b) != exp:
                    ok = False
                cells.append(f"{a}+{b}{mark}")
            print(f"{emb:<12}{view:<5}{cells[0]:>14}{cells[1]:>14}{cells[2]:>14}")
    print()
    print(f"  {'model':<14}{'actual':>8}{'scope':>8}")
    for fam, exp in MODEL_TOTAL.items():
        got = model_sum[fam]
        if got != exp:
            ok = False
        print(f"  {fam:<14}{got:>8}{exp:>8}   {'OK' if got == exp else 'MISMATCH'}")
    dense_total = sum(model_sum.values())
    if dense_total != 3166:
        ok = False
    print(f"  {'TOTAL':<14}{dense_total:>8}{3166:>8}   "
          f"{'OK' if dense_total == 3166 else 'MISMATCH'}")

    # ---------- 2. per method ----------
    print()
    print("=" * 84)
    print("2) PER ACCELERATION METHOD")
    print("=" * 84)
    print(f"{'method':<14}{'actual':>8}{'scope':>8}{'extras':>8}   notes")
    print("-" * 84)
    totals = {"dense": dense_total}
    method_cells = {}
    for m, exp in METHOD_TOTAL.items():
        counts, extras = data[m]
        got, gaps = 0, []
        for (emb, view, fam), want in SCOPE.items():
            key = best_variant(counts, emb, view, fam)
            n = sum(len(v) for v in counts[key].values()) if key else 0
            got += n
            method_cells[f"{m}|{emb}|{view}|{fam}"] = {
                "method": m, "embodiment": emb, "view": view, "model": fam,
                "total": n, "scope_total": sum(want),
            }
            if n == 0:
                gaps.append(f"no {emb}/{view}/{fam}")
            elif n < sum(want):
                gaps.append(f"{emb}/{view}/{fam} {n}/{sum(want)}")
        totals[m] = got
        if got != exp:
            ok = False
        note = "; ".join(gaps) if gaps else "complete"
        print(f"{m:<14}{got:>8}{exp:>8}{extras:>8}   {note}")
        if args.show_extras and extras:
            print(f"{'':<14}  (excluded {extras} non-benchmark leftover files)")

    # ---------- 3. groups ----------
    print()
    print("=" * 84)
    print("3) GROUP TOTALS")
    print("=" * 84)
    grand = 0
    for name, members, exp in GROUPS:
        got = sum(totals.get(x, 0) for x in members)
        grand += got
        if got != exp:
            ok = False
        print(f"  {name:<26}{got:>8}{exp:>8}   {'OK' if got == exp else 'MISMATCH'}")
    if grand != 29343:
        ok = False
    print(f"  {'TOTAL':<26}{grand:>8}{29343:>8}   "
          f"{'OK' if grand == 29343 else 'MISMATCH'}")

    print()
    print("=" * 84)
    print("VERDICT: distribution MATCHES the design scope exactly."
          if ok else "VERDICT: distribution DIFFERS from the design scope (see ! / MISMATCH).")
    print("=" * 84)

    if args.json:
        import json
        for k, v in dense_cells.items():
            method_cells[f"dense|{k}"] = {
                "method": "dense", "embodiment": v["embodiment"], "view": v["view"],
                "model": v["model"], "total": v["total"],
                "scope_total": v["scope_makovian"] + v["scope_non_makovian"],
            }
        payload = {
            "generated_by": "assets/scripts/audit_sampling_distribution.py",
            "purpose": "Audited rollout counts. Figures MUST read these, not a raw "
                       "directory walk: this applies the non-benchmark exclusions "
                       "(stale iter tags, _fps_compare) and merges the two "
                       "Ctrl-World tree variants into one model column.",
            "scope_matches_design": ok,
            "dense_total": dense_total,
            "grand_total": grand,
            "model_totals": {f: model_sum[f] for f in MODEL_TOTAL},
            "method_totals": totals,
            "groups": [{"name": n, "members": mem, "total":
                        sum(totals.get(x, 0) for x in mem)} for n, mem, _ in GROUPS],
            "dense_cells": list(dense_cells.values()),
            "method_cells": list(method_cells.values()),
        }
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"[ok] wrote {args.json}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
