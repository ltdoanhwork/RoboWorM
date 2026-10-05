#!/usr/bin/env python3
"""Aggregate per-episode multiview-consistency CSVs into a variant comparison.

``metrics/rigis/run.sh`` writes one CSV per episodes-root, with
columns ``name,geometric_consistency,pose_stability,num_disjoint_frames,
disjoint_frame_ratio,num_frames,time_seconds`` (schema:
``multiview_consistency/run_video_metrics.py::_write_metrics_csv``).

This script pools those rows per variant and prints a leaderboard. Both metrics
are in [0, 1] and HIGHER IS BETTER.

Usage::

    python metrics/rigis/aggregate.py \
        --results-root results/multiview_consistency \
        --variants dense pisa radial svg1 svg2

    # restrict to a subtree, and dump json
    python metrics/rigis/aggregate.py \
        --results-root results/multiview_consistency \
        --variants dense pisa --filter bimanual/ctrlworld \
        --json results/multiview_consistency/summary.json

By default only episodes present in EVERY requested variant are compared
(``--intersect``), so the leaderboard is apples-to-apples. Use ``--no-intersect``
to pool whatever each variant has.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = ("geometric_consistency", "pose_stability")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", required=True, help="Root containing <variant>/**/*.csv")
    p.add_argument("--variants", nargs="+", required=True, help="Variant dir names under --results-root")
    p.add_argument("--filter", default="", help="Only CSVs whose relative path contains this substring")
    p.add_argument("--baseline", default=None, help="Variant to report deltas against (default: first)")
    p.add_argument("--intersect", action=argparse.BooleanOptionalAction, default=True,
                   help="Compare only episodes present in every variant (default: on)")
    p.add_argument("--json", default=None, help="Also write the summary as JSON here")
    return p.parse_args()


def _load_variant(root: Path, variant: str, needle: str) -> dict[str, dict[str, float]]:
    """-> {episode_key: {metric: value}}; episode_key = <relative csv path>::<name>."""
    vdir = root / variant
    rows: dict[str, dict[str, float]] = {}
    if not vdir.is_dir():
        return rows
    for csv_path in sorted(vdir.rglob("*.csv")):
        rel = csv_path.relative_to(vdir).as_posix()
        if needle and needle not in rel:
            continue
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                name = (row.get("name") or "").strip()
                if not name:
                    continue
                vals: dict[str, float] = {}
                ok = True
                for m in METRICS:
                    try:
                        v = float(row[m])
                    except (TypeError, ValueError, KeyError):
                        ok = False
                        break
                    if not math.isfinite(v):
                        ok = False
                        break
                    vals[m] = v
                if not ok:
                    continue
                for extra in ("disjoint_frame_ratio", "num_frames", "time_seconds"):
                    try:
                        vals[extra] = float(row[extra])
                    except (TypeError, ValueError, KeyError):
                        pass
                rows[f"{rel}::{name}"] = vals
    return rows


def _stats(values: list[float]) -> dict[str, float]:
    n = len(values)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": sd,
        "sem": sd / math.sqrt(n) if n else 0.0,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    args = _parse_args()
    root = Path(args.results_root).resolve()

    per_variant = {v: _load_variant(root, v, args.filter) for v in args.variants}
    present = [v for v, rows in per_variant.items() if rows]
    missing = [v for v in args.variants if not per_variant[v]]
    if not present:
        raise SystemExit(f"no CSV rows found under {root} for variants={args.variants}")

    keys: set[str] | None = None
    if args.intersect:
        for v in present:
            ks = set(per_variant[v])
            keys = ks if keys is None else (keys & ks)
        if not keys:
            raise SystemExit(
                "no episodes are shared across all variants; rerun with --no-intersect "
                "or score the same episode set for every variant"
            )

    summary: dict[str, dict] = {}
    for v in present:
        rows = per_variant[v]
        sel = sorted(keys) if keys is not None else sorted(rows)
        out: dict[str, dict] = {}
        for m in METRICS:
            out[m] = _stats([rows[k][m] for k in sel if m in rows[k]])
        dj = [rows[k]["disjoint_frame_ratio"] for k in sel if "disjoint_frame_ratio" in rows[k]]
        if dj:
            out["disjoint_frame_ratio"] = _stats(dj)
        summary[v] = out

    baseline = args.baseline or present[0]
    scope = "intersection" if keys is not None else "all rows per variant"
    n_eps = len(keys) if keys is not None else None

    print(f"results_root : {root}")
    print(f"filter       : {args.filter or '(none)'}")
    print(f"scope        : {scope}" + (f" ({n_eps} episodes)" if n_eps is not None else ""))
    print(f"baseline     : {baseline}")
    if missing:
        print(f"no data for  : {', '.join(missing)}")
    print("")
    print("Both metrics are in [0, 1]; HIGHER IS BETTER.")
    print("")
    header = (
        f"{'variant':<12} {'n':>5} "
        f"{'geom_cons':>10} {'±sem':>7} {'Δ':>8}   "
        f"{'pose_stab':>10} {'±sem':>7} {'Δ':>8}   {'disjoint':>8}"
    )
    print(header)
    print("-" * len(header))

    order = sorted(present, key=lambda v: -summary[v]["geometric_consistency"]["mean"])
    for v in order:
        s = summary[v]
        g, p = s["geometric_consistency"], s["pose_stability"]
        dg = g["mean"] - summary[baseline]["geometric_consistency"]["mean"]
        dp = p["mean"] - summary[baseline]["pose_stability"]["mean"]
        dj = s.get("disjoint_frame_ratio", {}).get("mean", float("nan"))
        tag = " *" if v == baseline else ""
        print(
            f"{v + tag:<12} {g['n']:>5} "
            f"{g['mean']:>10.4f} {g['sem']:>7.4f} {dg:>+8.4f}   "
            f"{p['mean']:>10.4f} {p['sem']:>7.4f} {dp:>+8.4f}   {dj:>8.4f}"
        )
    print("")
    print("* = baseline.  Δ = variant mean − baseline mean (positive = better).")
    print("disjoint = mean fraction of frames with <2% covisible pixels (scored 0).")

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "results_root": str(root),
            "filter": args.filter,
            "scope": scope,
            "num_episodes": n_eps,
            "baseline": baseline,
            "variants_requested": list(args.variants),
            "variants_with_data": present,
            "variants_missing": missing,
            "higher_is_better": True,
            "summary": summary,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
