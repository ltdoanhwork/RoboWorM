#!/usr/bin/env python3
"""Aggregate kinescore hallucination scores into a variant comparison.

``metrics/hallucination/run.sh`` writes one output dir per
``(role, variant, split, embodiment)`` containing ``segments.csv``,
``metrics.csv`` and ``summary.json`` (schema:
``kinescore/violations/table.py::write_segment_table`` /
``write_table``).

This script pools those rows per variant and prints a leaderboard. The headline
numbers are the two detectors kinescore declares in
``violations/scorer.py::HEADLINE``:

    rigidity  (mm,     median over a 16-frame segment)
    jerk      (mm/s^3, worst frame of a 16-frame segment)

and are reported as **segment violation rate** = the fraction of 16-frame
segments whose reduced value exceeds the segment threshold, plus the
**severity ratio** = value / segment_threshold.

LOWER IS BETTER for every number this script prints: they are violation rates
and violation severities, not quality scores.

Read a variant's rate as its EXCESS over the real-video anchor's rate, never
against zero -- the reader's own keypoint error puts a nonzero floor under
every rate (``docs/hallucination_metric.md`` §7).

Usage::

    python metrics/hallucination/aggregate.py \
        --results-root results/hallucination \
        --variants dense pisa radial svg1 svg2

    # one embodiment, deltas against the real-video anchor, dump json
    python metrics/hallucination/aggregate.py \
        --results-root results/hallucination \
        --variants dense pisa --embodiment single_arm \
        --anchor-role gt --baseline real \
        --json results/hallucination/summary.json

By default only clips present in EVERY requested variant are compared
(``--intersect``), so the leaderboard is apples-to-apples. Use
``--no-intersect`` to pool whatever each variant has.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

#: ``violations/scorer.py::HEADLINE`` -- the detectors a benchmark number is
#: read off. The other three (teleport, joint_limit, self_collision) are
#: computed and stored, and surface via ``--detectors``.
HEADLINE = ("rigidity", "jerk")
ALL_DETECTORS = ("rigidity", "jerk", "teleport", "joint_limit",
                 "self_collision")

#: Detectors where a SMALLER per-frame value is the violation
#: (``Detector.higher_is_worse = False``). Only self_collision, whose score is
#: the minimum inter-keypoint distance.
LOWER_IS_WORSE = frozenset({"self_collision"})

#: ``<variant>.<split>.<tag>`` -- the dir name hallucination/run.sh emits.
_DIR_RE = re.compile(r"^(?P<variant>[^.]+)\.(?P<split>makovian|non_makovian)\."
                     r"(?P<tag>.+)$")

#: output-dir tag -> embodiment
_TAG_EMBODIMENT = {
    "sv1_4x3.a1x": "single_arm",
    "bimanual_sv16.aloha": "bimanual",
    "humanoid_sv16.gr1": "humanoid",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", required=True,
                   help="Root containing <role>/<variant>.<split>.<tag>/segments.csv")
    p.add_argument("--variants", nargs="+", required=True,
                   help="Variant names to compare")
    p.add_argument("--role", default="pred",
                   help="Which role subdir the variants are read from (default: pred)")
    p.add_argument("--anchor-role", default=None,
                   help="Also load this role (e.g. 'gt') and report it as the "
                        "variant named by --anchor-name. This is the real-video "
                        "anchor; without it the rates are uninterpretable.")
    p.add_argument("--anchor-name", default="real",
                   help="Row label for --anchor-role (default: real)")
    p.add_argument("--embodiment", default="",
                   choices=["", "single_arm", "bimanual", "humanoid"],
                   help="Restrict to one embodiment (default: pool all present). "
                        "Pooling mixes three different readers, so prefer one.")
    p.add_argument("--split", default="", choices=["", "makovian", "non_makovian"],
                   help="Restrict to one split (default: both)")
    p.add_argument("--detectors", nargs="+", default=list(HEADLINE),
                   help=f"Detectors to report (default: {' '.join(HEADLINE)}; "
                        f"available: {' '.join(ALL_DETECTORS)})")
    p.add_argument("--baseline", default=None,
                   help="Variant to report deltas against (default: the anchor "
                        "if loaded, else the first variant with data)")
    p.add_argument("--intersect", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Compare only clips present in every variant (default: on)")
    p.add_argument("--json", default=None, help="Also write the summary as JSON here")
    return p.parse_args()


def _clip_dirs(root: Path, role: str, variant: str, embodiment: str,
               split: str) -> list[tuple[Path, str, str]]:
    """-> [(dir, embodiment, split)] for one variant under one role."""
    base = root / role
    out = []
    if not base.is_dir():
        return out
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        m = _DIR_RE.match(d.name)
        if not m or m["variant"] != variant:
            continue
        emb = _TAG_EMBODIMENT.get(m["tag"], m["tag"])
        if embodiment and emb != embodiment:
            continue
        if split and m["split"] != split:
            continue
        out.append((d, emb, m["split"]))
    return out


def _float(row: dict, key: str) -> float | None:
    try:
        v = float(row[key])
    except (TypeError, ValueError, KeyError):
        return None
    return v if math.isfinite(v) else None


def _load_segments(d: Path, detectors: list[str]) -> dict[str, dict[str, list]]:
    """-> {clip_id: {detector: [(value, threshold, violated), ...]}} from segments.csv."""
    path = d / "segments.csv"
    if not path.is_file():
        return {}
    clips: dict[str, dict[str, list]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            clip = (row.get("id") or "").strip()
            if not clip:
                continue
            per = clips.setdefault(clip, {n: [] for n in detectors})
            for name in detectors:
                value = _float(row, f"{name}_value")
                thr = _float(row, f"{name}_threshold")
                if value is None or thr is None or thr == 0.0:
                    continue
                violated = (row.get(f"{name}_violated") or "").strip() == "True"
                per[name].append((value, thr, violated))
    return clips


def _per_clip(clips: dict[str, dict[str, list]], detectors: list[str]
              ) -> dict[str, dict[str, float]]:
    """Reduce each clip's segments to one number per detector."""
    out: dict[str, dict[str, float]] = {}
    for clip, per in clips.items():
        vals: dict[str, float] = {}
        for name in detectors:
            segs = per.get(name) or []
            if not segs:
                continue
            # violation rate: fraction of this clip's segments flagged
            vals[f"{name}_rate"] = (sum(1 for _, _, v in segs if v)
                                    / len(segs))
            # severity: value / segment_threshold, inverted where smaller is
            # worse, so > 1 always means "worse than the real-motion p95".
            ratios = [(t / v if name in LOWER_IS_WORSE else v / t)
                      for v, t, _ in segs if v > 0]
            if ratios:
                vals[f"{name}_severity"] = statistics.median(ratios)
            vals[f"{name}_n_segments"] = float(len(segs))
        if vals:
            out[clip] = vals
    return out


def _load_variant(root: Path, role: str, variant: str, embodiment: str,
                  split: str, detectors: list[str]
                  ) -> tuple[dict[str, dict[str, float]], dict]:
    rows: dict[str, dict[str, float]] = {}
    meta: dict = {"dirs": [], "readers": {}, "thresholds": {}}
    for d, emb, spl in _clip_dirs(root, role, variant, embodiment, split):
        clips = _per_clip(_load_segments(d, detectors), detectors)
        for clip, vals in clips.items():
            rows[f"{emb}/{spl}/{clip}"] = vals
        meta["dirs"].append(str(d))
        summary = d / "summary.json"
        if summary.is_file():
            try:
                s = json.loads(summary.read_text())
            except json.JSONDecodeError:
                continue
            meta["readers"][emb] = {
                "reader_id": s.get("reader_id"),
                "checkpoint": s.get("checkpoint"),
                "checkpoint_sha256": (s.get("checkpoint_sha256") or "")[:16],
                "n_calibration_clips": s.get("n_calibration_clips"),
                "percentile": s.get("percentile"),
            }
            meta["thresholds"][f"{emb}/{spl}"] = {
                n: t.get("segment_threshold")
                for n, t in (s.get("thresholds") or {}).items()
            }
    return rows, meta


def _stats(values: list[float]) -> dict[str, float]:
    n = len(values)
    if not n:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "sem": float("nan"), "median": float("nan"),
                "min": float("nan"), "max": float("nan")}
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    return {"n": n, "mean": mean, "std": sd,
            "sem": sd / math.sqrt(n) if n else 0.0,
            "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def main() -> None:
    args = _parse_args()
    root = Path(args.results_root).resolve()
    detectors = [d for d in args.detectors if d in ALL_DETECTORS]
    if not detectors:
        raise SystemExit(f"--detectors must name some of {ALL_DETECTORS}")

    per_variant: dict[str, dict[str, dict[str, float]]] = {}
    meta: dict[str, dict] = {}
    for v in args.variants:
        rows, m = _load_variant(root, args.role, v, args.embodiment,
                                args.split, detectors)
        per_variant[v], meta[v] = rows, m

    anchor = None
    if args.anchor_role:
        anchor = args.anchor_name
        for v in args.variants:
            rows, m = _load_variant(root, args.anchor_role, v, args.embodiment,
                                    args.split, detectors)
            for clip, vals in rows.items():
                per_variant.setdefault(anchor, {})[f"{v}/{clip}"] = vals
            if rows:
                meta.setdefault(anchor, {"dirs": [], "readers": {},
                                         "thresholds": {}})
                meta[anchor]["dirs"].extend(m["dirs"])
                meta[anchor]["readers"].update(m["readers"])
                meta[anchor]["thresholds"].update(m["thresholds"])

    present = [v for v, rows in per_variant.items() if rows]
    missing = [v for v in args.variants if not per_variant.get(v)]
    if not present:
        raise SystemExit(
            f"no segment rows found under {root}/{args.role} for "
            f"variants={args.variants} -- run metrics/hallucination/run.sh first")

    # The anchor is keyed by <variant>/<clip>, so it never intersects with the
    # prediction rows; intersect over the predictions only.
    pred_present = [v for v in present if v != anchor]
    keys: set[str] | None = None
    if args.intersect and len(pred_present) > 1:
        for v in pred_present:
            ks = set(per_variant[v])
            keys = ks if keys is None else (keys & ks)
        if not keys:
            raise SystemExit(
                "no clips are shared across all variants; rerun with "
                "--no-intersect or score the same clip set for every variant")

    summary: dict[str, dict] = {}
    for v in present:
        rows = per_variant[v]
        sel = (sorted(keys & set(rows)) if keys is not None and v != anchor
               else sorted(rows))
        out: dict[str, dict] = {}
        for name in detectors:
            for kind in ("rate", "severity"):
                key = f"{name}_{kind}"
                out[key] = _stats([rows[k][key] for k in sel if key in rows[k]])
        out["n_segments_total"] = sum(
            rows[k].get(f"{detectors[0]}_n_segments", 0.0) for k in sel)
        summary[v] = out

    baseline = args.baseline or (anchor if anchor in present else pred_present[0]
                                 if pred_present else present[0])
    if baseline not in summary:
        raise SystemExit(f"--baseline {baseline!r} has no data "
                         f"(have: {', '.join(present)})")

    scope = "intersection" if keys is not None else "all clips per variant"
    print(f"results_root : {root}")
    print(f"role         : {args.role}"
          + (f"   anchor role: {args.anchor_role} (as '{anchor}')" if anchor else ""))
    print(f"embodiment   : {args.embodiment or '(pooled -- mixes readers!)'}")
    print(f"split        : {args.split or '(both)'}")
    print(f"scope        : {scope}"
          + (f" ({len(keys)} clips)" if keys is not None else ""))
    print(f"baseline     : {baseline}")
    if missing:
        print(f"no data for  : {', '.join(missing)}")
    print("")
    print("rate = fraction of 16-frame segments flagged;  severity = median "
          "value / segment_threshold.")
    print("LOWER IS BETTER for every column. Read a rate as its EXCESS over "
          "the real-video anchor.")
    print("")

    cols = [f"{'variant':<14}", f"{'clips':>6}", f"{'segs':>7}"]
    for name in detectors:
        cols += [f"{name[:9] + '_rate':>15}", f"{'±sem':>7}", f"{'Δ':>8}",
                 f"{name[:9] + '_sev':>14}"]
    header = " ".join(cols)
    print(header)
    print("-" * len(header))

    key0 = f"{detectors[0]}_rate"
    order = sorted(present, key=lambda v: (summary[v][key0]["mean"]
                                           if summary[v][key0]["n"] else 9e9))
    for v in order:
        s = summary[v]
        tag = " *" if v == baseline else ""
        line = [f"{v + tag:<14}", f"{s[key0]['n']:>6}",
                f"{int(s['n_segments_total']):>7}"]
        for name in detectors:
            r, sv = s[f"{name}_rate"], s[f"{name}_severity"]
            d = r["mean"] - summary[baseline][f"{name}_rate"]["mean"]
            line += [f"{r['mean']:>15.4f}", f"{r['sem']:>7.4f}",
                     f"{d:>+8.4f}", f"{sv['mean']:>14.4f}"]
        print(" ".join(line))
    print("")
    print("* = baseline.  Δ = variant mean − baseline mean "
          "(NEGATIVE = fewer violations = better).")

    for v in present:
        readers = (meta.get(v) or {}).get("readers") or {}
        for emb, r in sorted(readers.items()):
            print(f"  reader[{v}/{emb}]: {r['reader_id']} "
                  f"sha {r['checkpoint_sha256']} "
                  f"(p{r['percentile']} of {r['n_calibration_clips']} real clips)")

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "results_root": str(root),
            "role": args.role,
            "anchor_role": args.anchor_role,
            "anchor_name": anchor,
            "embodiment": args.embodiment,
            "split": args.split,
            "detectors": detectors,
            "scope": scope,
            "num_clips_intersected": len(keys) if keys is not None else None,
            "baseline": baseline,
            "variants_requested": list(args.variants),
            "variants_with_data": present,
            "variants_missing": missing,
            "lower_is_better": True,
            "summary": summary,
            "provenance": meta,
        }, indent=2))
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
