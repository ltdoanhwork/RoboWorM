"""Run any set of registered metrics over a directory of generated videos.

Sample discovery (defaults match the ``sampling_dataset`` layout, where every
prediction sits next to its ground truth and prompt sidecar)::

    <dir>/episode_000000.mp4      pred      (--pred-glob, default **/*.mp4)
    <dir>/episode_000000_gt.mp4   ref       (--ref-template '{stem}_gt{suffix}')
    <dir>/episode_000000.txt      prompt    (--text-template '{stem}.txt', [Prompt] section)

Files whose stem ends in ``--exclude-suffix`` (default ``_gt``) are never
treated as predictions. A glob that matches directories yields directory
samples (multiview episode dirs with view1/ view2/).

Examples::

    PY=models/dreamgen/.venv/bin/python     # has piq + torchmetrics + CLIP

    # fidelity + CLIP on one cell
    $PY -m metrics.evaluate --pred-root sampling_dataset/dense/single_arm/output/singleview/dreamgen/makovian \\
        --metrics fidelity clip_score --out results/metrics/dense_single_arm_mk

    # kinescore hallucination with the GT anchor in the same call
    $PY -m metrics.evaluate --pred-root <dir> --metrics hallucination \\
        --opt hallucination.embodiment=single_arm --opt hallucination.score_ref=true --out <out>

    # multiview consistency on extracted episode dirs
    $PY -m metrics.evaluate --pred-root sampling_dataset/multiview_consistency/frames/dense/<...> \\
        --pred-glob '*' --ref-template '' --metrics multiview_consistency --out <out>

Outputs in ``--out``: ``<metric>.json`` (full MetricResult), ``per_sample.csv``
(one row per sample, all metrics joined on id) and ``summary.json``.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from .base import MetricResult, Sample, _json_safe
from .registry import available, build_metric
from .video_io import read_prompt


def discover_samples(pred_root: str | Path, pred_glob: str = "**/*.mp4",
                     ref_template: str = "{stem}_gt{suffix}", ref_root: str | None = None,
                     text_template: str = "{stem}.txt", prompt: str | None = None,
                     exclude_suffix: str = "_gt", limit: int = 0) -> list[Sample]:
    root = Path(pred_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"--pred-root is not a directory: {root}")
    samples: list[Sample] = []
    for p in sorted(root.glob(pred_glob)):
        stem = p.stem if p.is_file() else p.name
        if exclude_suffix and stem.endswith(exclude_suffix):
            continue
        if p.is_file() and p.suffix.lower() == ".txt":
            continue
        rel = p.relative_to(root)
        sid = str(rel.with_suffix("")) if p.is_file() else str(rel)
        fmt = {"stem": stem, "name": p.name, "suffix": p.suffix}

        ref = None
        if ref_template:
            base = (Path(ref_root).resolve() / rel.parent) if ref_root else p.parent
            cand = base / ref_template.format(**fmt)
            ref = cand if cand.exists() else None

        text = prompt
        if text is None and text_template:
            text = read_prompt(p.parent / text_template.format(**fmt))

        samples.append(Sample(id=sid, pred=p, ref=ref, text=text))
        if limit and len(samples) >= limit:
            break
    return samples


def _parse_opts(items: list[str]) -> dict[str, dict[str, Any]]:
    """``metric.key=value`` -> {metric: {key: value}}; value parsed as JSON if possible."""
    out: dict[str, dict[str, Any]] = {}
    for it in items:
        if "=" not in it or "." not in it.split("=", 1)[0]:
            raise SystemExit(f"--opt expects metric.key=value, got {it!r}")
        lhs, val = it.split("=", 1)
        metric, key = lhs.split(".", 1)
        try:
            parsed = json.loads(val)
        except json.JSONDecodeError:
            parsed = val
        out.setdefault(metric, {})[key] = parsed
    return out


def write_outputs(results: list[MetricResult], samples: list[Sample], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for r in results:
        r.save(out / f"{r.metric}.json")

    cols: list[str] = []
    for r in results:
        keys = sorted({k for row in r.per_sample.values() for k in row})
        cols += [f"{r.metric}.{k}" for k in keys]
    with (out / "per_sample.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "pred", "ref", *cols, "errors"])
        for s in samples:
            row: list[Any] = [s.id, str(s.pred), "" if s.ref is None else str(s.ref)]
            for r in results:
                keys = sorted({k for rr in r.per_sample.values() for k in rr})
                vals = r.per_sample.get(s.id, {})
                row += ["" if k not in vals else f"{vals[k]:.6g}" for k in keys]
            errs = "; ".join(f"{r.metric}: {r.errors[s.id]}" for r in results if s.id in r.errors)
            w.writerow(row + [errs])

    summary = {r.metric: {"summary": r.summary, "higher_is_better": r.higher_is_better,
                          "n_errors": len(r.errors), "seconds": r.seconds, "config": r.config}
               for r in results}
    (out / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--pred-glob", default="**/*.mp4")
    ap.add_argument("--ref-template", default="{stem}_gt{suffix}",
                    help="ref filename next to each pred ('' = no reference)")
    ap.add_argument("--ref-root", default=None,
                    help="look refs up under this root (same relative dir) instead of next to pred")
    ap.add_argument("--text-template", default="{stem}.txt")
    ap.add_argument("--prompt", default=None, help="one prompt for every sample (overrides sidecars)")
    ap.add_argument("--exclude-suffix", default="_gt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--metrics", nargs="+", required=True, help=f"any of: {' '.join(available())}")
    ap.add_argument("--opt", action="append", default=[], metavar="METRIC.KEY=VALUE",
                    help="per-metric constructor kwarg, e.g. hallucination.embodiment=bimanual")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fail-fast", action="store_true")
    args = ap.parse_args(argv)

    samples = discover_samples(args.pred_root, args.pred_glob, args.ref_template, args.ref_root,
                               args.text_template, args.prompt, args.exclude_suffix, args.limit)
    if not samples:
        print(f"no samples matched {args.pred_glob!r} under {args.pred_root}", file=sys.stderr)
        return 1
    n_ref = sum(s.ref is not None for s in samples)
    n_txt = sum(bool(s.text) for s in samples)
    print(f"[evaluate] {len(samples)} samples ({n_ref} with ref, {n_txt} with prompt)")

    opts = _parse_opts(args.opt)
    results: list[MetricResult] = []
    for name in args.metrics:
        metric = build_metric(name, device=args.device, **opts.get(name, {}))
        print(f"[evaluate] === {metric!r}")
        results.append(metric.evaluate(samples, fail_fast=args.fail_fast))
        write_outputs(results, samples, Path(args.out))  # incremental: survive a later crash

    print("\n" + "\n".join(r.headline() for r in results))
    print(f"[evaluate] wrote {args.out}/{{summary.json,per_sample.csv,<metric>.json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
