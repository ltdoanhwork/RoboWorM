"""``kinescore score``: rigidity and jerk violation rates for a directory of clips."""
from __future__ import annotations

import argparse

NAME = "score"
HELP = "rigidity/jerk violation rates of generated clips, against a thresholds file"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    from kinescore.cli._model import add_read_arguments

    add_read_arguments(parser)
    parser.add_argument("--threshold", default=None,
                        help="thresholds file written by `kinescore calibrate`")
    parser.add_argument("--rigidity-threshold", type=float, default=None,
                        help="rigidity threshold in mm (overrides --threshold)")
    parser.add_argument("--jerk-threshold", type=float, default=None,
                        help="jerk threshold in mm/s^3 (overrides --threshold)")
    parser.add_argument("--videos", required=True,
                        help="directory of clips to score, searched recursively")
    parser.add_argument("--out", default=None,
                        help="also write summary.json, metrics.csv (per clip) and "
                             "segments.csv (per 16-frame segment) here")


def _clip_id(path: str, root: str) -> str:
    """Unique id for a clip: its path below the root, no suffix."""
    import os

    rel = os.path.relpath(path, root)
    return os.path.splitext(rel)[0].replace(os.sep, "__")


def _thresholds(args: argparse.Namespace) -> dict:
    """``{detector: {threshold, segment_threshold}}`` from the file and the flags."""
    import json

    from kinescore.cli._model import DETECTOR_NAMES

    fitted = {}
    if args.threshold:
        with open(args.threshold) as f:
            fitted = json.load(f)["thresholds"]
    given = {"rigidity": args.rigidity_threshold, "jerk": args.jerk_threshold}
    for name, value in given.items():
        if value is not None:
            fitted[name] = {"threshold": value, "segment_threshold": value}
    missing = [n for n in DETECTOR_NAMES if n not in fitted]
    if missing:
        raise SystemExit(f"no threshold for {', '.join(missing)}: pass --threshold <file> "
                         "or --rigidity-threshold and --jerk-threshold")
    return fitted


def run(args: argparse.Namespace) -> int:
    import json
    import os
    import pathlib

    from kinescore.cli._model import DETECTOR_NAMES, build_scorer, find_clips, load_model
    from kinescore.violations import segments
    from kinescore.violations.table import write_segment_table, write_table

    fitted = _thresholds(args)
    clips = find_clips(args.videos, args.pattern, args.limit)
    if not clips:
        raise SystemExit(f"no {args.pattern} under {args.videos}")

    model = load_model(args.model, args.device)
    scorer = build_scorer()
    units = {}
    for det in scorer.detectors:
        t = fitted[det.name]
        det.threshold, det.segment_threshold = t["threshold"], t["segment_threshold"]
        units[det.name] = det.units

    rows = []
    for i, path in enumerate(clips, 1):
        row = {"id": _clip_id(path, args.videos), "source_path": path}
        try:
            row["violations"] = scorer.score(
                model.read(path, args.frame_chunk, args.max_frames, args.seed))
            row["segments"] = segments.report(row["violations"], DETECTOR_NAMES)
            print(f"[score] {i}/{len(clips)} {os.path.basename(path)}")
        except Exception as exc:  # noqa: BLE001 -- recorded per clip, the run goes on
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[score] failed  {path}: {exc}")
        rows.append(row)

    scored = [r for r in rows if r.get("segments")]
    n_seg = sum(len(r["segments"]) for r in scored)
    scores = {}
    for name in DETECTOR_NAMES:
        flagged = sum(seg["detectors"][name]["violated"]
                      for r in scored for seg in r["segments"] if name in seg["detectors"])
        scores[name] = {"rate": flagged / n_seg if n_seg else None,
                        "n_violated": flagged, "n_segments": n_seg,
                        "segment_threshold": fitted[name]["segment_threshold"],
                        "units": units[name]}

    failed = len(rows) - len(scored)
    print(f"[score] {len(scored)} clips, {n_seg} segments"
          + (f", {failed} failed" if failed else ""))
    for name, s in scores.items():
        if s["rate"] is not None:
            print(f"[score] {name:9s} {100 * s['rate']:5.1f}%   ({s['n_violated']}/{n_seg} "
                  f"segments > {s['segment_threshold']:g} {s['units']})")

    if args.out:
        out = pathlib.Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "results.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        summary = {"model": os.path.abspath(args.model),
                   "threshold_file": args.threshold and os.path.abspath(args.threshold),
                   "videos": os.path.abspath(args.videos), "n_clips": len(rows),
                   "n_failed": failed, "scores": scores, "thresholds": fitted}
        with open(out / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        write_table(rows, out / "metrics.csv", DETECTOR_NAMES)
        write_segment_table(scored, out / "segments.csv", DETECTOR_NAMES)
        print(f"[score] wrote {out}")
    return 0 if failed == 0 else 1
